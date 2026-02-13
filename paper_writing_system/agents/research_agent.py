"""
근거 수집 에이전트 (Research Evidence Agent)
연구 주제에 관련된 논문을 검색하고 핵심 근거를 추출합니다.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from core.rag_engine import RAGEngine
from core.state_manager import ResearchEvidence, PaperReference
from agents.tools import format_search_results, parse_llm_json_response

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are an expert academic research analyst specializing in finding and synthesizing evidence from research papers.

Your task is to analyze retrieved research papers and extract structured evidence relevant to a given research topic.

Instructions:
1. Carefully analyze each provided paper excerpt
2. Identify key findings, methodologies, and results
3. Assess relevance to the research topic
4. Identify research gaps and opportunities
5. Provide structured output in the specified JSON format

Always respond in the language matching the research focus. If the research topic is in Korean, respond in Korean. If in English, respond in English.

Output format (JSON):
```json
{
    "relevant_papers": [
        {
            "title": "Paper Title",
            "authors": "Author names (if available)",
            "key_findings": "Main contributions and findings",
            "methodology": "Research methodology used",
            "relevance": "How this paper relates to the current research"
        }
    ],
    "evidence_summary": "Overall summary of the collected evidence",
    "research_gaps": ["Gap 1", "Gap 2", ...]
}
```"""

QUERY_GENERATION_PROMPT = """Based on the following research topic, generate 5-8 diverse search queries to find relevant academic papers.

Research Topic: {title}
Research Focus: {research_focus}
Keywords: {keywords}

Generate queries covering:
1. Core concepts and direct matches
2. Related terminology and synonyms
3. Specific methodologies
4. Application domains
5. Recent trends and state-of-the-art

Respond with a JSON array of query strings:
```json
["query1", "query2", "query3", ...]
```"""

EVIDENCE_EXTRACTION_PROMPT = """Research Topic: {title}
Research Focus: {research_focus}
Keywords: {keywords}

Below are excerpts from relevant research papers retrieved from the knowledge base.
Analyze them and extract structured evidence.

Retrieved Papers:
{search_results}

Please provide your analysis in the specified JSON format. Focus on:
1. Main contributions of each paper
2. Methodologies that could be relevant
3. Key experimental results
4. How each paper connects to our research
5. Identified gaps in current research"""


class ResearchAgent:
    """근거 수집 에이전트"""

    def __init__(
        self,
        llm: BaseChatModel,
        rag_engine: RAGEngine,
        top_k: int = 10,
    ):
        self.llm = llm
        self.rag_engine = rag_engine
        self.top_k = top_k

    async def collect_evidence(
        self,
        title: str,
        research_focus: str,
        keywords: list[str],
    ) -> ResearchEvidence:
        """
        연구 주제에 대한 근거를 수집합니다.

        Args:
            title: 논문 제목
            research_focus: 연구 초점
            keywords: 키워드 목록

        Returns:
            ResearchEvidence 결과
        """
        logger.info(f"근거 수집 시작: {title}")

        # 1단계: 검색 쿼리 생성
        search_queries = await self._generate_search_queries(
            title, research_focus, keywords
        )
        logger.info(f"생성된 검색 쿼리: {len(search_queries)}개")

        # 2단계: RAG 검색 수행
        all_results = []
        seen_sources = set()

        for query in search_queries:
            results = self.rag_engine.hybrid_search(
                query=query,
                keywords=keywords,
                top_k=self.top_k,
            )
            for r in results:
                source = r.get("source_file", "")
                if source not in seen_sources:
                    seen_sources.add(source)
                    all_results.append(r)

        # 상위 결과만 유지 (LLM 컨텍스트 제한)
        all_results = sorted(
            all_results, key=lambda x: x.get("score", 0), reverse=True
        )[:self.top_k * 2]

        logger.info(f"검색된 고유 논문: {len(all_results)}개")

        if not all_results:
            return ResearchEvidence(
                relevant_papers=[],
                evidence_summary="관련 논문을 찾을 수 없습니다.",
                research_gaps=["기존 연구가 부족하여 추가 문헌 조사가 필요합니다."],
                search_queries_used=search_queries,
                total_papers_searched=0,
            )

        # 3단계: LLM으로 근거 분석
        evidence = await self._analyze_evidence(
            title, research_focus, keywords, all_results
        )

        evidence["search_queries_used"] = search_queries
        evidence["total_papers_searched"] = len(all_results)

        return evidence

    async def _generate_search_queries(
        self,
        title: str,
        research_focus: str,
        keywords: list[str],
    ) -> list[str]:
        """LLM을 사용하여 다양한 검색 쿼리를 생성합니다."""
        prompt = QUERY_GENERATION_PROMPT.format(
            title=title,
            research_focus=research_focus,
            keywords=", ".join(keywords),
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content="You are a search query generator for academic research. Generate diverse, specific queries."),
                    HumanMessage(content=prompt),
                ]
            )

            queries = parse_llm_json_response(response.content)
            if isinstance(queries, list):
                return queries
        except Exception as e:
            logger.warning(f"쿼리 생성 실패: {e}")

        # 폴백: 기본 쿼리 생성
        fallback_queries = [
            title,
            research_focus,
            " ".join(keywords),
        ]
        for kw in keywords[:3]:
            fallback_queries.append(f"{kw} {keywords[0] if keywords else ''}")
        return fallback_queries

    async def _analyze_evidence(
        self,
        title: str,
        research_focus: str,
        keywords: list[str],
        search_results: list[dict],
    ) -> dict:
        """LLM을 사용하여 검색 결과를 분석합니다."""
        formatted_results = format_search_results(search_results)

        prompt = EVIDENCE_EXTRACTION_PROMPT.format(
            title=title,
            research_focus=research_focus,
            keywords=", ".join(keywords),
            search_results=formatted_results,
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            parsed = parse_llm_json_response(response.content)
            if parsed:
                # PaperReference 형식으로 변환
                papers = []
                for i, p in enumerate(
                    parsed.get("relevant_papers", []), 1
                ):
                    papers.append(
                        PaperReference(
                            title=p.get("title", "Unknown"),
                            authors=p.get("authors", "Unknown"),
                            source=self._find_source_path(
                                p.get("title", ""), search_results
                            ),
                            key_findings=p.get("key_findings", ""),
                            methodology=p.get("methodology", ""),
                            relevance=p.get("relevance", ""),
                            citation_key=f"[{i}]",
                        )
                    )

                return ResearchEvidence(
                    relevant_papers=papers,
                    evidence_summary=parsed.get(
                        "evidence_summary", ""
                    ),
                    research_gaps=parsed.get("research_gaps", []),
                    search_queries_used=[],
                    total_papers_searched=0,
                )

        except Exception as e:
            logger.error(f"근거 분석 실패: {e}")

        # 폴백 결과
        return ResearchEvidence(
            relevant_papers=[],
            evidence_summary="근거 분석에 실패했습니다. 수동 검토가 필요합니다.",
            research_gaps=[],
            search_queries_used=[],
            total_papers_searched=0,
        )

    def _find_source_path(
        self, title: str, results: list[dict]
    ) -> str:
        """검색 결과에서 제목에 해당하는 소스 경로를 찾습니다."""
        title_lower = title.lower()
        for r in results:
            meta_title = r.get("metadata", {}).get("title", "").lower()
            if title_lower in meta_title or meta_title in title_lower:
                return r.get("source_file", "")
        return ""
