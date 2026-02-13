"""
논문 작성 에이전트 (Paper Writing Agent)
근거를 기반으로 학술 논문 섹션을 생성합니다.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from core.state_manager import (
    PaperDraft,
    PaperSection,
    ResearchEvidence,
    PaperReference,
)
from agents.tools import CitationManager, ContentAnalyzer

logger = logging.getLogger(__name__)

WRITING_SYSTEM_PROMPT = """You are an expert academic paper writer with extensive experience in writing IEEE and ACM conference/journal papers.

Your writing must follow these principles:
1. Academic tone - formal, precise, objective
2. Evidence-based claims - every claim should be supported
3. Logical flow - clear transitions between ideas
4. Proper citation integration - seamlessly reference source materials
5. Technical accuracy - use correct terminology

Paper Format: {paper_format}
Language: {language}

When writing each section, follow the specific guidelines for that section type.
Use citation keys (e.g., [1], [2]) when referring to source papers.
"""

SECTION_PROMPTS = {
    "Abstract": """Write a concise abstract (150-250 words) for the paper.
The abstract should:
- State the problem/motivation (1-2 sentences)
- Describe the approach/methodology (1-2 sentences)
- Summarize key results (1-2 sentences)
- State the significance/contribution (1 sentence)

Paper Title: {title}
Research Focus: {research_focus}
Key Evidence Summary: {evidence_summary}""",

    "Introduction": """Write the Introduction section (500-800 words).
Structure:
1. Background and context (general to specific)
2. Problem statement and motivation
3. Research gap identification
4. Contributions of this paper (enumerate)
5. Paper organization (brief overview of sections)

Paper Title: {title}
Research Focus: {research_focus}
Research Gaps: {research_gaps}
Key Evidence: {evidence_summary}
Available Citations:
{citations_info}""",

    "Related Work": """Write the Related Work section (600-1000 words).
Structure:
1. Organize by theme or chronologically
2. Critically analyze each group of works
3. Show evolution of ideas
4. Highlight differences from current work
5. Identify limitations of existing approaches

Paper Title: {title}
Research Focus: {research_focus}
Related Papers:
{papers_detail}""",

    "Methodology": """Write the Methodology/Proposed Method section (600-1000 words).
Structure:
1. Problem formulation (mathematical if applicable)
2. System architecture/design overview
3. Detailed description of each component
4. Algorithm or process flow
5. Theoretical analysis (if applicable)

Paper Title: {title}
Research Focus: {research_focus}
Evidence on existing methods:
{evidence_summary}""",

    "Proposed Method": """Write the Proposed Method section (600-1000 words).
Structure:
1. Overview of the proposed approach
2. System architecture
3. Key components and their interactions
4. Algorithm details
5. Complexity analysis (if applicable)

Paper Title: {title}
Research Focus: {research_focus}
Evidence on existing methods:
{evidence_summary}""",

    "System Model / Problem Formulation": """Write the System Model and Problem Formulation section (400-700 words).
Structure:
1. System model description
2. Assumptions and constraints
3. Mathematical formulation of the problem
4. Objective function
5. Key variables and notation

Paper Title: {title}
Research Focus: {research_focus}
Evidence: {evidence_summary}""",

    "Experiments": """Write the Experimental Setup section (400-700 words).
Structure:
1. Dataset/environment description
2. Implementation details
3. Baseline methods for comparison
4. Evaluation metrics
5. Experimental parameters and configurations

Paper Title: {title}
Research Focus: {research_focus}
Methodologies from related work:
{evidence_summary}""",

    "Experimental Setup": """Write the Experimental Setup section (400-700 words).
Structure:
1. Dataset/environment description
2. Implementation details
3. Baseline methods for comparison
4. Evaluation metrics
5. Experimental parameters

Paper Title: {title}
Research Focus: {research_focus}
Methodologies from related work:
{evidence_summary}""",

    "Results and Discussion": """Write the Results and Discussion section (500-800 words).
Structure:
1. Main results presentation
2. Comparison with baselines
3. Analysis of key findings
4. Discussion of strengths
5. Limitations and failure cases

Paper Title: {title}
Research Focus: {research_focus}
Evidence from related work:
{evidence_summary}""",

    "Evaluation": """Write the Evaluation section (500-800 words).
Structure:
1. Experimental results
2. Performance comparison
3. Ablation studies
4. Analysis and interpretation
5. Discussion of results

Paper Title: {title}
Research Focus: {research_focus}
Evidence: {evidence_summary}""",

    "Discussion": """Write the Discussion section (300-500 words).
Structure:
1. Summary of key findings
2. Implications of results
3. Comparison with prior work
4. Limitations
5. Broader impact

Paper Title: {title}
Research Focus: {research_focus}
Evidence: {evidence_summary}""",

    "Conclusion": """Write the Conclusion section (200-400 words).
Structure:
1. Brief summary of the paper
2. Main contributions (reiterate)
3. Key findings
4. Future work directions (2-3 specific directions)

Paper Title: {title}
Research Focus: {research_focus}
Research Gaps for future work: {research_gaps}
Evidence Summary: {evidence_summary}""",

    "Background and Related Work": """Write the Background and Related Work section (700-1100 words).
Structure:
1. Background concepts and definitions
2. Related work organized thematically
3. Critical analysis of existing approaches
4. Research gap identification

Paper Title: {title}
Research Focus: {research_focus}
Related Papers:
{papers_detail}""",

    "Design and Implementation": """Write the Design and Implementation section (600-900 words).
Structure:
1. Design goals and principles
2. System architecture
3. Key components
4. Implementation details
5. Technical challenges and solutions

Paper Title: {title}
Research Focus: {research_focus}
Evidence: {evidence_summary}""",
}


class WritingAgent:
    """논문 작성 에이전트"""

    def __init__(
        self,
        llm: BaseChatModel,
        paper_format: str = "ieee",
        language: str = "en",
    ):
        self.llm = llm
        self.paper_format = paper_format
        self.language = language
        self.citation_manager = CitationManager(style=paper_format)

    async def write_paper(
        self,
        title: str,
        research_focus: str,
        evidence: ResearchEvidence,
        target_sections: list[str],
    ) -> PaperDraft:
        """
        논문 전체를 섹션별로 작성합니다.

        Args:
            title: 논문 제목
            research_focus: 연구 초점
            evidence: 수집된 근거
            target_sections: 작성할 섹션 목록

        Returns:
            PaperDraft 결과
        """
        logger.info(f"논문 작성 시작: {title}")

        # 인용 등록
        for paper in evidence.get("relevant_papers", []):
            self.citation_manager.add_paper(
                {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", ""),
                }
            )

        # Abstract는 마지막에 작성
        sections_order = [s for s in target_sections if s != "Abstract"]
        if "Abstract" in target_sections:
            sections_order.append("Abstract")

        written_sections: list[PaperSection] = []
        previous_content = ""

        for section_name in sections_order:
            logger.info(f"섹션 작성 중: {section_name}")

            section = await self._write_section(
                section_name=section_name,
                title=title,
                research_focus=research_focus,
                evidence=evidence,
                previous_content=previous_content,
            )

            written_sections.append(section)
            previous_content += f"\n\n## {section_name}\n{section['content']}"

        # 참고문헌 생성
        references = self.citation_manager.get_all_references()

        # 전체 단어 수
        total_words = sum(s["word_count"] for s in written_sections)

        draft = PaperDraft(
            sections=written_sections,
            references=references,
            total_word_count=total_words,
        )

        logger.info(
            f"논문 작성 완료: {len(written_sections)}개 섹션, "
            f"{total_words} 단어"
        )
        return draft

    async def write_single_section(
        self,
        section_name: str,
        title: str,
        research_focus: str,
        evidence: ResearchEvidence,
        previous_content: str = "",
    ) -> PaperSection:
        """단일 섹션을 작성합니다."""
        return await self._write_section(
            section_name, title, research_focus, evidence, previous_content
        )

    async def _write_section(
        self,
        section_name: str,
        title: str,
        research_focus: str,
        evidence: ResearchEvidence,
        previous_content: str = "",
    ) -> PaperSection:
        """개별 섹션 작성"""

        # 프롬프트 준비
        section_prompt = self._build_section_prompt(
            section_name, title, research_focus, evidence
        )

        system_prompt = WRITING_SYSTEM_PROMPT.format(
            paper_format=self.paper_format.upper(),
            language="English" if self.language == "en" else "Korean",
        )

        # 이전 내용 컨텍스트 추가
        context_note = ""
        if previous_content:
            # 마지막 500자만
            context_note = (
                f"\n\nPreviously written sections (for context and consistency):\n"
                f"...{previous_content[-1000:]}"
            )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=section_prompt + context_note
                    ),
                ]
            )

            content = response.content.strip()

            # 마크다운 코드블록 제거
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            # 인용 키 추출
            import re
            citations_used = re.findall(r"\[\d+\]", content)
            citations_used = list(set(citations_used))

            word_count = ContentAnalyzer.count_words(content)

            return PaperSection(
                name=section_name,
                content=content,
                word_count=word_count,
                citations_used=citations_used,
            )

        except Exception as e:
            logger.error(f"섹션 작성 실패 ({section_name}): {e}")
            return PaperSection(
                name=section_name,
                content=f"[ERROR: {section_name} 섹션 작성에 실패했습니다. 오류: {e}]",
                word_count=0,
                citations_used=[],
            )

    def _build_section_prompt(
        self,
        section_name: str,
        title: str,
        research_focus: str,
        evidence: ResearchEvidence,
    ) -> str:
        """섹션별 프롬프트 구성"""

        # 기본 프롬프트 템플릿 선택
        template = SECTION_PROMPTS.get(
            section_name,
            SECTION_PROMPTS.get(
                "Discussion",
                "Write the {section_name} section.\nTitle: {title}\nFocus: {research_focus}",
            ),
        )

        # 논문 상세 정보
        papers_detail = self._format_papers_detail(
            evidence.get("relevant_papers", [])
        )

        # 인용 정보
        citations_info = self._format_citations_info(
            evidence.get("relevant_papers", [])
        )

        # 연구 공백
        research_gaps = "\n".join(
            f"- {gap}" for gap in evidence.get("research_gaps", [])
        )

        # 프롬프트 포맷팅
        try:
            prompt = template.format(
                title=title,
                research_focus=research_focus,
                evidence_summary=evidence.get("evidence_summary", ""),
                research_gaps=research_gaps or "Not specified",
                papers_detail=papers_detail,
                citations_info=citations_info,
                section_name=section_name,
            )
        except KeyError:
            # 일부 변수가 없는 경우 간소화된 프롬프트
            prompt = (
                f"Write the {section_name} section for the paper:\n"
                f"Title: {title}\n"
                f"Research Focus: {research_focus}\n"
                f"Evidence: {evidence.get('evidence_summary', '')}\n"
            )

        return prompt

    def _format_papers_detail(
        self, papers: list[dict]
    ) -> str:
        """논문 상세 정보 포맷팅"""
        if not papers:
            return "No papers available."

        details = []
        for paper in papers:
            citation_key = paper.get("citation_key", "")
            details.append(
                f"{citation_key} {paper.get('title', 'Unknown')}\n"
                f"  Authors: {paper.get('authors', 'Unknown')}\n"
                f"  Key Findings: {paper.get('key_findings', 'N/A')}\n"
                f"  Methodology: {paper.get('methodology', 'N/A')}\n"
                f"  Relevance: {paper.get('relevance', 'N/A')}\n"
            )
        return "\n".join(details)

    def _format_citations_info(
        self, papers: list[dict]
    ) -> str:
        """인용 정보 요약"""
        if not papers:
            return "No citations available."

        info = []
        for paper in papers:
            key = paper.get("citation_key", "")
            title = paper.get("title", "Unknown")
            info.append(f"{key} - {title}")
        return "\n".join(info)

    async def revise_section(
        self,
        section: PaperSection,
        issues: list[dict],
        evidence: ResearchEvidence,
    ) -> PaperSection:
        """
        검증 에이전트의 피드백을 기반으로 섹션을 수정합니다.

        Args:
            section: 수정할 섹션
            issues: 발견된 이슈 목록
            evidence: 참고 근거

        Returns:
            수정된 PaperSection
        """
        issues_text = "\n".join(
            f"- [{issue.get('severity', 'medium')}] {issue.get('description', '')}\n"
            f"  제안: {issue.get('suggestion', '')}"
            for issue in issues
            if issue.get("section") == section["name"]
        )

        if not issues_text:
            return section  # 이슈가 없으면 원본 반환

        prompt = f"""Please revise the following paper section based on the identified issues.

Section: {section['name']}

Current Content:
{section['content']}

Issues Found:
{issues_text}

Please provide the revised version that addresses all the identified issues.
Maintain academic tone and proper citations.
Return only the revised content without any explanations."""

        system_prompt = WRITING_SYSTEM_PROMPT.format(
            paper_format=self.paper_format.upper(),
            language="English" if self.language == "en" else "Korean",
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                ]
            )

            revised_content = response.content.strip()
            word_count = ContentAnalyzer.count_words(revised_content)

            import re
            citations_used = list(
                set(re.findall(r"\[\d+\]", revised_content))
            )

            return PaperSection(
                name=section["name"],
                content=revised_content,
                word_count=word_count,
                citations_used=citations_used,
            )

        except Exception as e:
            logger.error(
                f"섹션 수정 실패 ({section['name']}): {e}"
            )
            return section
