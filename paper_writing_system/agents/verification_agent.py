"""
검증 에이전트 (Verification Agent)
작성된 논문 내용의 품질, 일관성, 인용 정확성을 검증합니다.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from core.state_manager import (
    VerificationResult,
    VerificationIssue,
    PaperDraft,
    ResearchEvidence,
)
from agents.tools import parse_llm_json_response

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """You are an expert academic paper reviewer with extensive experience reviewing for top-tier conferences and journals (IEEE, ACM).

Your task is to thoroughly review academic paper sections for:
1. **Logical Consistency**: Check for contradictions, unsupported claims, logical gaps
2. **Citation Accuracy**: Verify that citations are used correctly and support the claims made
3. **Academic Quality**: Assess writing clarity, structure, argument strength, and technical depth
4. **Plagiarism Risk**: Check for potential issues with unattributed content
5. **Format Compliance**: Verify adherence to IEEE/ACM formatting conventions

Be thorough but constructive. For each issue found, provide:
- The specific problem
- Its severity (high/medium/low)
- A concrete suggestion for improvement

Respond in JSON format as specified."""

VERIFICATION_PROMPT = """Review the following academic paper draft.

Paper Title: {title}
Paper Format: {paper_format}

--- PAPER SECTIONS ---
{paper_content}

--- REFERENCE EVIDENCE (from source papers) ---
{evidence_summary}

--- AVAILABLE REFERENCES ---
{references}

Please provide a comprehensive review in the following JSON format:
```json
{{
    "overall_score": 0.0,
    "consistency_score": 0.0,
    "citation_accuracy_score": 0.0,
    "quality_score": 0.0,
    "issues": [
        {{
            "section": "Section Name",
            "issue_type": "consistency|citation|quality|plagiarism|format",
            "severity": "high|medium|low",
            "description": "Specific description of the issue",
            "suggestion": "How to fix this issue"
        }}
    ],
    "improvement_suggestions": [
        "General suggestion 1",
        "General suggestion 2"
    ]
}}
```

Scoring guidelines (0.0 to 1.0):
- 0.8-1.0: Excellent, ready for submission
- 0.6-0.8: Good, minor revisions needed
- 0.4-0.6: Fair, significant revisions needed
- 0.0-0.4: Poor, major rewrite needed"""


class VerificationAgent:
    """검증 에이전트"""

    def __init__(
        self,
        llm: BaseChatModel,
        quality_threshold: float = 0.7,
    ):
        self.llm = llm
        self.quality_threshold = quality_threshold

    async def verify_paper(
        self,
        draft: PaperDraft,
        evidence: ResearchEvidence,
        title: str,
        paper_format: str = "ieee",
    ) -> VerificationResult:
        """
        논문 초안을 검증합니다.

        Args:
            draft: 논문 초안
            evidence: 수집된 근거
            title: 논문 제목
            paper_format: 논문 포맷

        Returns:
            VerificationResult 결과
        """
        logger.info("논문 검증 시작")

        # 논문 내용 포맷팅
        paper_content = self._format_paper_content(draft)
        references = "\n".join(draft.get("references", []))
        evidence_summary = evidence.get("evidence_summary", "")

        # 근거 논문 상세
        papers_info = []
        for p in evidence.get("relevant_papers", []):
            papers_info.append(
                f"- {p.get('citation_key', '')}: {p.get('title', '')} "
                f"(Key findings: {p.get('key_findings', 'N/A')})"
            )
        if papers_info:
            evidence_summary += "\n\nSource Papers:\n" + "\n".join(
                papers_info
            )

        prompt = VERIFICATION_PROMPT.format(
            title=title,
            paper_format=paper_format.upper(),
            paper_content=paper_content,
            evidence_summary=evidence_summary,
            references=references or "No references provided",
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            parsed = parse_llm_json_response(response.content)
            if parsed:
                return self._build_result(parsed)

        except Exception as e:
            logger.error(f"검증 실패: {e}")

        # 폴백: 기본 검증 수행
        return self._basic_verification(draft)

    def _format_paper_content(self, draft: PaperDraft) -> str:
        """논문 초안을 텍스트로 포맷팅"""
        content_parts = []
        for section in draft.get("sections", []):
            name = section.get("name", "Unknown Section")
            text = section.get("content", "")
            word_count = section.get("word_count", 0)
            content_parts.append(
                f"## {name} (Word count: {word_count})\n\n{text}\n"
            )
        return "\n".join(content_parts)

    def _build_result(self, parsed: dict) -> VerificationResult:
        """파싱된 LLM 응답을 VerificationResult로 변환"""
        overall_score = float(parsed.get("overall_score", 0.5))
        consistency_score = float(
            parsed.get("consistency_score", 0.5)
        )
        citation_score = float(
            parsed.get("citation_accuracy_score", 0.5)
        )
        quality_score = float(parsed.get("quality_score", 0.5))

        issues = []
        for issue_data in parsed.get("issues", []):
            issues.append(
                VerificationIssue(
                    section=issue_data.get("section", "General"),
                    issue_type=issue_data.get(
                        "issue_type", "quality"
                    ),
                    severity=issue_data.get("severity", "medium"),
                    description=issue_data.get("description", ""),
                    suggestion=issue_data.get("suggestion", ""),
                )
            )

        is_valid = overall_score >= self.quality_threshold

        return VerificationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            consistency_score=consistency_score,
            citation_accuracy_score=citation_score,
            quality_score=quality_score,
            issues=issues,
            improvement_suggestions=parsed.get(
                "improvement_suggestions", []
            ),
            revised_sections={},
        )

    def _basic_verification(
        self, draft: PaperDraft
    ) -> VerificationResult:
        """LLM 호출 없이 기본 검증 수행"""
        issues = []
        sections = draft.get("sections", [])

        for section in sections:
            name = section.get("name", "")
            content = section.get("content", "")
            word_count = section.get("word_count", 0)

            # 너무 짧은 섹션
            if word_count < 100 and name not in ["Abstract"]:
                issues.append(
                    VerificationIssue(
                        section=name,
                        issue_type="quality",
                        severity="high",
                        description=f"섹션이 너무 짧습니다 ({word_count} 단어).",
                        suggestion="섹션의 내용을 보충해야 합니다.",
                    )
                )

            # 에러 메시지 포함
            if "[ERROR" in content:
                issues.append(
                    VerificationIssue(
                        section=name,
                        issue_type="quality",
                        severity="high",
                        description="섹션에 오류 메시지가 포함되어 있습니다.",
                        suggestion="해당 섹션을 다시 생성해야 합니다.",
                    )
                )

            # 인용 없는 주요 섹션
            citations = section.get("citations_used", [])
            if (
                name in ["Introduction", "Related Work", "Background and Related Work"]
                and not citations
            ):
                issues.append(
                    VerificationIssue(
                        section=name,
                        issue_type="citation",
                        severity="medium",
                        description="이 섹션에 인용이 없습니다.",
                        suggestion="관련 논문 인용을 추가해야 합니다.",
                    )
                )

        high_issues = sum(
            1
            for i in issues
            if i.get("severity") == "high"
        )
        overall = max(0.0, 1.0 - high_issues * 0.2 - len(issues) * 0.05)

        return VerificationResult(
            is_valid=overall >= self.quality_threshold,
            overall_score=overall,
            consistency_score=0.5,
            citation_accuracy_score=0.5,
            quality_score=overall,
            issues=issues,
            improvement_suggestions=[
                "LLM 기반 심층 검증이 필요합니다."
            ],
            revised_sections={},
        )

    async def verify_section(
        self,
        section_name: str,
        section_content: str,
        evidence: ResearchEvidence,
    ) -> list[VerificationIssue]:
        """개별 섹션 검증"""
        prompt = f"""Review this specific section of an academic paper:

Section: {section_name}

Content:
{section_content}

Evidence Summary:
{evidence.get('evidence_summary', '')}

Identify any issues and provide them in JSON format:
```json
{{
    "issues": [
        {{
            "issue_type": "consistency|citation|quality|plagiarism|format",
            "severity": "high|medium|low",
            "description": "Description",
            "suggestion": "How to fix"
        }}
    ]
}}
```"""

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            parsed = parse_llm_json_response(response.content)
            if parsed:
                return [
                    VerificationIssue(
                        section=section_name,
                        issue_type=i.get("issue_type", "quality"),
                        severity=i.get("severity", "medium"),
                        description=i.get("description", ""),
                        suggestion=i.get("suggestion", ""),
                    )
                    for i in parsed.get("issues", [])
                ]
        except Exception as e:
            logger.error(f"섹션 검증 실패 ({section_name}): {e}")

        return []
