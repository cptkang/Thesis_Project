"""
LangGraph 노드 구현
워크플로우의 각 단계를 노드 함수로 정의합니다.
"""

import logging
from typing import Any

from core.state_manager import PaperWritingState, get_sections_for_format

logger = logging.getLogger(__name__)


async def validate_topic_node(
    state: PaperWritingState,
) -> dict[str, Any]:
    """주제 검증 노드"""
    topic = state["topic"]

    errors = []
    if not topic.get("title"):
        errors.append("논문 제목이 필요합니다.")
    if not topic.get("research_focus"):
        errors.append("연구 초점이 필요합니다.")
    if not topic.get("keywords"):
        errors.append("최소 1개의 키워드가 필요합니다.")

    # 섹션 목록 설정
    if not topic.get("target_sections"):
        paper_format = topic.get("paper_format", "ieee")
        topic["target_sections"] = get_sections_for_format(paper_format)

    if errors:
        return {
            "current_step": "error",
            "error_message": "; ".join(errors),
            "log_messages": [f"[검증 실패] {'; '.join(errors)}"],
        }

    return {
        "topic": topic,
        "current_step": "research",
        "log_messages": [
            f"[주제 검증 완료] 제목: {topic['title']}, "
            f"섹션: {len(topic['target_sections'])}개"
        ],
    }


async def research_evidence_node(
    state: PaperWritingState,
    research_agent,
) -> dict[str, Any]:
    """근거 수집 노드"""
    topic = state["topic"]
    retry_count = state.get("research_retry_count", 0)

    logger.info(
        f"근거 수집 시작 (시도 {retry_count + 1})"
    )

    try:
        evidence = await research_agent.collect_evidence(
            title=topic["title"],
            research_focus=topic["research_focus"],
            keywords=topic["keywords"],
        )

        papers_found = len(evidence.get("relevant_papers", []))

        return {
            "research_evidence": evidence,
            "research_retry_count": retry_count + 1,
            "current_step": "plan",
            "log_messages": [
                f"[근거 수집 완료] {papers_found}개 관련 논문 발견, "
                f"검색 쿼리 {len(evidence.get('search_queries_used', []))}개 사용"
            ],
        }

    except Exception as e:
        logger.error(f"근거 수집 오류: {e}")
        return {
            "research_retry_count": retry_count + 1,
            "current_step": "research",
            "error_message": f"근거 수집 실패: {e}",
            "log_messages": [
                f"[근거 수집 오류] {e} (시도 {retry_count + 1})"
            ],
        }


async def planning_node(
    state: PaperWritingState,
) -> dict[str, Any]:
    """논문 구조 계획 노드"""
    topic = state["topic"]
    evidence = state.get("research_evidence", {})

    # 섹션 목록과 근거 매핑
    sections = topic.get("target_sections", [])
    papers = evidence.get("relevant_papers", [])

    outline_parts = [f"# {topic['title']}\n"]
    outline_parts.append(f"Research Focus: {topic['research_focus']}\n")
    outline_parts.append(f"Available Evidence: {len(papers)} papers\n")
    outline_parts.append(f"\n## Sections to Write:\n")

    for i, section in enumerate(sections, 1):
        outline_parts.append(f"{i}. {section}")

    outline = "\n".join(outline_parts)

    return {
        "paper_outline": outline,
        "current_step": "write",
        "log_messages": [
            f"[구조 계획 완료] {len(sections)}개 섹션 계획"
        ],
    }


async def write_sections_node(
    state: PaperWritingState,
    writing_agent,
) -> dict[str, Any]:
    """논문 작성 노드"""
    topic = state["topic"]
    evidence = state.get("research_evidence", {})

    logger.info("논문 작성 시작")

    try:
        draft = await writing_agent.write_paper(
            title=topic["title"],
            research_focus=topic["research_focus"],
            evidence=evidence,
            target_sections=topic["target_sections"],
        )

        sections_written = len(draft.get("sections", []))
        total_words = draft.get("total_word_count", 0)

        return {
            "draft": draft,
            "current_step": "verify",
            "log_messages": [
                f"[논문 작성 완료] {sections_written}개 섹션, "
                f"총 {total_words} 단어"
            ],
        }

    except Exception as e:
        logger.error(f"논문 작성 오류: {e}")
        return {
            "current_step": "error",
            "error_message": f"논문 작성 실패: {e}",
            "log_messages": [f"[논문 작성 오류] {e}"],
        }


async def verification_node(
    state: PaperWritingState,
    verification_agent,
) -> dict[str, Any]:
    """검증 노드"""
    draft = state.get("draft")
    evidence = state.get("research_evidence", {})
    topic = state["topic"]

    if not draft:
        return {
            "current_step": "error",
            "error_message": "검증할 초안이 없습니다.",
            "log_messages": ["[검증 오류] 초안이 없습니다."],
        }

    logger.info("논문 검증 시작")

    try:
        verification = await verification_agent.verify_paper(
            draft=draft,
            evidence=evidence,
            title=topic["title"],
            paper_format=topic.get("paper_format", "ieee"),
        )

        issues_count = len(verification.get("issues", []))
        overall_score = verification.get("overall_score", 0)
        is_valid = verification.get("is_valid", False)

        return {
            "verification": verification,
            "current_step": "format" if is_valid else "revise",
            "log_messages": [
                f"[검증 완료] 점수: {overall_score:.2f}, "
                f"이슈: {issues_count}개, "
                f"통과: {'예' if is_valid else '아니오'}"
            ],
        }

    except Exception as e:
        logger.error(f"검증 오류: {e}")
        return {
            "current_step": "format",  # 검증 실패 시 포맷팅으로 진행
            "log_messages": [f"[검증 오류] {e}, 검증 없이 진행"],
        }


async def revision_node(
    state: PaperWritingState,
    writing_agent,
) -> dict[str, Any]:
    """수정 노드"""
    draft = state.get("draft")
    verification = state.get("verification", {})
    evidence = state.get("research_evidence", {})
    revision_count = state.get("revision_count", 0)

    if not draft:
        return {
            "current_step": "error",
            "error_message": "수정할 초안이 없습니다.",
            "log_messages": ["[수정 오류] 초안이 없습니다."],
        }

    logger.info(f"논문 수정 시작 (수정 {revision_count + 1}회차)")

    issues = verification.get("issues", [])
    revised_sections = []

    for section in draft.get("sections", []):
        section_issues = [
            i for i in issues if i.get("section") == section["name"]
        ]
        if section_issues:
            revised = await writing_agent.revise_section(
                section=section,
                issues=section_issues,
                evidence=evidence,
            )
            revised_sections.append(revised)
        else:
            revised_sections.append(section)

    # 수정된 드래프트 생성
    revised_draft = {
        "sections": revised_sections,
        "references": draft.get("references", []),
        "total_word_count": sum(
            s.get("word_count", 0) for s in revised_sections
        ),
    }

    return {
        "draft": revised_draft,
        "revision_count": revision_count + 1,
        "current_step": "verify",
        "log_messages": [
            f"[수정 완료] {len([i for i in issues])}개 이슈 처리, "
            f"수정 {revision_count + 1}회차"
        ],
    }


async def format_export_node(
    state: PaperWritingState,
) -> dict[str, Any]:
    """포맷팅 및 내보내기 노드"""
    draft = state.get("draft")
    topic = state["topic"]

    if not draft:
        return {
            "current_step": "error",
            "error_message": "포맷팅할 초안이 없습니다.",
            "log_messages": ["[포맷팅 오류] 초안이 없습니다."],
        }

    # 최종 논문 마크다운 생성
    paper_parts = [f"# {topic['title']}\n"]

    for section in draft.get("sections", []):
        name = section.get("name", "")
        content = section.get("content", "")
        paper_parts.append(f"\n## {name}\n\n{content}")

    # 참고문헌
    references = draft.get("references", [])
    if references:
        paper_parts.append("\n## References\n")
        for ref in references:
            paper_parts.append(ref)

    final_paper = "\n".join(paper_parts)

    return {
        "final_paper": final_paper,
        "current_step": "done",
        "log_messages": [
            f"[완료] 논문 생성 완료, "
            f"총 {draft.get('total_word_count', 0)} 단어"
        ],
    }
