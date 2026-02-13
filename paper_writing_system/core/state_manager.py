"""
LangGraph 상태 관리 모듈
워크플로우에서 사용하는 모든 상태 타입을 정의합니다.
"""

from typing import Optional, Annotated
from typing_extensions import TypedDict
import operator


# ──────────────────────────────────────────────
# 입력 타입
# ──────────────────────────────────────────────

class PaperTopic(TypedDict):
    """논문 주제 입력"""
    title: str                    # 논문 제목
    research_focus: str           # 연구 초점/질문
    keywords: list[str]           # 핵심 키워드
    target_sections: list[str]    # 작성할 섹션 목록
    paper_format: str             # ieee / acm
    language: str                 # en / ko


# ──────────────────────────────────────────────
# 에이전트 출력 타입
# ──────────────────────────────────────────────

class PaperReference(TypedDict):
    """논문 참고문헌"""
    title: str
    authors: str
    source: str         # 원본 파일 경로
    key_findings: str   # 핵심 발견사항
    methodology: str    # 방법론
    relevance: str      # 연구와의 관련성
    citation_key: str   # 인용 키 (예: [1])


class ResearchEvidence(TypedDict):
    """근거 수집 에이전트 출력"""
    relevant_papers: list[PaperReference]  # 관련 논문 목록
    evidence_summary: str                   # 근거 요약
    research_gaps: list[str]               # 식별된 연구 공백
    search_queries_used: list[str]         # 사용된 검색 쿼리
    total_papers_searched: int             # 검색된 전체 논문 수


class PaperSection(TypedDict):
    """논문 섹션"""
    name: str           # 섹션 이름
    content: str        # 섹션 내용
    word_count: int     # 단어 수
    citations_used: list[str]  # 사용된 인용 키


class PaperDraft(TypedDict):
    """논문 작성 에이전트 출력"""
    sections: list[PaperSection]   # 섹션 목록
    references: list[str]          # 참고문헌 목록
    total_word_count: int          # 전체 단어 수


class VerificationIssue(TypedDict):
    """검증 이슈"""
    section: str        # 해당 섹션
    issue_type: str     # consistency / citation / quality / plagiarism
    severity: str       # high / medium / low
    description: str    # 이슈 설명
    suggestion: str     # 개선 제안


class VerificationResult(TypedDict):
    """검증 에이전트 출력"""
    is_valid: bool                              # 통과 여부
    overall_score: float                        # 전체 점수 (0~1)
    consistency_score: float                    # 논리적 일관성 점수
    citation_accuracy_score: float              # 인용 정확성 점수
    quality_score: float                        # 학술적 품질 점수
    issues: list[VerificationIssue]            # 발견된 이슈 목록
    improvement_suggestions: list[str]         # 개선 제안 목록
    revised_sections: dict[str, str]           # 수정된 섹션 (섹션명: 수정 내용)


# ──────────────────────────────────────────────
# 전체 워크플로우 상태
# ──────────────────────────────────────────────

class PaperWritingState(TypedDict):
    """LangGraph 워크플로우 전체 상태"""

    # 입력
    topic: PaperTopic

    # 에이전트 출력
    research_evidence: Optional[ResearchEvidence]
    paper_outline: Optional[str]
    draft: Optional[PaperDraft]
    verification: Optional[VerificationResult]

    # 워크플로우 제어
    current_step: str
    iteration_count: int
    research_retry_count: int
    revision_count: int

    # 에러 및 로그
    error_message: Optional[str]
    log_messages: Annotated[list[str], operator.add]

    # 최종 출력
    final_paper: Optional[str]


def create_initial_state(topic: PaperTopic) -> PaperWritingState:
    """초기 워크플로우 상태 생성"""
    return PaperWritingState(
        topic=topic,
        research_evidence=None,
        paper_outline=None,
        draft=None,
        verification=None,
        current_step="validate_topic",
        iteration_count=0,
        research_retry_count=0,
        revision_count=0,
        error_message=None,
        log_messages=[],
        final_paper=None,
    )


# ──────────────────────────────────────────────
# 기본 섹션 목록
# ──────────────────────────────────────────────

DEFAULT_SECTIONS = [
    "Introduction",
    "Related Work",
    "Methodology",
    "Experiments",
    "Results and Discussion",
    "Conclusion",
]

IEEE_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "System Model / Problem Formulation",
    "Proposed Method",
    "Experimental Setup",
    "Results and Discussion",
    "Conclusion",
    "References",
]

ACM_SECTIONS = [
    "Abstract",
    "Introduction",
    "Background and Related Work",
    "Design and Implementation",
    "Evaluation",
    "Discussion",
    "Conclusion",
    "References",
]


def get_sections_for_format(paper_format: str) -> list[str]:
    """논문 포맷에 맞는 섹션 목록 반환"""
    if paper_format.lower() == "ieee":
        return IEEE_SECTIONS
    elif paper_format.lower() == "acm":
        return ACM_SECTIONS
    return DEFAULT_SECTIONS
