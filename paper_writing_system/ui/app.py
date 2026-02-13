"""
Streamlit 웹 UI
논문 작성 에이전트 시스템의 사용자 인터페이스를 제공합니다.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

import streamlit as st

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from core.pdf_processor import PDFProcessor
from core.rag_engine import RAGEngine
from core.state_manager import (
    PaperTopic,
    get_sections_for_format,
    IEEE_SECTIONS,
    ACM_SECTIONS,
)
from graph.workflow import PaperWritingWorkflow

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────

st.set_page_config(
    page_title="Academic Paper Writing Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        "settings": None,
        "rag_engine": None,
        "workflow": None,
        "is_initialized": False,
        "is_indexing": False,
        "is_generating": False,
        "generation_logs": [],
        "final_state": None,
        "index_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ──────────────────────────────────────────
# 초기화 함수
# ──────────────────────────────────────────

def initialize_system(api_key: str, data_path: str):
    """시스템 초기화"""
    os.environ["ANTHROPIC_API_KEY"] = api_key

    settings = Settings(
        anthropic_api_key=api_key,
        data_path=data_path,
    )

    rag_engine = RAGEngine(
        embedding_model_name=settings.embedding_model,
        vector_store_path=settings.get_vector_store_path(),
    )
    rag_engine.initialize()

    st.session_state.settings = settings
    st.session_state.rag_engine = rag_engine
    st.session_state.is_initialized = True
    st.session_state.index_stats = rag_engine.get_index_stats()


def index_papers(data_path: str, force: bool = False):
    """PDF 논문 인덱싱"""
    st.session_state.is_indexing = True

    settings = st.session_state.settings
    rag_engine = st.session_state.rag_engine

    if force:
        rag_engine.clear_index()

    processor = PDFProcessor(
        chunk_size=settings.pdf_chunk_size,
        chunk_overlap=settings.pdf_chunk_overlap,
    )
    cache_dir = settings.get_vector_store_path() / "cache"
    processor.set_cache_dir(cache_dir)

    data_dir = Path(data_path)
    chunks = processor.process_directory(data_dir, force_reprocess=force)

    if chunks:
        rag_engine.index_documents(chunks)

    st.session_state.index_stats = rag_engine.get_index_stats()
    st.session_state.is_indexing = False


async def generate_paper(topic: PaperTopic):
    """논문 생성 실행"""
    settings = st.session_state.settings
    rag_engine = st.session_state.rag_engine

    workflow = PaperWritingWorkflow(
        settings=settings,
        rag_engine=rag_engine,
    )

    st.session_state.generation_logs = []
    st.session_state.is_generating = True

    def log_callback(node_name: str, message: str):
        st.session_state.generation_logs.append(
            f"[{node_name}] {message}"
        )

    final_state = await workflow.run(topic, callback=log_callback)
    st.session_state.final_state = final_state
    st.session_state.is_generating = False

    return final_state


# ──────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.title("Settings")

        # API 키 입력
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Claude API 키를 입력하세요",
        )

        # 데이터 경로
        default_data_path = str(
            PROJECT_ROOT.parent / "data"
        )
        data_path = st.text_input(
            "Data Directory",
            value=default_data_path,
            help="연구 논문 PDF가 있는 디렉터리 경로",
        )

        # 초기화 버튼
        if st.button(
            "Initialize System",
            disabled=not api_key,
            type="primary",
        ):
            with st.spinner("시스템 초기화 중..."):
                try:
                    initialize_system(api_key, data_path)
                    st.success("시스템 초기화 완료!")
                except Exception as e:
                    st.error(f"초기화 실패: {e}")

        st.divider()

        # 인덱스 상태
        st.subheader("Knowledge Base")

        if st.session_state.is_initialized:
            stats = st.session_state.index_stats
            if stats:
                st.metric(
                    "Indexed Chunks",
                    stats.get("total_chunks", 0),
                )
                st.metric(
                    "Unique Files",
                    stats.get("total_unique_files", 0),
                )

                # 디렉터리별 분포
                dirs = stats.get("chunks_by_directory", {})
                if dirs:
                    with st.expander("Directory Breakdown"):
                        for d, count in sorted(
                            dirs.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        ):
                            st.text(f"  {d}: {count}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Index Papers"):
                    with st.spinner("인덱싱 중..."):
                        index_papers(data_path)
                    st.rerun()
            with col2:
                if st.button("Re-index"):
                    with st.spinner("재인덱싱 중..."):
                        index_papers(data_path, force=True)
                    st.rerun()
        else:
            st.info("시스템을 먼저 초기화하세요.")

        st.divider()

        # 논문 설정
        st.subheader("Paper Config")
        paper_format = st.selectbox(
            "Format",
            options=["ieee", "acm"],
            index=0,
        )
        language = st.selectbox(
            "Language",
            options=["en", "ko"],
            format_func=lambda x: "English" if x == "en" else "Korean",
        )

        return paper_format, language, data_path


# ──────────────────────────────────────────
# 메인 영역
# ──────────────────────────────────────────

def render_main(paper_format: str, language: str):
    """메인 영역 렌더링"""
    st.title("Academic Paper Writing Agent")
    st.caption(
        "LangGraph + Claude API 기반 멀티 에이전트 논문 작성 시스템"
    )

    # 탭 구성
    tab_input, tab_result, tab_evidence, tab_verify, tab_export = st.tabs(
        ["Input", "Generated Paper", "Evidence", "Verification", "Export"]
    )

    # ── Input 탭 ──
    with tab_input:
        render_input_tab(paper_format, language)

    # ── Generated Paper 탭 ──
    with tab_result:
        render_result_tab()

    # ── Evidence 탭 ──
    with tab_evidence:
        render_evidence_tab()

    # ── Verification 탭 ──
    with tab_verify:
        render_verification_tab()

    # ── Export 탭 ──
    with tab_export:
        render_export_tab()


def render_input_tab(paper_format: str, language: str):
    """입력 탭 렌더링"""
    st.subheader("Paper Topic")

    title = st.text_input(
        "Paper Title",
        placeholder="예: LLM-based Network AIOps: A Comprehensive Framework",
    )
    research_focus = st.text_area(
        "Research Focus",
        placeholder="이 연구의 핵심 질문과 목표를 기술하세요...",
        height=100,
    )
    keywords_input = st.text_input(
        "Keywords (comma-separated)",
        placeholder="예: AIOps, LLM, Network Management, Log Analysis",
    )

    # 섹션 선택
    available_sections = get_sections_for_format(paper_format)
    selected_sections = st.multiselect(
        "Target Sections",
        options=available_sections,
        default=available_sections,
    )

    # 진행 상황 표시
    st.divider()
    st.subheader("Progress")

    # 로그 표시 영역
    log_container = st.container()
    with log_container:
        if st.session_state.generation_logs:
            for log in st.session_state.generation_logs:
                st.text(log)

    # 생성 버튼
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        start_disabled = (
            not st.session_state.is_initialized
            or not title
            or not research_focus
            or st.session_state.is_generating
        )
        if st.button(
            "Start Writing",
            type="primary",
            disabled=start_disabled,
        ):
            keywords = [
                k.strip() for k in keywords_input.split(",") if k.strip()
            ]
            topic = PaperTopic(
                title=title,
                research_focus=research_focus,
                keywords=keywords,
                target_sections=selected_sections,
                paper_format=paper_format,
                language=language,
            )

            with st.spinner("논문 생성 중... (수 분 소요될 수 있습니다)"):
                result = asyncio.run(generate_paper(topic))
                st.session_state.final_state = result

            st.rerun()

    with col2:
        if st.button("Reset", disabled=st.session_state.is_generating):
            st.session_state.final_state = None
            st.session_state.generation_logs = []
            st.rerun()

    if not st.session_state.is_initialized:
        st.warning(
            "사이드바에서 API 키를 입력하고 시스템을 초기화하세요."
        )


def render_result_tab():
    """생성된 논문 탭"""
    state = st.session_state.final_state

    if not state:
        st.info("논문이 아직 생성되지 않았습니다.")
        return

    final_paper = state.get("final_paper")
    draft = state.get("draft")

    if final_paper:
        st.markdown(final_paper)
    elif draft:
        for section in draft.get("sections", []):
            with st.expander(
                f"{section['name']} ({section.get('word_count', 0)} words)",
                expanded=True,
            ):
                st.markdown(section.get("content", ""))

        # 참고문헌
        refs = draft.get("references", [])
        if refs:
            st.subheader("References")
            for ref in refs:
                st.text(ref)
    else:
        st.warning("논문 생성에 실패했습니다.")
        error = state.get("error_message")
        if error:
            st.error(error)


def render_evidence_tab():
    """근거 탭"""
    state = st.session_state.final_state

    if not state:
        st.info("근거가 아직 수집되지 않았습니다.")
        return

    evidence = state.get("research_evidence")
    if not evidence:
        st.warning("근거 정보가 없습니다.")
        return

    st.subheader("Evidence Summary")
    st.markdown(evidence.get("evidence_summary", ""))

    st.subheader("Relevant Papers")
    papers = evidence.get("relevant_papers", [])
    for paper in papers:
        with st.expander(
            f"{paper.get('citation_key', '')} {paper.get('title', 'Unknown')}"
        ):
            st.text(f"Authors: {paper.get('authors', 'Unknown')}")
            st.text(f"Source: {paper.get('source', '')}")
            st.markdown(f"**Key Findings:** {paper.get('key_findings', '')}")
            st.markdown(f"**Methodology:** {paper.get('methodology', '')}")
            st.markdown(f"**Relevance:** {paper.get('relevance', '')}")

    st.subheader("Research Gaps")
    gaps = evidence.get("research_gaps", [])
    for gap in gaps:
        st.markdown(f"- {gap}")

    st.subheader("Search Queries Used")
    queries = evidence.get("search_queries_used", [])
    for q in queries:
        st.text(f"  - {q}")


def render_verification_tab():
    """검증 탭"""
    state = st.session_state.final_state

    if not state:
        st.info("검증 결과가 없습니다.")
        return

    verification = state.get("verification")
    if not verification:
        st.warning("검증이 수행되지 않았습니다.")
        return

    # 점수 대시보드
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = verification.get("overall_score", 0)
        st.metric("Overall Score", f"{score:.2f}")
    with col2:
        st.metric(
            "Consistency",
            f"{verification.get('consistency_score', 0):.2f}",
        )
    with col3:
        st.metric(
            "Citation Accuracy",
            f"{verification.get('citation_accuracy_score', 0):.2f}",
        )
    with col4:
        st.metric(
            "Quality",
            f"{verification.get('quality_score', 0):.2f}",
        )

    is_valid = verification.get("is_valid", False)
    if is_valid:
        st.success("검증 통과!")
    else:
        st.warning("검증 미통과 - 수정이 필요합니다.")

    # 이슈 목록
    st.subheader("Issues Found")
    issues = verification.get("issues", [])

    if not issues:
        st.info("발견된 이슈가 없습니다.")
    else:
        for issue in issues:
            severity = issue.get("severity", "medium")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                severity, "⚪"
            )
            with st.expander(
                f"{icon} [{severity.upper()}] {issue.get('section', '')} - "
                f"{issue.get('issue_type', '')}"
            ):
                st.markdown(f"**Description:** {issue.get('description', '')}")
                st.markdown(f"**Suggestion:** {issue.get('suggestion', '')}")

    # 개선 제안
    suggestions = verification.get("improvement_suggestions", [])
    if suggestions:
        st.subheader("Improvement Suggestions")
        for s in suggestions:
            st.markdown(f"- {s}")


def render_export_tab():
    """내보내기 탭"""
    state = st.session_state.final_state

    if not state:
        st.info("내보낼 논문이 없습니다.")
        return

    final_paper = state.get("final_paper", "")
    if not final_paper:
        st.warning("최종 논문이 생성되지 않았습니다.")
        return

    st.subheader("Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download as Markdown",
            data=final_paper,
            file_name="paper.md",
            mime="text/markdown",
        )

    with col2:
        st.download_button(
            label="Download as Text",
            data=final_paper,
            file_name="paper.txt",
            mime="text/plain",
        )

    with col3:
        st.info("DOCX/PDF export 기능은 추후 추가 예정")

    st.divider()
    st.subheader("Preview")
    st.markdown(final_paper)


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

def main():
    """메인 앱 실행"""
    paper_format, language, data_path = render_sidebar()
    render_main(paper_format, language)


if __name__ == "__main__":
    main()
else:
    main()
