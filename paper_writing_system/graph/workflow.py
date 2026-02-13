"""
LangGraph 워크플로우 정의
논문 작성 파이프라인의 전체 그래프를 정의합니다.
"""

import logging
from typing import Optional
from functools import partial

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from core.llm import get_chat_model

from config.settings import Settings
from core.rag_engine import RAGEngine
from core.state_manager import PaperWritingState, create_initial_state, PaperTopic
from agents.research_agent import ResearchAgent
from agents.writing_agent import WritingAgent
from agents.verification_agent import VerificationAgent
from graph.nodes import (
    validate_topic_node,
    research_evidence_node,
    planning_node,
    write_sections_node,
    verification_node,
    revision_node,
    format_export_node,
)

logger = logging.getLogger(__name__)


class PaperWritingWorkflow:
    """논문 작성 LangGraph 워크플로우"""

    def __init__(
        self,
        settings: Settings,
        rag_engine: RAGEngine,
    ):
        self.settings = settings
        self.rag_engine = rag_engine

        # LLM 초기화
        self.llm = get_chat_model(settings)

        # 에이전트 초기화
        self.research_agent = ResearchAgent(
            llm=self.llm,
            rag_engine=self.rag_engine,
            top_k=settings.retrieval_top_k,
        )
        self.writing_agent = WritingAgent(
            llm=self.llm,
            paper_format=settings.paper_format,
            language=settings.default_language,
        )
        self.verification_agent = VerificationAgent(
            llm=self.llm,
            quality_threshold=settings.verification_threshold,
        )

        # 워크플로우 그래프 빌드
        self._graph = self._build_graph()
        self._compiled_graph = self._graph.compile()

    def _build_graph(self) -> StateGraph:
        """LangGraph 상태 그래프 구성"""
        graph = StateGraph(PaperWritingState)

        # 노드 추가 (에이전트 의존성 주입)
        graph.add_node("validate_topic", validate_topic_node)
        graph.add_node(
            "research",
            partial(
                research_evidence_node,
                research_agent=self.research_agent,
            ),
        )
        graph.add_node("plan", planning_node)
        graph.add_node(
            "write",
            partial(
                write_sections_node,
                writing_agent=self.writing_agent,
            ),
        )
        graph.add_node(
            "verify",
            partial(
                verification_node,
                verification_agent=self.verification_agent,
            ),
        )
        graph.add_node(
            "revise",
            partial(
                revision_node,
                writing_agent=self.writing_agent,
            ),
        )
        graph.add_node("format", format_export_node)

        # 엣지 정의
        graph.set_entry_point("validate_topic")

        # validate_topic → research 또는 error
        graph.add_conditional_edges(
            "validate_topic",
            self._route_after_validation,
            {"research": "research", "error": END},
        )

        # research → plan 또는 research (재시도) 또는 error
        graph.add_conditional_edges(
            "research",
            self._route_after_research,
            {"plan": "plan", "research": "research", "error": END},
        )

        # plan → write
        graph.add_edge("plan", "write")

        # write → verify 또는 error
        graph.add_conditional_edges(
            "write",
            self._route_after_write,
            {"verify": "verify", "error": END},
        )

        # verify → format (통과) 또는 revise (미통과)
        graph.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {"format": "format", "revise": "revise"},
        )

        # revise → verify
        graph.add_edge("revise", "verify")

        # format → END
        graph.add_edge("format", END)

        return graph

    # ──────────────────────────────────────────
    # 라우팅 함수
    # ──────────────────────────────────────────

    def _route_after_validation(
        self, state: PaperWritingState
    ) -> str:
        """검증 후 라우팅"""
        if state.get("current_step") == "error":
            return "error"
        return "research"

    def _route_after_research(
        self, state: PaperWritingState
    ) -> str:
        """근거 수집 후 라우팅"""
        evidence = state.get("research_evidence")
        retry_count = state.get("research_retry_count", 0)

        # 근거가 없고 재시도 가능하면 재시도
        if (
            not evidence
            or not evidence.get("relevant_papers")
        ) and retry_count < self.settings.max_research_retries:
            return "research"

        # 최대 재시도 초과 또는 에러
        if state.get("current_step") == "error":
            if retry_count >= self.settings.max_research_retries:
                return "error"
            return "research"

        return "plan"

    def _route_after_write(
        self, state: PaperWritingState
    ) -> str:
        """작성 후 라우팅"""
        if state.get("current_step") == "error":
            return "error"
        return "verify"

    def _route_after_verification(
        self, state: PaperWritingState
    ) -> str:
        """검증 후 라우팅"""
        verification = state.get("verification", {})
        revision_count = state.get("revision_count", 0)

        # 통과했거나 최대 수정 횟수 도달
        if (
            verification.get("is_valid", False)
            or revision_count >= self.settings.max_revision_loops
        ):
            return "format"

        return "revise"

    # ──────────────────────────────────────────
    # 실행 인터페이스
    # ──────────────────────────────────────────

    async def run(
        self,
        topic: PaperTopic,
        callback=None,
    ) -> PaperWritingState:
        """
        워크플로우 실행

        Args:
            topic: 논문 주제
            callback: 진행 상황 콜백 (선택)

        Returns:
            최종 상태
        """
        initial_state = create_initial_state(topic)
        logger.info(f"워크플로우 시작: {topic['title']}")

        final_state = None

        async for event in self._compiled_graph.astream(
            initial_state,
            {"recursion_limit": 25},
        ):
            # 이벤트에서 노드 이름과 상태 추출
            for node_name, node_state in event.items():
                log_messages = node_state.get("log_messages", [])
                for msg in log_messages:
                    logger.info(msg)
                    if callback:
                        callback(node_name, msg)

                final_state = node_state

        # 최종 상태 로깅
        if final_state:
            step = final_state.get("current_step", "unknown")
            logger.info(f"워크플로우 종료: {step}")

        return final_state

    async def run_step_by_step(
        self,
        topic: PaperTopic,
    ):
        """
        제너레이터 방식으로 단계별 실행 (Streamlit 호환)

        Yields:
            (node_name, state_update) 튜플
        """
        initial_state = create_initial_state(topic)

        async for event in self._compiled_graph.astream(
            initial_state,
            {"recursion_limit": 25},
        ):
            for node_name, node_state in event.items():
                yield node_name, node_state

    def get_graph_visualization(self) -> str:
        """그래프 구조를 Mermaid 형식으로 반환"""
        return """
graph TD
    A[Start] --> B[주제 검증]
    B -->|유효| C[근거 수집]
    B -->|오류| Z[종료]
    C -->|근거 발견| D[구조 계획]
    C -->|근거 부족| C
    D --> E[논문 작성]
    E --> F[검증]
    F -->|통과| G[포맷팅 & 내보내기]
    F -->|미통과| H[수정]
    H --> F
    G --> Z[종료]
"""
