"""
논문 작성 에이전트 시스템 - 진입점
CLI 또는 Streamlit 모드로 실행합니다.
"""

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from core.pdf_processor import PDFProcessor
from core.rag_engine import RAGEngine
from core.state_manager import PaperTopic, get_sections_for_format


def setup_logging(level: str = "INFO"):
    """로깅 설정"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def run_cli(args):
    """CLI 모드 실행"""
    settings = Settings()

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("  .env 파일에 키를 추가하거나 환경 변수를 설정하세요.")
        sys.exit(1)

    # RAG 엔진 초기화
    print("=== RAG 엔진 초기화 ===")
    rag_engine = RAGEngine(
        embedding_model_name=settings.embedding_model,
        vector_store_path=settings.get_vector_store_path(),
    )
    rag_engine.initialize()

    # 인덱싱 (필요한 경우)
    if args.index or rag_engine.document_count == 0:
        print("=== PDF 인덱싱 시작 ===")
        processor = PDFProcessor(
            chunk_size=settings.pdf_chunk_size,
            chunk_overlap=settings.pdf_chunk_overlap,
        )
        cache_dir = settings.get_vector_store_path() / "cache"
        processor.set_cache_dir(cache_dir)

        data_dir = settings.get_data_path()
        print(f"데이터 디렉터리: {data_dir}")

        chunks = processor.process_directory(
            data_dir, force_reprocess=args.reindex
        )
        if chunks:
            rag_engine.index_documents(chunks)
        print(f"인덱싱 완료: {rag_engine.document_count}개 청크")

    stats = rag_engine.get_index_stats()
    print(f"\n=== 인덱스 통계 ===")
    print(f"  총 청크: {stats['total_chunks']}")
    print(f"  고유 파일: {stats['total_unique_files']}")

    # 논문 생성
    if args.title:
        print(f"\n=== 논문 생성 시작 ===")
        print(f"  제목: {args.title}")
        print(f"  연구 초점: {args.focus}")
        print(f"  키워드: {args.keywords}")

        from graph.workflow import PaperWritingWorkflow

        topic = PaperTopic(
            title=args.title,
            research_focus=args.focus or args.title,
            keywords=args.keywords.split(",") if args.keywords else [],
            target_sections=get_sections_for_format(
                args.format or settings.paper_format
            ),
            paper_format=args.format or settings.paper_format,
            language=args.language or settings.default_language,
        )

        workflow = PaperWritingWorkflow(
            settings=settings,
            rag_engine=rag_engine,
        )

        def progress_callback(node: str, msg: str):
            print(f"  [{node}] {msg}")

        final_state = await workflow.run(topic, callback=progress_callback)

        # 결과 출력 또는 저장
        final_paper = final_state.get("final_paper", "")
        if final_paper:
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(final_paper, encoding="utf-8")
                print(f"\n논문이 저장되었습니다: {output_path}")
            else:
                print("\n" + "=" * 60)
                print(final_paper)
                print("=" * 60)
        else:
            print("\n논문 생성에 실패했습니다.")
            error = final_state.get("error_message")
            if error:
                print(f"  오류: {error}")


def run_streamlit():
    """Streamlit 모드 실행"""
    streamlit_script = PROJECT_ROOT / 'ui' / 'app.py'
    os.system(
        f"{sys.executable} -m streamlit run \"{streamlit_script}\" "
        f"--server.port 8501"
    )


def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        description="Academic Paper Writing Agent System"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="web",
        help="실행 모드 (cli: 명령줄, web: Streamlit 웹 UI)",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="논문 제목",
    )
    parser.add_argument(
        "--focus",
        type=str,
        help="연구 초점",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="키워드 (쉼표 구분)",
    )
    parser.add_argument(
        "--format",
        choices=["ieee", "acm"],
        default="ieee",
        help="논문 포맷",
    )
    parser.add_argument(
        "--language",
        choices=["en", "ko"],
        default="en",
        help="논문 언어",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="PDF 인덱싱 수행",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="기존 인덱스 삭제 후 재인덱싱",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.mode == "web":
        run_streamlit()
    else:
        asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
