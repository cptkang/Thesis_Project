"""
RAG 엔진 모듈
FAISS 벡터 스토어를 사용한 시맨틱 검색 파이프라인을 제공합니다.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG (Retrieval-Augmented Generation) 엔진"""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: Optional[Path] = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = vector_store_path
        self._embedding_model = None
        self._index = None
        self._documents: list[dict] = []  # 문서 메타데이터 저장
        self._is_initialized = False

    @property
    def embedding_model(self):
        """임베딩 모델 지연 로딩"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"임베딩 모델 로딩: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        return self._embedding_model

    @property
    def document_count(self) -> int:
        """인덱싱된 문서 수"""
        return len(self._documents)

    def initialize(self):
        """벡터 스토어 초기화 (기존 인덱스 로드 또는 새로 생성)"""
        if self.vector_store_path and self._try_load_index():
            logger.info(
                f"기존 인덱스 로드 완료: {self.document_count}개 문서"
            )
            self._is_initialized = True
            return

        logger.info("새 벡터 스토어 생성")
        self._is_initialized = True

    def index_documents(self, chunks: list) -> int:
        """
        문서 청크를 벡터 스토어에 인덱싱합니다.

        Args:
            chunks: DocumentChunk 리스트

        Returns:
            인덱싱된 문서 수
        """
        import faiss

        if not chunks:
            logger.warning("인덱싱할 청크가 없습니다.")
            return 0

        logger.info(f"{len(chunks)}개 청크 인덱싱 시작")

        # 텍스트 추출
        texts = [chunk.content for chunk in chunks]

        # 임베딩 생성
        logger.info("임베딩 생성 중...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )

        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        if self._index is None:
            # Inner Product (코사인 유사도, 정규화된 벡터 사용 시)
            self._index = faiss.IndexFlatIP(dimension)

        # 인덱스에 추가
        self._index.add(np.array(embeddings, dtype=np.float32))

        # 문서 메타데이터 저장
        for i, chunk in enumerate(chunks):
            self._documents.append(
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "source_file": chunk.source_file,
                }
            )

        # 인덱스 영속화
        if self.vector_store_path:
            self._save_index()

        logger.info(
            f"인덱싱 완료: 총 {self.document_count}개 문서"
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        시맨틱 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            score_threshold: 최소 유사도 점수

        Returns:
            검색 결과 리스트 (score, content, metadata 포함)
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("인덱스가 비어 있습니다.")
            return []

        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )

        # 검색
        scores, indices = self._index.search(
            np.array(query_embedding, dtype=np.float32),
            min(top_k, self._index.ntotal),
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            if score < score_threshold:
                continue

            doc = self._documents[idx]
            results.append(
                {
                    "score": float(score),
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "chunk_index": doc["chunk_index"],
                    "source_file": doc["source_file"],
                }
            )

        return results

    def hybrid_search(
        self,
        query: str,
        keywords: list[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        시맨틱 검색 + 키워드 매칭 하이브리드 검색

        Args:
            query: 시맨틱 검색 쿼리
            keywords: 키워드 목록 (키워드 매칭용)
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트 (중복 제거, 점수 기반 정렬)
        """
        # 시맨틱 검색
        semantic_results = self.search(query, top_k=top_k * 2)

        if not keywords:
            return semantic_results[:top_k]

        # 키워드 매칭으로 점수 보정
        keyword_lower = [kw.lower() for kw in keywords]
        boosted_results = []

        for result in semantic_results:
            content_lower = result["content"].lower()
            title_lower = result["metadata"].get("title", "").lower()

            # 키워드 매칭 보너스
            keyword_hits = sum(
                1 for kw in keyword_lower
                if kw in content_lower or kw in title_lower
            )
            boost = keyword_hits * 0.1  # 키워드 1개당 0.1 보너스
            result["score"] += boost
            result["keyword_hits"] = keyword_hits
            boosted_results.append(result)

        # 점수 기반 재정렬
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        return boosted_results[:top_k]

    def get_papers_by_directory(self, directory_name: str) -> list[dict]:
        """특정 디렉터리(출처)의 논문만 검색"""
        return [
            doc
            for doc in self._documents
            if doc["metadata"].get("source_directory") == directory_name
        ]

    def get_unique_sources(self) -> list[str]:
        """인덱싱된 고유 소스 파일 목록"""
        seen = set()
        sources = []
        for doc in self._documents:
            src = doc["source_file"]
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def get_index_stats(self) -> dict:
        """인덱스 통계 반환"""
        directories = {}
        for doc in self._documents:
            d = doc["metadata"].get("source_directory", "unknown")
            directories[d] = directories.get(d, 0) + 1

        return {
            "total_chunks": self.document_count,
            "total_unique_files": len(self.get_unique_sources()),
            "chunks_by_directory": directories,
            "index_size": (
                self._index.ntotal if self._index else 0
            ),
        }

    def _save_index(self):
        """인덱스와 문서 메타데이터를 디스크에 저장"""
        import faiss

        if not self.vector_store_path or self._index is None:
            return

        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 저장
        index_path = self.vector_store_path / "faiss_index.bin"
        faiss.write_index(self._index, str(index_path))

        # 문서 메타데이터 저장
        docs_path = self.vector_store_path / "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

        logger.info(f"인덱스 저장 완료: {index_path}")

    def _try_load_index(self) -> bool:
        """디스크에서 인덱스 로드 시도"""
        import faiss

        if not self.vector_store_path:
            return False

        index_path = self.vector_store_path / "faiss_index.bin"
        docs_path = self.vector_store_path / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))
            with open(docs_path, "r", encoding="utf-8") as f:
                self._documents = json.load(f)
            return True
        except Exception as e:
            logger.error(f"인덱스 로드 실패: {e}")
            return False

    def clear_index(self):
        """인덱스 초기화"""
        self._index = None
        self._documents = []
        if self.vector_store_path:
            index_path = self.vector_store_path / "faiss_index.bin"
            docs_path = self.vector_store_path / "documents.json"
            if index_path.exists():
                index_path.unlink()
            if docs_path.exists():
                docs_path.unlink()
        logger.info("인덱스 초기화 완료")
