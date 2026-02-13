"""
PDF 처리 모듈
연구 논문 PDF에서 텍스트를 추출하고, 청킹하여 RAG 파이프라인에 사용할 수 있도록 준비합니다.
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """논문 메타데이터"""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    source_path: str = ""
    source_directory: str = ""
    file_hash: str = ""
    page_count: int = 0
    language: str = ""  # en, ko, mixed


@dataclass
class DocumentChunk:
    """문서 청크"""
    content: str
    metadata: dict
    chunk_index: int
    source_file: str


class PDFProcessor:
    """PDF 문서 처리기"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._cache_dir: Optional[Path] = None

    def set_cache_dir(self, cache_dir: Path):
        """캐시 디렉터리 설정"""
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def process_directory(
        self, data_dir: Path, force_reprocess: bool = False
    ) -> list[DocumentChunk]:
        """
        디렉터리 내 모든 PDF를 재귀적으로 처리합니다.

        Args:
            data_dir: 데이터 디렉터리 경로
            force_reprocess: 캐시 무시 여부

        Returns:
            DocumentChunk 리스트
        """
        pdf_files = list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.PDF"))
        logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")

        all_chunks: list[DocumentChunk] = []
        errors: list[str] = []

        for pdf_path in tqdm(pdf_files, desc="PDF 처리 중"):
            try:
                # 캐시 확인
                if not force_reprocess and self._is_cached(pdf_path):
                    cached = self._load_cache(pdf_path)
                    if cached:
                        all_chunks.extend(cached)
                        continue

                chunks = self.process_single_pdf(pdf_path)
                all_chunks.extend(chunks)

                # 캐시 저장
                if self._cache_dir:
                    self._save_cache(pdf_path, chunks)

            except Exception as e:
                error_msg = f"PDF 처리 실패: {pdf_path.name} - {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        logger.info(
            f"처리 완료: {len(all_chunks)}개 청크 생성, "
            f"{len(errors)}개 오류 발생"
        )
        return all_chunks

    def process_single_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        """
        단일 PDF 파일을 처리합니다.

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            DocumentChunk 리스트
        """
        # 텍스트 추출
        text, page_count = self._extract_text(pdf_path)

        if not text or len(text.strip()) < 100:
            logger.warning(f"텍스트가 부족합니다: {pdf_path.name}")
            return []

        # 메타데이터 추출
        metadata = self._extract_metadata(text, pdf_path, page_count)

        # 텍스트 정제
        cleaned_text = self._clean_text(text)

        # 청킹
        chunks = self._chunk_text(cleaned_text, metadata, str(pdf_path))

        logger.debug(
            f"처리 완료: {pdf_path.name} → {len(chunks)}개 청크"
        )
        return chunks

    def _extract_text(self, pdf_path: Path) -> tuple[str, int]:
        """PDF에서 텍스트를 추출합니다."""
        text_parts = []
        page_count = 0

        # pdfplumber를 우선 시도 (더 정확한 추출)
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            if text_parts:
                return "\n\n".join(text_parts), page_count
        except Exception as e:
            logger.debug(f"pdfplumber 실패, PyPDF2로 재시도: {e}")

        # PyPDF2로 폴백
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(pdf_path))
            page_count = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n\n".join(text_parts), page_count
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패: {pdf_path.name} - {e}")
            raise

    def _extract_metadata(
        self, text: str, pdf_path: Path, page_count: int
    ) -> PaperMetadata:
        """텍스트에서 논문 메타데이터를 추출합니다."""
        metadata = PaperMetadata(
            source_path=str(pdf_path),
            source_directory=pdf_path.parent.name,
            file_hash=self._compute_file_hash(pdf_path),
            page_count=page_count,
        )

        # 제목 추출 (첫 몇 줄에서)
        lines = text.strip().split("\n")
        non_empty_lines = [l.strip() for l in lines[:10] if l.strip()]
        if non_empty_lines:
            # 가장 긴 첫 줄을 제목으로 추정
            title_candidates = non_empty_lines[:3]
            metadata.title = max(title_candidates, key=len)

        # Abstract 추출
        abstract_match = re.search(
            r"(?:Abstract|ABSTRACT|요약)[:\s—\-]*(.+?)(?:\n\n|Keywords|KEYWORDS|Index Terms|1\.\s|I\.\s)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if abstract_match:
            metadata.abstract = abstract_match.group(1).strip()[:1000]

        # 언어 감지 (한글 포함 여부)
        korean_chars = len(re.findall(r"[\uac00-\ud7af]", text[:2000]))
        total_chars = len(text[:2000])
        if korean_chars > total_chars * 0.3:
            metadata.language = "ko"
        elif korean_chars > 0:
            metadata.language = "mixed"
        else:
            metadata.language = "en"

        return metadata

    def _clean_text(self, text: str) -> str:
        """텍스트를 정제합니다."""
        # 여러 공백을 하나로
        text = re.sub(r"[ \t]+", " ", text)
        # 3개 이상 연속 개행을 2개로
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 페이지 번호 패턴 제거
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # 헤더/푸터 반복 패턴 제거 (간단한 휴리스틱)
        text = re.sub(
            r"(?:Authorized licensed use limited to:.*?\n)", "", text
        )
        text = re.sub(
            r"(?:Downloaded on.*?from IEEE Xplore.*?\n)", "", text
        )
        return text.strip()

    def _chunk_text(
        self,
        text: str,
        metadata: PaperMetadata,
        source_file: str,
    ) -> list[DocumentChunk]:
        """텍스트를 청크로 분할합니다."""
        chunks: list[DocumentChunk] = []

        # 단락 기반 분할 시도
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 현재 청크에 추가해도 크기 초과하지 않으면 추가
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += ("\n\n" + para) if current_chunk else para
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(
                        DocumentChunk(
                            content=current_chunk,
                            metadata={
                                "title": metadata.title,
                                "authors": metadata.authors,
                                "abstract": metadata.abstract[:300],
                                "source_directory": metadata.source_directory,
                                "language": metadata.language,
                                "page_count": metadata.page_count,
                            },
                            chunk_index=chunk_index,
                            source_file=source_file,
                        )
                    )
                    chunk_index += 1

                # 오버랩 처리: 이전 청크 끝부분을 새 청크 시작으로
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

        # 마지막 청크 저장
        if current_chunk:
            chunks.append(
                DocumentChunk(
                    content=current_chunk,
                    metadata={
                        "title": metadata.title,
                        "authors": metadata.authors,
                        "abstract": metadata.abstract[:300],
                        "source_directory": metadata.source_directory,
                        "language": metadata.language,
                        "page_count": metadata.page_count,
                    },
                    chunk_index=chunk_index,
                    source_file=source_file,
                )
            )

        return chunks

    def _compute_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (캐시 유효성 확인용)"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def _is_cached(self, pdf_path: Path) -> bool:
        """캐시 존재 여부 확인"""
        if not self._cache_dir:
            return False
        cache_file = self._get_cache_path(pdf_path)
        return cache_file.exists()

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """캐시 파일 경로 생성"""
        name_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
        return self._cache_dir / f"{name_hash}.json"

    def _save_cache(self, pdf_path: Path, chunks: list[DocumentChunk]):
        """처리 결과를 캐시에 저장"""
        if not self._cache_dir:
            return
        cache_data = {
            "file_hash": self._compute_file_hash(pdf_path),
            "chunks": [
                {
                    "content": c.content,
                    "metadata": c.metadata,
                    "chunk_index": c.chunk_index,
                    "source_file": c.source_file,
                }
                for c in chunks
            ],
        }
        cache_path = self._get_cache_path(pdf_path)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def _load_cache(self, pdf_path: Path) -> Optional[list[DocumentChunk]]:
        """캐시에서 처리 결과 로드"""
        cache_path = self._get_cache_path(pdf_path)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # 파일 해시 확인
            current_hash = self._compute_file_hash(pdf_path)
            if cache_data.get("file_hash") != current_hash:
                return None

            return [
                DocumentChunk(
                    content=c["content"],
                    metadata=c["metadata"],
                    chunk_index=c["chunk_index"],
                    source_file=c["source_file"],
                )
                for c in cache_data["chunks"]
            ]
        except Exception:
            return None
