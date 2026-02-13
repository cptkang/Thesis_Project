"""
설정 관리 모듈
환경 변수 및 애플리케이션 설정을 중앙 관리합니다.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # API 설정
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")

    # LLM Provider 설정
    llm_provider: str = Field(default="anthropic", env="LLM_PROVIDER")

    # LLM 모델 설정
    llm_model: str = Field(default="claude-sonnet-4-5-20250929", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=4096)

    # 임베딩 설정
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
    )

    # RAG 설정
    pdf_chunk_size: int = Field(default=1500, env="PDF_CHUNK_SIZE")
    pdf_chunk_overlap: int = Field(default=200, env="PDF_CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=10, env="RETRIEVAL_TOP_K")

    # 경로 설정
    base_dir: str = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent
        )
    )
    data_path: str = Field(default="", env="DATA_PATH")
    vector_store_path: str = Field(default="", env="VECTOR_STORE_PATH")

    # 논문 작성 설정
    paper_format: str = Field(default="ieee")  # ieee / acm
    default_language: str = Field(default="en")  # en / ko

    # 워크플로우 설정
    max_research_retries: int = Field(default=3)
    max_revision_loops: int = Field(default=2)
    verification_threshold: float = Field(default=0.7)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_data_path(self) -> Path:
        """데이터 경로 반환"""
        if self.data_path:
            p = Path(self.data_path)
            if not p.is_absolute():
                p = Path(self.base_dir) / p
            return p
        return Path(self.base_dir).parent / "data"

    def get_vector_store_path(self) -> Path:
        """벡터 스토어 경로 반환"""
        if self.vector_store_path:
            p = Path(self.vector_store_path)
            if not p.is_absolute():
                p = Path(self.base_dir) / p
            return p
        return Path(self.base_dir) / "vector_store"

    def get_prompts_path(self) -> Path:
        """프롬프트 디렉터리 경로 반환"""
        return Path(self.base_dir) / "prompts"


@lru_cache()
def get_settings() -> Settings:
    """싱글턴 Settings 인스턴스 반환"""
    return Settings()
