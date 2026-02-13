"""
에이전트 공유 도구 모듈
모든 에이전트가 공유하는 유틸리티 함수와 도구를 제공합니다.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CitationManager:
    """인용 관리자 - IEEE/ACM 스타일 인용 생성 및 관리"""

    def __init__(self, style: str = "ieee"):
        self.style = style.lower()
        self._citations: dict[str, dict] = {}  # citation_key -> paper_info
        self._counter = 0

    def add_paper(self, paper_info: dict) -> str:
        """
        논문을 인용 목록에 추가하고 인용 키를 반환합니다.

        Args:
            paper_info: 논문 정보 딕셔너리

        Returns:
            인용 키 (예: [1])
        """
        # 중복 체크 (제목 기반)
        title = paper_info.get("title", "")
        for key, info in self._citations.items():
            if info.get("title", "").lower() == title.lower():
                return key

        self._counter += 1
        citation_key = f"[{self._counter}]"
        self._citations[citation_key] = paper_info
        return citation_key

    def format_reference(self, citation_key: str) -> str:
        """인용 키에 해당하는 참고문헌 포맷팅"""
        info = self._citations.get(citation_key, {})
        if not info:
            return ""

        title = info.get("title", "Unknown Title")
        authors = info.get("authors", "Unknown Authors")

        if self.style == "ieee":
            return f'{citation_key} {authors}, "{title}."'
        elif self.style == "acm":
            return f'{citation_key} {authors}. {title}.'
        return f"{citation_key} {authors}, {title}."

    def get_all_references(self) -> list[str]:
        """모든 참고문헌을 포맷팅하여 반환"""
        refs = []
        for key in sorted(
            self._citations.keys(),
            key=lambda x: int(x.strip("[]")),
        ):
            refs.append(self.format_reference(key))
        return refs

    def get_citation_count(self) -> int:
        """현재 인용 수 반환"""
        return len(self._citations)


class ContentAnalyzer:
    """텍스트 분석 유틸리티"""

    @staticmethod
    def count_words(text: str) -> int:
        """단어 수 계산 (한글/영문 혼합 지원)"""
        # 영어 단어
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        # 한글 글자 (한글은 글자 단위로 세는 것이 일반적)
        korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))
        return english_words + korean_chars

    @staticmethod
    def extract_section_names(text: str) -> list[str]:
        """텍스트에서 섹션 이름 추출"""
        patterns = [
            r"^(?:\d+\.?\s+)(.+)$",           # 1. Introduction
            r"^(?:[IVX]+\.?\s+)(.+)$",         # I. Introduction
            r"^(?:#{1,3}\s+)(.+)$",            # ## Introduction
        ]
        sections = []
        for line in text.split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    sections.append(match.group(1).strip())
                    break
        return sections

    @staticmethod
    def assess_academic_tone(text: str) -> dict:
        """학술적 어조 간단 평가"""
        # 비학술적 표현 패턴
        informal_patterns = [
            r"\b(really|very|a lot|kind of|sort of|thing|stuff)\b",
            r"\b(gonna|wanna|gotta|ain't)\b",
            r"!{2,}",
            r"\?\?+",
        ]

        issues = []
        for pattern in informal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(
                    f"비학술적 표현 발견: {', '.join(set(matches))}"
                )

        # 수동태 비율 (학술 논문에서 선호)
        passive_count = len(
            re.findall(
                r"\b(?:is|are|was|were|been|be)\s+\w+ed\b",
                text,
                re.IGNORECASE,
            )
        )

        return {
            "issues": issues,
            "issue_count": len(issues),
            "passive_voice_count": passive_count,
            "is_academic": len(issues) == 0,
        }


def format_search_results(results: list[dict]) -> str:
    """
    RAG 검색 결과를 LLM에 전달하기 위한 포맷으로 변환합니다.

    Args:
        results: RAG 검색 결과 리스트

    Returns:
        포맷팅된 텍스트
    """
    if not results:
        return "검색 결과가 없습니다."

    formatted = []
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        title = metadata.get("title", "제목 없음")
        source = result.get("source_file", "출처 없음")
        score = result.get("score", 0)
        content = result.get("content", "")

        # 내용 길이 제한
        if len(content) > 800:
            content = content[:800] + "..."

        formatted.append(
            f"--- 논문 {i} (유사도: {score:.3f}) ---\n"
            f"제목: {title}\n"
            f"출처: {source}\n"
            f"내용:\n{content}\n"
        )

    return "\n".join(formatted)


def parse_llm_json_response(response: str) -> Optional[dict]:
    """
    LLM 응답에서 JSON 부분을 추출하고 파싱합니다.

    Args:
        response: LLM 응답 텍스트

    Returns:
        파싱된 딕셔너리 또는 None
    """
    import json

    # JSON 코드 블록 추출
    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        response,
        re.DOTALL,
    )
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 전체 응답을 JSON으로 파싱 시도
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 중괄호 기반 추출
    brace_match = re.search(r"\{.*\}", response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("LLM 응답에서 JSON을 파싱할 수 없습니다.")
    return None
