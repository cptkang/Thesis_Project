# 학술 논문 작성 멀티 에이전트 시스템 — 요구사항 및 스펙

## 1. 프로젝트 개요

LangGraph 기반 멀티 에이전트 파이프라인으로, 사용자가 보유한 연구 논문(PDF)을 RAG로 활용하여 **근거 수집 → 논문 작성 → 검증**의 3단계를 자동화하는 학술 논문 작성 지원 시스템이다.

| 항목 | 내용 |
|------|------|
| 프로젝트명 | Academic Paper Writing Agent System |
| 버전 | 0.1.0 |
| 대상 사용자 | 학술 연구자 (Network AIOps / LLM 분야) |
| 대상 논문 형식 | IEEE, ACM 학술 논문 |

---

## 2. 기술 스택

| 구분 | 기술 | 버전 요구사항 |
|------|------|---------------|
| 언어 | Python | >= 3.10 |
| 에이전트 프레임워크 | LangChain + LangGraph | langchain >= 0.3.0, langgraph >= 0.2.0 |
| LLM (기본) | Anthropic Claude API | langchain-anthropic >= 0.3.0 |
| LLM (대체) | Google Gemini API | langchain-google-genai >= 0.3.0 |
| 임베딩 | sentence-transformers | >= 2.2.0 (모델: all-MiniLM-L6-v2) |
| 벡터 스토어 | FAISS | faiss-cpu >= 1.7.4 |
| PDF 처리 | pdfplumber, PyPDF2 | pdfplumber >= 0.10.0, PyPDF2 >= 3.0.0 |
| 웹 UI | Streamlit | >= 1.28.0 |
| 설정 관리 | pydantic-settings + dotenv | pydantic-settings >= 2.0.0 |

---

## 3. 시스템 아키텍처

### 3.1 전체 구조

```
paper_writing_system/
├── config/                 # 설정 관리
│   ├── __init__.py
│   └── settings.py         # pydantic-settings 기반 환경 변수 관리
├── core/                   # 핵심 인프라
│   ├── llm.py              # LLM 팩토리 (Anthropic / Google 선택)
│   ├── pdf_processor.py    # PDF 텍스트 추출 및 청킹
│   ├── rag_engine.py       # FAISS 벡터 스토어 및 검색
│   └── state_manager.py    # LangGraph 상태 타입 정의
├── agents/                 # 에이전트 구현
│   ├── research_agent.py   # 에이전트 1: 근거 수집
│   ├── writing_agent.py    # 에이전트 2: 논문 작성
│   ├── verification_agent.py # 에이전트 3: 검증
│   └── tools.py            # 공유 도구 (인용 관리, 텍스트 분석)
├── graph/                  # LangGraph 워크플로우
│   ├── workflow.py          # 상태 그래프 정의 및 컴파일
│   └── nodes.py            # 그래프 노드 함수
├── prompts/                # 프롬프트 템플릿 (YAML)
│   ├── research_prompts.yaml
│   ├── writing_prompts.yaml
│   └── verification_prompts.yaml
├── ui/                     # 사용자 인터페이스
│   └── app.py              # Streamlit 웹 앱
├── vector_store/           # FAISS 인덱스 영속 저장소
├── tests/                  # 테스트
├── main.py                 # 진입점 (CLI / Web 모드)
├── requirements.txt        # Python 의존성
├── .env.example            # 환경 변수 템플릿
└── SPEC.md                 # 본 문서
```

### 3.2 LangGraph 워크플로우

```
[주제 입력]
    │
    ▼
[주제 검증] ──에러──▶ [종료]
    │
    ▼
[근거 수집 에이전트] ──근거 부족──▶ [재시도, 최대 3회]
    │
    ▼
[구조 계획]
    │
    ▼
[논문 작성 에이전트] ──에러──▶ [종료]
    │
    ▼
[검증 에이전트]
    │         │
    ▼통과      ▼미통과
[포맷팅]   [수정 루프, 최대 2회]
    │         │
    ▼         ▼
[종료]     [검증 에이전트로 복귀]
```

**노드 목록:**

| 노드 | 함수 | 역할 |
|------|------|------|
| validate_topic | `validate_topic_node` | 입력 유효성 검증, 섹션 목록 자동 설정 |
| research | `research_evidence_node` | RAG 기반 관련 논문 검색 및 근거 추출 |
| plan | `planning_node` | 논문 구조 계획 및 근거-섹션 매핑 |
| write | `write_sections_node` | 섹션별 학술 논문 내용 생성 |
| verify | `verification_node` | 일관성·인용·품질 검증 |
| revise | `revision_node` | 검증 피드백 기반 섹션 수정 |
| format | `format_export_node` | 최종 마크다운 생성 |

**조건부 엣지:**
- `research` → 근거 부족 시 재검색 (최대 `max_research_retries`회)
- `verify` → 점수 미달 시 수정 루프 (최대 `max_revision_loops`회)

---

## 4. 기능 요구사항

### 4.1 에이전트 1 — 근거 수집 (Research Evidence Agent)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| R-01 | 연구 주제에서 5~8개의 다양한 검색 쿼리를 자동 생성한다 | 필수 |
| R-02 | FAISS 벡터 스토어에서 시맨틱 검색을 수행한다 | 필수 |
| R-03 | 시맨틱 + 키워드 하이브리드 검색을 지원한다 | 필수 |
| R-04 | 검색된 논문에서 핵심 발견사항, 방법론, 결과를 구조적으로 추출한다 | 필수 |
| R-05 | 논문 간 관계 및 연구 공백(research gap)을 식별한다 | 필수 |
| R-06 | 근거 부족 시 검색 쿼리를 변경하여 재시도한다 (최대 3회) | 필수 |
| R-07 | 추출 결과를 JSON 구조로 반환한다 (`ResearchEvidence` 타입) | 필수 |

### 4.2 에이전트 2 — 논문 작성 (Paper Writing Agent)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| W-01 | IEEE/ACM 포맷에 맞는 섹션 구조로 논문을 생성한다 | 필수 |
| W-02 | 섹션별 순차 생성: Introduction → Related Work → ... → Abstract (마지막) | 필수 |
| W-03 | 근거 에이전트의 결과를 인용 형식([1], [2])으로 통합한다 | 필수 |
| W-04 | 각 섹션에 맞는 전용 프롬프트 템플릿을 사용한다 | 필수 |
| W-05 | 이전 섹션의 내용을 컨텍스트로 제공하여 일관성을 유지한다 | 필수 |
| W-06 | 참고문헌 목록을 자동 생성한다 | 필수 |
| W-07 | 검증 피드백에 따라 개별 섹션을 수정한다 (`revise_section`) | 필수 |
| W-08 | 영어(en) 및 한국어(ko) 언어를 지원한다 | 필수 |

### 4.3 에이전트 3 — 검증 (Verification Agent)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| V-01 | 논리적 일관성을 검증한다 (주장-근거 정합, 섹션 간 모순 검출) | 필수 |
| V-02 | 인용 정확성을 확인한다 (인용 키와 참고문헌 매칭, 주장-인용 부합) | 필수 |
| V-03 | 학술적 품질을 평가한다 (어조, 구조, 논증 강도) | 필수 |
| V-04 | 표절 위험도를 평가한다 (원본 논문과 비교) | 필수 |
| V-05 | 0.0~1.0 범위의 점수를 산출한다 (overall, consistency, citation, quality) | 필수 |
| V-06 | 이슈별 심각도(high/medium/low)와 구체적 개선 제안을 제공한다 | 필수 |
| V-07 | `verification_threshold` (기본 0.7) 이상이면 통과로 판정한다 | 필수 |
| V-08 | LLM 호출 실패 시 기본 규칙 기반 검증으로 폴백한다 | 필수 |

### 4.4 RAG 파이프라인

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| RAG-01 | `data/` 디렉터리의 PDF를 재귀적으로 탐색하여 텍스트를 추출한다 | 필수 |
| RAG-02 | pdfplumber 우선, 실패 시 PyPDF2로 폴백한다 | 필수 |
| RAG-03 | 1500토큰 단위 + 200토큰 오버랩으로 청킹한다 | 필수 |
| RAG-04 | 논문 메타데이터를 추출한다 (제목, 초록, 언어, 페이지 수) | 필수 |
| RAG-05 | sentence-transformers로 임베딩을 생성하여 FAISS 인덱스에 저장한다 | 필수 |
| RAG-06 | 인덱스와 메타데이터를 디스크에 영속화한다 (faiss_index.bin, documents.json) | 필수 |
| RAG-07 | 파일 해시 기반 캐싱으로 재처리를 방지한다 | 필수 |
| RAG-08 | 한글/영문 혼합 문서를 지원한다 | 필수 |

### 4.5 웹 UI (Streamlit)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| UI-01 | 사이드바: API 키 입력, 데이터 경로 설정, 시스템 초기화 | 필수 |
| UI-02 | 사이드바: 지식 베이스 상태 표시 (인덱싱된 청크/파일 수) | 필수 |
| UI-03 | 사이드바: PDF 인덱싱 / 재인덱싱 버튼 | 필수 |
| UI-04 | Input 탭: 제목, 연구 초점, 키워드, 대상 섹션 입력 | 필수 |
| UI-05 | Input 탭: 실시간 진행 로그 표시 | 필수 |
| UI-06 | Generated Paper 탭: 생성된 논문 섹션별 표시 | 필수 |
| UI-07 | Evidence 탭: 수집된 근거 논문 목록 및 상세 정보 | 필수 |
| UI-08 | Verification 탭: 점수 대시보드 및 이슈 목록 | 필수 |
| UI-09 | Export 탭: Markdown / Text 다운로드 | 필수 |
| UI-10 | Export 탭: DOCX / PDF 내보내기 | 향후 |

---

## 5. 비기능 요구사항

| ID | 요구사항 | 상세 |
|----|----------|------|
| NF-01 | LLM Provider 확장성 | `core/llm.py` 팩토리 패턴으로 Anthropic/Google 외 추가 가능 |
| NF-02 | 에러 내성 | LLM 호출 실패 시 폴백, 최대 재시도 제한으로 무한 루프 방지 |
| NF-03 | 캐싱 | PDF 처리 결과 캐싱 (파일 해시 기반), FAISS 인덱스 영속화 |
| NF-04 | 설정 관리 | `.env` 파일 또는 환경 변수로 모든 파라미터 외부 주입 |
| NF-05 | 로깅 | Python logging 모듈, CLI에서 로그 레벨 설정 가능 |
| NF-06 | 모듈성 | 에이전트/코어/그래프/UI 계층 분리, 독립적 테스트 가능 |
| NF-07 | 다국어 | 영어(en) 및 한국어(ko) 논문 생성 지원 |

---

## 6. 데이터 모델 (State Types)

### 6.1 입력 타입

```
PaperTopic
├── title: str                    # 논문 제목
├── research_focus: str           # 연구 초점/질문
├── keywords: list[str]           # 핵심 키워드
├── target_sections: list[str]    # 작성할 섹션 목록
├── paper_format: str             # "ieee" | "acm"
└── language: str                 # "en" | "ko"
```

### 6.2 에이전트 출력 타입

```
ResearchEvidence
├── relevant_papers: list[PaperReference]
├── evidence_summary: str
├── research_gaps: list[str]
├── search_queries_used: list[str]
└── total_papers_searched: int

PaperReference
├── title, authors, source, key_findings
├── methodology, relevance, citation_key

PaperDraft
├── sections: list[PaperSection]
├── references: list[str]
└── total_word_count: int

PaperSection
├── name, content, word_count, citations_used

VerificationResult
├── is_valid: bool
├── overall_score, consistency_score, citation_accuracy_score, quality_score: float
├── issues: list[VerificationIssue]
├── improvement_suggestions: list[str]
└── revised_sections: dict[str, str]

VerificationIssue
├── section, issue_type, severity, description, suggestion
```

### 6.3 워크플로우 상태

```
PaperWritingState
├── topic: PaperTopic
├── research_evidence: ResearchEvidence | None
├── paper_outline: str | None
├── draft: PaperDraft | None
├── verification: VerificationResult | None
├── current_step: str
├── iteration_count, research_retry_count, revision_count: int
├── error_message: str | None
├── log_messages: list[str]          # Annotated[list, operator.add]
└── final_paper: str | None
```

---

## 7. 설정 파라미터

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `ANTHROPIC_API_KEY` | (필수) | Anthropic Claude API 키 |
| `GOOGLE_API_KEY` | (선택) | Google Gemini API 키 |
| `LLM_PROVIDER` | `anthropic` | LLM 제공자 (`anthropic` / `google`) |
| `LLM_MODEL` | `claude-sonnet-4-5-20250929` | 사용할 LLM 모델명 |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 임베딩 모델 |
| `PDF_CHUNK_SIZE` | `1500` | PDF 청크 크기 (문자 수) |
| `PDF_CHUNK_OVERLAP` | `200` | 청크 간 오버랩 (문자 수) |
| `RETRIEVAL_TOP_K` | `10` | RAG 검색 시 반환할 상위 결과 수 |
| `DATA_PATH` | `../data` | 연구 논문 PDF 디렉터리 |
| `VECTOR_STORE_PATH` | `./vector_store` | FAISS 인덱스 저장 경로 |

**워크플로우 내부 설정:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `max_research_retries` | 3 | 근거 수집 최대 재시도 횟수 |
| `max_revision_loops` | 2 | 검증 미통과 시 최대 수정 횟수 |
| `verification_threshold` | 0.7 | 검증 통과 최소 점수 (0.0~1.0) |
| `llm_temperature` | 0.3 | LLM 생성 온도 |
| `llm_max_tokens` | 4096 | LLM 최대 출력 토큰 수 |

---

## 8. 논문 섹션 템플릿

### 8.1 IEEE 포맷

1. Abstract
2. Introduction
3. Related Work
4. System Model / Problem Formulation
5. Proposed Method
6. Experimental Setup
7. Results and Discussion
8. Conclusion
9. References

### 8.2 ACM 포맷

1. Abstract
2. Introduction
3. Background and Related Work
4. Design and Implementation
5. Evaluation
6. Discussion
7. Conclusion
8. References

---

## 9. 실행 방법

### 9.1 Web UI (Streamlit)

```bash
pip install -r requirements.txt
python main.py --mode web
# 브라우저에서 http://localhost:8501 접속
```

### 9.2 CLI 모드

```bash
# PDF 인덱싱 + 논문 생성
python main.py --mode cli \
  --title "LLM-based Network AIOps Framework" \
  --focus "Applying large language models to automate network fault management" \
  --keywords "AIOps,LLM,Network Management,Log Analysis" \
  --format ieee \
  --language en \
  --index \
  --output output_paper.md

# 인덱스만 재구축
python main.py --mode cli --reindex
```

### 9.3 CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--mode {cli,web}` | 실행 모드 (기본: web) |
| `--title` | 논문 제목 |
| `--focus` | 연구 초점 |
| `--keywords` | 쉼표 구분 키워드 |
| `--format {ieee,acm}` | 논문 포맷 (기본: ieee) |
| `--language {en,ko}` | 논문 언어 (기본: en) |
| `--output` | 출력 파일 경로 |
| `--index` | PDF 인덱싱 수행 |
| `--reindex` | 기존 인덱스 삭제 후 재구축 |
| `--log-level` | 로그 레벨 (DEBUG/INFO/WARNING/ERROR) |

---

## 10. 향후 확장 계획

| 우선순위 | 기능 | 설명 |
|----------|------|------|
| 높음 | DOCX 내보내기 | python-docx를 활용한 Word 문서 출력 |
| 높음 | arXiv 연동 | 외부 논문 데이터베이스 실시간 검색 연동 |
| 중간 | LaTeX 출력 | IEEE/ACM LaTeX 템플릿 기반 출력 |
| 중간 | 인용 그래프 시각화 | 논문 간 인용 관계를 네트워크 그래프로 시각화 |
| 중간 | 대화형 수정 | 사용자가 개별 섹션을 선택하여 대화형으로 수정 |
| 낮음 | 다중 LLM 비교 | 동일 섹션을 다른 LLM으로 생성하여 비교 |
| 낮음 | 벤치마크 | 생성된 논문의 품질 자동 평가 프레임워크 |
