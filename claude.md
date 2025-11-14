# Claude Code 개발 가이드

> 이 문서는 Claude Code를 활용한 코드 생성 시 참고할 규칙입니다.

## 📋 코드 작성 원칙

### 1. 기본 규칙
- **타입 힌팅 필수**: 모든 함수 파라미터와 반환값에 타입 명시
- **Docstring 필수**: 모든 함수/클래스에 설명 추가 (Google Style)
- **에러 처리**: try-except로 예외 상황 대비, 로깅 포함
- **명확한 변수명**: 축약 최소화, 의도가 명확한 이름 사용

### 2. 함수 작성 패턴
```python
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def function_name(
    param1: str,
    param2: int = 10,
    param3: Optional[Dict] = None
) -> List[Dict]:
    """
    함수 설명 (한 줄 요약)
    
    Args:
        param1: 파라미터 설명
        param2: 파라미터 설명 (기본값 설명)
        param3: 선택적 파라미터 설명
    
    Returns:
        반환값 설명
    
    Raises:
        ValueError: 발생 조건
        APIError: 발생 조건
    """
    try:
        logger.info(f"함수 시작: param1={param1}")
        
        # 구현
        result = []
        
        logger.info(f"함수 완료: {len(result)}개 결과")
        return result
        
    except Exception as e:
        logger.error(f"함수 실패: {e}")
        raise
```

### 3. 클래스 작성 패턴
```python
class ClassName:
    """클래스 설명"""
    
    def __init__(self, param: str):
        """
        초기화
        
        Args:
            param: 파라미터 설명
        """
        self.param = param
        self._private_var = None  # 내부 변수는 _ prefix
        
    def public_method(self) -> str:
        """공개 메서드"""
        return self._private_method()
    
    def _private_method(self) -> str:
        """내부 메서드는 _ prefix"""
        return self.param
```

---

## 🏗️ 프로젝트 구조

```
movie-recommender/
├── src/
│   ├── config.py              # 전역 설정
│   ├── rag/
│   │   ├── embeddings.py      # 임베딩 생성
│   │   ├── vector_store.py    # ChromaDB 관리
│   │   └── search.py          # RAG 검색
│   ├── tmdb/
│   │   └── api.py             # TMDB API 클라이언트
│   ├── agent/
│   │   ├── state.py           # Agent State 정의
│   │   ├── nodes.py           # LangGraph 노드
│   │   └── graph.py           # LangGraph 구성
│   └── utils/
│       └── logger.py          # 로깅
├── scripts/
│   └── build_rag.py           # RAG DB 구축
└── app.py                     # Streamlit 앱
```

---

## 🔧 기술 스택별 규칙

### OpenAI API 사용
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import OPENAI_API_KEY, GPT_MODEL

# LLM 초기화
llm = ChatOpenAI(
    model=GPT_MODEL,  # "gpt-4o-mini"
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Embeddings 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
```

### TMDB API 사용
```python
import requests
from typing import List, Dict

def api_call_with_retry(
    url: str,
    params: Dict,
    max_retries: int = 3
) -> Dict:
    """재시도 로직 포함 API 호출"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # exponential backoff
```

### ChromaDB 사용
```python
from langchain_chroma import Chroma
from langchain.schema import Document

# 컬렉션 초기화
vectorstore = Chroma(
    collection_name="movies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 문서 추가
doc = Document(
    page_content="텍스트 내용",
    metadata={"movie_id": 123, "title": "제목"}
)
vectorstore.add_documents([doc])

# 검색
results = vectorstore.similarity_search(query, k=10)
```

### LangGraph 사용
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# State 정의
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_input: str
    # ... 기타 필드

# 노드 함수
def node_function(state: AgentState) -> AgentState:
    """노드 설명"""
    # state 수정
    state["field"] = "value"
    return state

# 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("node_name", node_function)
workflow.set_entry_point("node_name")
workflow.add_edge("node_name", END)
graph = workflow.compile()
```

### Streamlit 사용
```python
import streamlit as st

# 세션 상태 초기화
if 'key' not in st.session_state:
    st.session_state.key = default_value

# 레이아웃
col1, col2 = st.columns([1, 2])
with col1:
    st.write("왼쪽")
with col2:
    st.write("오른쪽")

# 사이드바
with st.sidebar:
    st.header("제목")
    
# 캐싱
@st.cache_data
def cached_function(param: str) -> List:
    """캐싱이 필요한 함수"""
    return expensive_operation(param)
```

---

## ⚠️ 에러 처리 패턴

### 1. API 호출 에러
```python
try:
    response = api_call()
except requests.exceptions.Timeout:
    logger.error("API 타임아웃")
    raise
except requests.exceptions.HTTPError as e:
    logger.error(f"HTTP 에러: {e.response.status_code}")
    raise
except Exception as e:
    logger.error(f"예상치 못한 에러: {e}")
    raise
```

### 2. 데이터 검증
```python
def process_data(data: Dict) -> Dict:
    """데이터 처리"""
    # 필수 필드 검증
    required_fields = ["id", "title"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"필수 필드 누락: {field}")
    
    # 타입 검증
    if not isinstance(data["id"], int):
        raise TypeError(f"id는 int여야 함: {type(data['id'])}")
    
    return data
```

### 3. 파일 처리
```python
from pathlib import Path

def load_file(filepath: str) -> str:
    """파일 로드"""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {filepath}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        logger.error(f"인코딩 에러: {filepath}")
        raise
```

---

## 📝 로깅 규칙

### 로깅 레벨
```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: 상세한 디버깅 정보
logger.debug(f"변수 값: {variable}")

# INFO: 일반적인 정보
logger.info("작업 시작")

# WARNING: 경고 (동작은 하지만 주의 필요)
logger.warning("API 응답 느림")

# ERROR: 에러 (기능 실패)
logger.error(f"처리 실패: {e}")

# CRITICAL: 심각한 에러 (앱 중단)
logger.critical("DB 연결 실패")
```

### 로깅 포맷
```python
# 함수 시작
logger.info(f"{function_name} 시작: param={param}")

# 중요 단계
logger.info(f"단계 완료: {step_name}, 결과={result_count}개")

# 함수 종료
logger.info(f"{function_name} 완료: {execution_time:.2f}초")

# 에러
logger.error(f"{function_name} 실패: {error_type} - {error_message}")
```

---

## 🎯 모듈별 특수 규칙

### config.py
- 모든 설정값은 대문자 상수로
- 환경 변수 로드 시 기본값 제공
- API 키는 검증 후 사용

### embeddings.py
- 배치 처리 시 progress bar 필수 (tqdm)
- Rate limit 대응 (time.sleep)
- 빈 텍스트 처리

### vector_store.py
- 메타데이터는 dict 형태로 저장
- persist() 호출로 영구 저장
- 컬렉션 존재 여부 확인

### api.py (TMDB)
- 모든 API 호출에 timeout 설정
- 페이지네이션 처리
- 한국어 우선 (language="ko-KR")

### nodes.py (LangGraph)
- 각 노드는 state만 수정, 새 객체 생성 금지
- 노드 시작/종료 로깅
- 에러 발생 시 state에 에러 정보 추가

### app.py (Streamlit)
- st.session_state로 상태 관리
- 긴 작업은 st.spinner 사용
- 에러는 st.error로 표시

---

## 🚫 금지 사항

### 절대 하지 말 것
1. **API 키 하드코딩**: 항상 환경 변수 사용
2. **에러 무시**: 모든 예외는 처리 또는 로깅
3. **전역 변수 남용**: 함수/클래스로 캡슐화
4. **타입 힌팅 생략**: 모든 함수에 타입 명시
5. **Docstring 생략**: 모든 공개 함수/클래스에 문서화

### 피해야 할 패턴
```python
# ❌ 나쁜 예
def func(x):  # 타입 힌팅 없음
    return x + 1  # Docstring 없음

# ✅ 좋은 예
def func(x: int) -> int:
    """x에 1을 더함"""
    return x + 1
```

---

## 📦 의존성

### 핵심 라이브러리
```python
# LLM
langchain>=0.1.0
langgraph>=0.0.20
langchain-openai>=0.0.5
langchain-chroma>=0.1.0

# API
requests>=2.31.0

# DB
chromadb>=0.4.0

# UI
streamlit>=1.30.0

# 유틸
python-dotenv>=1.0.0
tqdm>=4.66.0
```

---

## 🎨 코드 스타일

### Import 순서
```python
# 1. 표준 라이브러리
import os
import sys
from typing import List, Dict

# 2. 서드파티 라이브러리
import requests
import streamlit as st
from langchain_openai import ChatOpenAI

# 3. 로컬 모듈
from src.config import OPENAI_API_KEY
from src.rag.search import MovieRAG
```

### 줄 길이
- 최대 100자 (PEP 8 권장: 79자, 하지만 가독성 위해 100자 허용)
- 긴 함수 호출은 파라미터별로 줄바꿈

### 주석
```python
# 좋은 주석: 왜 이렇게 했는지 설명
# RAG 유사도에 2배 가중치 부여 (실험 결과 정확도 15% 향상)
score += similarity * 2.0

# 나쁜 주석: 코드 그대로 반복
# score에 similarity * 2.0을 더함
score += similarity * 2.0
```

---

## ✅ 체크리스트

코드 생성 후 확인:
- [ ] 모든 함수에 타입 힌팅
- [ ] 모든 함수에 Docstring
- [ ] try-except 에러 처리
- [ ] 로깅 추가
- [ ] import 정리
- [ ] 변수명 명확성
- [ ] API 키 환경 변수 사용
- [ ] 하드코딩된 값 없음

---

## 💡 AI 도구 활용에 대하여

이 프로젝트는 Claude Code를 활용한 페어 프로그래밍 방식으로 개발됩니다.

### 역할 분담
- **설계**: 개발자가 직접 (아키텍처, 알고리즘, 데이터 흐름)
- **구현**: Claude Code 협업 (반복 코드, 보일러플레이트)
- **검증**: 개발자가 직접 (코드 리뷰, 테스트, 디버깅)

### 철학
> "AI는 타이핑 시간을 줄여주는 도구일 뿐,  
> 최종 책임과 이해는 개발자에게 있습니다."
