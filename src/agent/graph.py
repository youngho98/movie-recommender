"""
LangGraph 그래프 구성

Agent의 노드들을 연결하여 실행 흐름을 정의합니다.
"""

# 표준 라이브러리
import logging
from typing import Literal

# 서드파티 라이브러리
from langgraph.graph import StateGraph, END

# 로컬 모듈
from src.agent.state import AgentState
from src.agent.nodes import (
    analyze_intent_node,
    search_tmdb_node,
    search_rag_node,
    rank_results_node,
    generate_response_node
)

# 로깅 설정
logger = logging.getLogger(__name__)


def route_by_intent(state: AgentState) -> Literal["search_only", "both"]:
    """
    의도에 따라 검색 경로를 결정합니다.

    라우팅 규칙:
    - "최신", "오늘" 등의 키워드 → "search_only" (TMDB만)
    - "비슷한", "같은" 등의 키워드 → "both" (TMDB + RAG)
    - intent가 "similar" → "both"
    - 기타 → "both"

    Args:
        state: 현재 Agent 상태

    Returns:
        "search_only" 또는 "both"
    """
    intent = state.get("intent", "")
    user_input = state.get("user_input", "").lower()
    keywords = state.get("keywords", [])

    logger.info(f"라우팅 판단: intent={intent}, user_input='{user_input}'")

    # "최신", "오늘" 등 → TMDB만 (실시간 데이터 필요)
    time_keywords = ["최신", "오늘", "이번주", "신작", "개봉", "상영중", "현재"]
    for keyword in time_keywords:
        if keyword in user_input:
            logger.info(f"라우팅 결정: search_only (키워드: {keyword})")
            return "search_only"

    # "비슷한", "같은" → 둘 다 (유사도 검색 필요)
    similar_keywords = ["비슷한", "같은", "유사한", "비슷", "닮은"]
    for keyword in similar_keywords:
        if keyword in user_input:
            logger.info(f"라우팅 결정: both (키워드: {keyword})")
            return "both"

    # intent가 "similar" → 둘 다
    if intent == "similar":
        logger.info("라우팅 결정: both (intent=similar)")
        return "both"

    # 기본: 둘 다 실행 (하이브리드 검색)
    logger.info("라우팅 결정: both (기본)")
    return "both"


def create_agent_graph():
    """
    Agent 그래프를 생성하고 컴파일합니다.

    그래프 구조:
    1. analyze_intent: 의도 분석 및 키워드 추출
    2. search_tmdb: TMDB API 검색 (항상 실행)
    3. route_by_intent: 조건부 분기
       - "search_only": rank_results로 직행
       - "both": search_rag 실행
    4. search_rag: RAG 벡터 검색 (조건부)
    5. rank_results: 결과 병합 및 순위 계산
    6. generate_response: 자연어 응답 생성

    Returns:
        컴파일된 LangGraph 그래프

    Raises:
        Exception: 그래프 생성 실패 시
    """
    logger.info("Agent 그래프 생성 시작")

    try:
        # StateGraph 생성
        workflow = StateGraph(AgentState)

        # 노드 추가
        logger.info("노드 추가 중...")
        workflow.add_node("analyze_intent", analyze_intent_node)
        workflow.add_node("search_tmdb", search_tmdb_node)
        workflow.add_node("search_rag", search_rag_node)
        workflow.add_node("rank_results", rank_results_node)
        workflow.add_node("generate_response", generate_response_node)
        logger.info("노드 추가 완료: 5개")

        # 엔트리 포인트 설정
        workflow.set_entry_point("analyze_intent")
        logger.info("엔트리 포인트 설정: analyze_intent")

        # 엣지 연결
        logger.info("엣지 연결 중...")

        # analyze_intent → search_tmdb (항상)
        workflow.add_edge("analyze_intent", "search_tmdb")

        # search_tmdb → (조건부 분기)
        workflow.add_conditional_edges(
            "search_tmdb",
            route_by_intent,
            {
                "search_only": "rank_results",  # TMDB만 사용
                "both": "search_rag"  # RAG 추가 실행
            }
        )

        # search_rag → rank_results
        workflow.add_edge("search_rag", "rank_results")

        # rank_results → generate_response
        workflow.add_edge("rank_results", "generate_response")

        # generate_response → END
        workflow.add_edge("generate_response", END)

        logger.info("엣지 연결 완료")

        # 그래프 컴파일
        logger.info("그래프 컴파일 중...")
        graph = workflow.compile()
        logger.info("그래프 컴파일 완료")

        logger.info("Agent 그래프 생성 완료")
        return graph

    except Exception as e:
        logger.error(f"Agent 그래프 생성 실패: {e}")
        raise


def run_agent(user_input: str, user_profile: dict = None):
    """
    Agent를 실행하고 결과를 반환합니다.

    Args:
        user_input: 사용자 입력 텍스트
        user_profile: 사용자 프로필 (선택적)

    Returns:
        실행 결과 (AgentState)

    Raises:
        Exception: Agent 실행 실패 시

    Examples:
        >>> result = run_agent("SF 영화 추천해줘")
        >>> print(result["final_movies"])
    """
    logger.info(f"Agent 실행: user_input='{user_input}'")

    try:
        # 그래프 생성
        graph = create_agent_graph()

        # 초기 상태 생성
        from src.agent.state import create_initial_state
        initial_state = create_initial_state(user_input, user_profile)

        # 그래프 실행
        logger.info("그래프 실행 시작...")
        result = graph.invoke(initial_state)
        logger.info("그래프 실행 완료")

        return result

    except Exception as e:
        logger.error(f"Agent 실행 실패: {e}")
        raise


# 그래프 시각화 (선택적)
def visualize_graph(output_path: str = "agent_graph.png") -> None:
    """
    Agent 그래프를 이미지로 저장합니다.

    Args:
        output_path: 출력 파일 경로 (기본값: "agent_graph.png")

    Raises:
        ImportError: graphviz가 설치되지 않은 경우
        Exception: 시각화 실패 시

    Note:
        graphviz 설치 필요: pip install graphviz
    """
    try:
        import graphviz
    except ImportError:
        logger.error("graphviz가 설치되지 않았습니다. 'pip install graphviz' 실행")
        raise

    logger.info(f"그래프 시각화 시작: {output_path}")

    try:
        graph = create_agent_graph()

        # Mermaid 다이어그램 생성
        mermaid = graph.get_graph().draw_mermaid()
        logger.info("Mermaid 다이어그램:")
        logger.info(mermaid)

        # PNG 이미지 생성 (graphviz 사용)
        try:
            png_data = graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_data)
            logger.info(f"그래프 이미지 저장 완료: {output_path}")
        except Exception as e:
            logger.warning(f"PNG 저장 실패: {e}")

    except Exception as e:
        logger.error(f"그래프 시각화 실패: {e}")
        raise
