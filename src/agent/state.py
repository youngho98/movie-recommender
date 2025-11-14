"""
Agent State 정의

LangGraph에서 사용하는 상태(State)와 사용자 프로필을 정의합니다.
"""

# 표준 라이브러리
import operator
from typing import TypedDict, Annotated, List, Dict, Optional


class UserProfile(TypedDict, total=False):
    """
    사용자 프로필 정보를 저장하는 딕셔너리

    Attributes:
        preferred_genres: 선호하는 장르 리스트
        mood_preferences: 선호하는 분위기/무드 리스트
        liked_movies: 좋아하는 영화 ID 리스트
        disliked_movies: 싫어하는 영화 ID 리스트
        conversation_count: 대화 횟수 (프로필 갱신용)
    """
    preferred_genres: List[str]  # 선호 장르 (예: ["SF", "드라마", "스릴러"])
    mood_preferences: List[str]  # 선호 무드 (예: ["감동적인", "긴장감있는"])
    liked_movies: List[int]  # 좋아하는 영화 ID
    disliked_movies: List[int]  # 싫어하는 영화 ID
    conversation_count: int  # 대화 횟수


class AgentState(TypedDict, total=False):
    """
    LangGraph Agent의 상태

    각 노드는 이 상태를 읽고 수정합니다.
    messages 필드는 operator.add로 누적됩니다.

    Attributes:
        messages: 대화 메시지 리스트 (누적)
        user_input: 사용자 입력 텍스트
        intent: 의도 분류 결과 (예: "search", "similar", "profile_based")
        keywords: 추출된 키워드 리스트
        tmdb_results: TMDB API 검색 결과
        rag_results: RAG 벡터 검색 결과
        user_profile: 사용자 프로필 정보
        final_movies: 최종 추천 영화 리스트
        reasoning: 추천 이유 설명
    """
    # 대화 메시지 (누적)
    messages: Annotated[List[Dict], operator.add]

    # 사용자 입력
    user_input: str  # 현재 사용자 입력

    # 의도 분석
    intent: str  # 의도 분류 ("search", "similar", "profile_based", "casual")
    keywords: List[str]  # 추출된 키워드 (장르, 감정, 테마 등)

    # 검색 결과
    tmdb_results: List[Dict]  # TMDB API 검색 결과
    rag_results: List[Dict]  # RAG 벡터 검색 결과

    # 사용자 정보
    user_profile: Dict  # 사용자 프로필 (UserProfile 형태)

    # 최종 결과
    final_movies: List[Dict]  # 최종 추천 영화 리스트
    reasoning: str  # 추천 이유 설명 텍스트


def create_initial_state(user_input: str, user_profile: Optional[UserProfile] = None) -> AgentState:
    """
    초기 Agent 상태를 생성합니다.

    Args:
        user_input: 사용자 입력 텍스트
        user_profile: 사용자 프로필 (선택적)

    Returns:
        초기화된 AgentState

    Examples:
        >>> state = create_initial_state("SF 영화 추천해줘")
        >>> print(state["user_input"])
        SF 영화 추천해줘
    """
    # 기본 프로필
    default_profile: UserProfile = {
        "preferred_genres": [],
        "mood_preferences": [],
        "liked_movies": [],
        "disliked_movies": [],
        "conversation_count": 0
    }

    state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "intent": "",
        "keywords": [],
        "tmdb_results": [],
        "rag_results": [],
        "user_profile": user_profile if user_profile else default_profile,
        "final_movies": [],
        "reasoning": ""
    }

    return state


def create_default_profile() -> UserProfile:
    """
    기본 사용자 프로필을 생성합니다.

    Returns:
        초기화된 UserProfile

    Examples:
        >>> profile = create_default_profile()
        >>> print(profile["conversation_count"])
        0
    """
    profile: UserProfile = {
        "preferred_genres": [],
        "mood_preferences": [],
        "liked_movies": [],
        "disliked_movies": [],
        "conversation_count": 0
    }

    return profile
