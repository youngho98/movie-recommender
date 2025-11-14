"""
AI 영화 추천 Streamlit 앱

LangGraph Agent를 사용한 하이브리드 영화 추천 시스템
"""

# 표준 라이브러리
import logging
from typing import Dict, List

# 서드파티 라이브러리
import streamlit as st
from langchain_openai import ChatOpenAI

# 로컬 모듈
from src.agent.graph import create_agent_graph
from src.agent.state import create_default_profile, create_initial_state
from src.config import OPENAI_API_KEY, GPT_MODEL

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 페이지 설정
st.set_page_config(
    page_title="🎬 AI 영화 추천",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session() -> None:
    """
    세션 상태를 초기화합니다.

    초기화 항목:
    - user_profile: 사용자 프로필
    - messages: 대화 메시지 리스트
    - agent: LangGraph Agent 그래프

    Raises:
        Exception: 초기화 실패 시
    """
    logger.info("세션 초기화 시작")

    try:
        # 사용자 프로필 초기화
        if "user_profile" not in st.session_state:
            st.session_state.user_profile = create_default_profile()
            logger.info("사용자 프로필 초기화 완료")

        # 대화 메시지 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.info("대화 메시지 초기화 완료")

        # Agent 그래프 초기화 (캐싱)
        if "agent" not in st.session_state:
            with st.spinner("Agent 초기화 중..."):
                st.session_state.agent = create_agent_graph()
            logger.info("Agent 그래프 초기화 완료")

    except Exception as e:
        logger.error(f"세션 초기화 실패: {e}")
        st.error(f"초기화 중 오류가 발생했습니다: {e}")
        raise


def show_profile_sidebar() -> None:
    """
    사이드바에 사용자 프로필을 표시합니다.

    표시 내용:
    - 좋아하는 장르 (배지)
    - 선호 분위기 (배지)
    - 대화 횟수
    - 프로필 초기화 버튼
    """
    st.header("👤 학습된 취향")

    profile = st.session_state.user_profile

    # 선호 장르
    st.subheader("좋아하는 장르")
    preferred_genres = profile.get("preferred_genres", [])
    if preferred_genres:
        # 배지 형식으로 표시
        for genre in preferred_genres:
            st.markdown(f"🎭 `{genre}`")
    else:
        st.caption("아직 학습된 장르가 없습니다.")

    st.markdown("---")

    # 선호 분위기
    st.subheader("선호 분위기")
    mood_preferences = profile.get("mood_preferences", [])
    if mood_preferences:
        for mood in mood_preferences:
            st.markdown(f"🎨 `{mood}`")
    else:
        st.caption("아직 학습된 분위기가 없습니다.")

    st.markdown("---")

    # 통계
    st.subheader("📊 통계")
    conversation_count = profile.get("conversation_count", 0)
    st.metric("대화 횟수", conversation_count)

    liked_count = len(profile.get("liked_movies", []))
    disliked_count = len(profile.get("disliked_movies", []))
    st.metric("좋아요한 영화", liked_count)
    st.metric("싫어요한 영화", disliked_count)

    st.markdown("---")

    # 프로필 초기화 버튼
    if st.button("🔄 프로필 초기화", use_container_width=True):
        st.session_state.user_profile = create_default_profile()
        st.session_state.messages = []
        st.success("프로필이 초기화되었습니다!")
        st.rerun()


def display_movie_card(movie: Dict, idx: int) -> None:
    """
    영화 정보를 카드 형식으로 표시합니다.

    Args:
        movie: 영화 정보 딕셔너리
        idx: 영화 인덱스 (1부터 시작)
    """
    with st.container():
        st.markdown(f"### {idx}. {movie.get('title', 'Unknown')}")

        # 2컬럼 레이아웃
        col1, col2 = st.columns([1, 3])

        with col1:
            # 포스터 이미지
            poster_path = movie.get("poster_path", "")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                st.image(poster_url, use_container_width=True)
            else:
                st.info("포스터 없음")

        with col2:
            # 영화 정보
            # 평점
            vote_average = movie.get("vote_average", 0.0)
            if vote_average:
                st.markdown(f"⭐ **평점**: {vote_average}/10")

            # 장르
            genres = movie.get("genres", [])
            if genres:
                if isinstance(genres, list):
                    genres_text = ", ".join(genres)
                else:
                    genres_text = genres
                st.markdown(f"🎭 **장르**: {genres_text}")

            # 개봉일
            release_date = movie.get("release_date", "")
            if release_date:
                st.markdown(f"📅 **개봉일**: {release_date}")

            # 줄거리
            overview = movie.get("overview", "")
            if overview:
                # 100자 요약
                if len(overview) > 100:
                    overview_short = overview[:100] + "..."
                else:
                    overview_short = overview
                st.markdown(f"📝 **줄거리**: {overview_short}")

            # 좋아요/싫어요 버튼
            col_like, col_dislike = st.columns(2)

            movie_id = movie.get("movie_id")

            with col_like:
                if st.button(f"👍 좋아요", key=f"like_{movie_id}_{idx}"):
                    if movie_id not in st.session_state.user_profile.get("liked_movies", []):
                        st.session_state.user_profile["liked_movies"].append(movie_id)
                        st.success("좋아요를 추가했습니다!")
                        st.rerun()

            with col_dislike:
                if st.button(f"👎 싫어요", key=f"dislike_{movie_id}_{idx}"):
                    if movie_id not in st.session_state.user_profile.get("disliked_movies", []):
                        st.session_state.user_profile["disliked_movies"].append(movie_id)
                        st.info("싫어요를 추가했습니다.")
                        st.rerun()

        st.markdown("---")


def update_user_profile(user_input: str) -> None:
    """
    사용자 입력으로부터 취향을 추출하여 프로필을 업데이트합니다.

    Args:
        user_input: 사용자 입력 텍스트

    Raises:
        Exception: 프로필 업데이트 실패 시
    """
    logger.info(f"프로필 업데이트 시작: user_input='{user_input}'")

    try:
        # LLM으로 취향 추출
        llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        prompt = f"""사용자의 영화 취향을 분석하세요.

사용자 입력: "{user_input}"

다음 정보를 추출하세요:
1. 선호 장르 (SF, 액션, 드라마, 로맨스, 스릴러 등)
2. 선호 분위기 (감동적인, 긴장감있는, 코믹한, 어두운 등)

JSON 형식으로 출력:
{{
    "genres": ["장르1", "장르2"],
    "moods": ["분위기1", "분위기2"]
}}

추출할 수 없으면 빈 리스트를 반환하세요.
JSON만 출력:"""

        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # JSON 파싱
        import json
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        result = json.loads(response_text)

        # 프로필 업데이트
        extracted_genres = result.get("genres", [])
        extracted_moods = result.get("moods", [])

        # 기존 리스트에 추가 (중복 제거)
        for genre in extracted_genres:
            if genre not in st.session_state.user_profile["preferred_genres"]:
                st.session_state.user_profile["preferred_genres"].append(genre)

        for mood in extracted_moods:
            if mood not in st.session_state.user_profile["mood_preferences"]:
                st.session_state.user_profile["mood_preferences"].append(mood)

        # 대화 횟수 증가
        st.session_state.user_profile["conversation_count"] += 1

        logger.info(f"프로필 업데이트 완료: genres={extracted_genres}, moods={extracted_moods}")

    except Exception as e:
        logger.error(f"프로필 업데이트 실패: {e}")
        # 에러 발생해도 대화 횟수는 증가
        st.session_state.user_profile["conversation_count"] += 1


def main() -> None:
    """
    메인 애플리케이션 함수

    Raises:
        Exception: 앱 실행 실패 시
    """
    try:
        # 세션 초기화
        initialize_session()

        # 사이드바
        with st.sidebar:
            show_profile_sidebar()

        # 메인 영역
        st.title("🎬 AI 영화 추천 시스템")
        st.markdown("""
        안녕하세요! AI 영화 추천 시스템입니다.

        원하는 영화를 말씀해주세요:
        - "SF 영화 추천해줘"
        - "인터스텔라 같은 영화"
        - "감동적인 드라마 찾아줘"
        """)

        st.markdown("---")

        # 채팅 인터페이스
        st.subheader("💬 대화")

        # 대화 히스토리 표시
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            movies = message.get("movies", [])

            with st.chat_message(role):
                st.markdown(content)

                # Assistant 메시지에 영화 카드 표시
                if role == "assistant" and movies:
                    st.markdown("---")
                    st.markdown("### 🎥 추천 영화")
                    for idx, movie in enumerate(movies, 1):
                        display_movie_card(movie, idx)

        # 입력창
        user_input = st.chat_input("영화를 검색하거나 추천받으세요...")

        if user_input:
            logger.info(f"사용자 입력: {user_input}")

            # 사용자 메시지 추가
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.markdown(user_input)

            # Agent 실행
            with st.chat_message("assistant"):
                with st.spinner("영화를 찾고 있습니다..."):
                    try:
                        # 초기 상태 생성
                        initial_state = create_initial_state(
                            user_input,
                            st.session_state.user_profile
                        )

                        # Agent 실행
                        result = st.session_state.agent.invoke(initial_state)

                        # 결과 추출
                        final_movies = result.get("final_movies", [])
                        messages = result.get("messages", [])

                        # Assistant 응답 추출
                        assistant_response = ""
                        if messages:
                            for msg in messages:
                                if msg.get("role") == "assistant":
                                    assistant_response = msg.get("content", "")

                        if not assistant_response:
                            assistant_response = "영화를 찾았습니다!"

                        # 응답 표시
                        st.markdown(assistant_response)

                        # 영화 카드 표시
                        if final_movies:
                            st.markdown("---")
                            st.markdown("### 🎥 추천 영화")
                            for idx, movie in enumerate(final_movies, 1):
                                display_movie_card(movie, idx)

                            # 메시지에 저장
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": assistant_response,
                                "movies": final_movies
                            })

                        else:
                            st.warning("조건에 맞는 영화를 찾지 못했습니다. 다른 검색어를 시도해보세요.")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "조건에 맞는 영화를 찾지 못했습니다. 다른 검색어를 시도해보세요."
                            })

                        # 프로필 업데이트
                        update_user_profile(user_input)

                    except Exception as e:
                        logger.error(f"Agent 실행 실패: {e}")
                        error_message = f"오류가 발생했습니다: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message
                        })

            # 페이지 새로고침
            st.rerun()

    except Exception as e:
        logger.error(f"앱 실행 실패: {e}")
        st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
