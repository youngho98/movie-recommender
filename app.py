"""
AI 영화 추천 Streamlit 앱

LangGraph Agent를 사용한 하이브리드 영화 추천 시스템
"""

# 표준 라이브러리
import logging

# 서드파티 라이브러리
import streamlit as st

# 로컬 모듈
from src.agent.graph import create_agent_graph
from src.agent.state import create_default_profile

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
            st.header("⚙️ 설정")
            st.markdown("---")

            # 빈 상태 (나중에 기능 추가)
            st.info("프로필 설정 및 기타 옵션은 곧 추가됩니다.")

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

        # 채팅 인터페이스 영역 (빈 상태)
        st.subheader("💬 대화")

        # 대화 메시지 표시 영역 (나중에 구현)
        chat_container = st.container()

        # 입력창
        st.markdown("---")
        user_input = st.chat_input("영화를 검색하거나 추천받으세요...")

        if user_input:
            st.info(f"입력: {user_input}")
            # 나중에 Agent 실행 로직 추가

    except Exception as e:
        logger.error(f"앱 실행 실패: {e}")
        st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
