"""
전역 설정 및 환경 변수 관리

이 모듈은 프로젝트의 모든 설정값을 중앙 관리하며,
환경 변수를 로드하고 API 키를 검증합니다.
"""

# 표준 라이브러리
import os
import logging
from typing import Optional

# 서드파티 라이브러리
from dotenv import load_dotenv

# 로깅 설정
logger = logging.getLogger(__name__)


def _load_environment() -> None:
    """
    .env 파일에서 환경 변수를 로드합니다.

    Raises:
        RuntimeError: .env 파일 로드 실패 시
    """
    try:
        load_dotenv()
        logger.info("환경 변수 로드 완료")
    except Exception as e:
        logger.error(f"환경 변수 로드 실패: {e}")
        raise RuntimeError(f"환경 변수 로드 중 오류 발생: {e}") from e


def _validate_api_key(key_name: str, key_value: Optional[str]) -> str:
    """
    API 키의 유효성을 검증합니다.

    Args:
        key_name: 환경 변수 이름 (로깅용)
        key_value: 검증할 API 키 값

    Returns:
        검증된 API 키 문자열

    Raises:
        ValueError: API 키가 없거나 빈 문자열인 경우
    """
    if not key_value:
        error_msg = f"{key_name}이(가) 설정되지 않았습니다. .env 파일을 확인하세요."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not key_value.strip():
        error_msg = f"{key_name}이(가) 빈 문자열입니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"{key_name} 검증 완료")
    return key_value.strip()


# 환경 변수 로드
_load_environment()

# API 키 (필수)
OPENAI_API_KEY: str = _validate_api_key(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY")
)

TMDB_API_KEY: str = _validate_api_key(
    "TMDB_API_KEY",
    os.getenv("TMDB_API_KEY")
)

# TMDB API 설정
TMDB_BASE_URL: str = "https://api.themoviedb.org/3"

# ChromaDB 설정
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# OpenAI 모델 설정
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
GPT_RESPONSE_MODEL: str = os.getenv("GPT_RESPONSE_MODEL", "gpt-4o")

# 데이터 처리 설정
MAX_MOVIES: int = int(os.getenv("MAX_MOVIES", "5000"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

# 설정 로깅
logger.info("=" * 50)
logger.info("설정 로드 완료")
logger.info(f"TMDB_BASE_URL: {TMDB_BASE_URL}")
logger.info(f"CHROMA_DB_PATH: {CHROMA_DB_PATH}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"GPT_MODEL: {GPT_MODEL}")
logger.info(f"GPT_RESPONSE_MODEL: {GPT_RESPONSE_MODEL}")
logger.info(f"MAX_MOVIES: {MAX_MOVIES}")
logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
logger.info("=" * 50)
