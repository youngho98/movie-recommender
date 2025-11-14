"""
임베딩 생성 모듈

OpenAI의 text-embedding-3-small 모델을 사용하여
텍스트를 벡터로 변환합니다.
"""

# 표준 라이브러리
import logging
import time
from typing import List

# 서드파티 라이브러리
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings

# 로컬 모듈
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

# 로깅 설정
logger = logging.getLogger(__name__)


def create_embedding(
    text: str,
    max_retries: int = 3
) -> List[float]:
    """
    단일 텍스트의 임베딩을 생성합니다.

    Args:
        text: 임베딩할 텍스트
        max_retries: 최대 재시도 횟수 (기본값: 3)

    Returns:
        임베딩 벡터 (float 리스트)

    Raises:
        ValueError: 텍스트가 비어있는 경우
        Exception: OpenAI API 호출 실패 시
    """
    # 빈 텍스트 검증
    if not text or not text.strip():
        error_msg = "임베딩할 텍스트가 비어있습니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    text = text.strip()
    logger.debug(f"임베딩 생성 시작: 텍스트 길이={len(text)}")

    # OpenAI Embeddings 초기화
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        logger.error(f"OpenAIEmbeddings 초기화 실패: {e}")
        raise

    # 재시도 로직
    for attempt in range(max_retries):
        try:
            logger.debug(f"임베딩 생성 시도 {attempt + 1}/{max_retries}")

            # 임베딩 생성
            embedding_vector = embeddings.embed_query(text)

            logger.debug(f"임베딩 생성 완료: 벡터 차원={len(embedding_vector)}")
            return embedding_vector

        except Exception as e:
            logger.warning(f"임베딩 생성 실패 (시도 {attempt + 1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                logger.error(f"임베딩 생성 최종 실패: {e}")
                raise

            # Exponential backoff
            sleep_time = 2 ** attempt
            logger.debug(f"재시도 대기 중: {sleep_time}초")
            time.sleep(sleep_time)

    # 이 지점에 도달하면 안 되지만 안전을 위해
    raise Exception("임베딩 생성 실패")


def batch_create_embeddings(
    texts: List[str],
    batch_size: int = 100,
    rate_limit_delay: float = 1.0
) -> List[List[float]]:
    """
    여러 텍스트의 임베딩을 배치로 생성합니다.

    Args:
        texts: 임베딩할 텍스트 리스트
        batch_size: 배치 크기 (기본값: 100)
        rate_limit_delay: 배치 간 대기 시간(초) (기본값: 1.0)

    Returns:
        임베딩 벡터 리스트 (각 벡터는 float 리스트)

    Raises:
        ValueError: 텍스트 리스트가 비어있거나 batch_size가 유효하지 않은 경우
        Exception: 임베딩 생성 실패 시
    """
    # 입력 검증
    if not texts:
        error_msg = "임베딩할 텍스트 리스트가 비어있습니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if batch_size <= 0:
        error_msg = f"batch_size는 양수여야 합니다: {batch_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"배치 임베딩 생성 시작: 총 {len(texts)}개 텍스트, 배치 크기={batch_size}")

    # OpenAI Embeddings 초기화
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        logger.error(f"OpenAIEmbeddings 초기화 실패: {e}")
        raise

    all_embeddings = []
    failed_indices = []

    # 배치 단위로 처리
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"총 {total_batches}개 배치로 처리")

    try:
        # tqdm progress bar
        with tqdm(total=len(texts), desc="임베딩 생성 중") as pbar:
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                logger.debug(f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_texts)}개 텍스트")

                # 배치 내 각 텍스트 처리
                batch_embeddings = []
                for idx, text in enumerate(batch_texts):
                    global_idx = batch_idx + idx

                    try:
                        # 빈 텍스트 처리
                        if not text or not text.strip():
                            logger.warning(f"인덱스 {global_idx}: 빈 텍스트 발견, 건너뜀")
                            failed_indices.append(global_idx)
                            # 빈 벡터로 대체 (차원 맞추기 위해)
                            batch_embeddings.append([])
                            pbar.update(1)
                            continue

                        # 임베딩 생성
                        embedding_vector = embeddings.embed_query(text.strip())
                        batch_embeddings.append(embedding_vector)

                    except Exception as e:
                        logger.error(f"인덱스 {global_idx} 임베딩 생성 실패: {e}")
                        failed_indices.append(global_idx)
                        # 빈 벡터로 대체
                        batch_embeddings.append([])

                    pbar.update(1)

                all_embeddings.extend(batch_embeddings)

                # Rate limit 대응 (마지막 배치 제외)
                if batch_idx + batch_size < len(texts):
                    logger.debug(f"Rate limit 대응: {rate_limit_delay}초 대기")
                    time.sleep(rate_limit_delay)

                logger.debug(f"배치 {batch_num}/{total_batches} 완료")

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        raise

    except Exception as e:
        logger.error(f"배치 임베딩 생성 중 예상치 못한 에러: {e}")
        raise

    # 결과 요약
    success_count = len(texts) - len(failed_indices)
    logger.info(f"배치 임베딩 생성 완료: 성공={success_count}/{len(texts)}, 실패={len(failed_indices)}")

    if failed_indices:
        logger.warning(f"실패한 인덱스: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

    return all_embeddings
