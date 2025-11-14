"""
TMDB API 클라이언트

TMDB (The Movie Database) API와 통신하여 영화 정보를 조회합니다.
모든 API 호출은 재시도 로직과 에러 처리를 포함합니다.
"""

# 표준 라이브러리
import logging
import time
from typing import List, Dict, Optional

# 서드파티 라이브러리
import requests

# 로깅 설정
logger = logging.getLogger(__name__)


class TMDBClient:
    """TMDB API 클라이언트 클래스"""

    def __init__(self, api_key: str, base_url: str = "https://api.themoviedb.org/3"):
        """
        TMDB API 클라이언트를 초기화합니다.

        Args:
            api_key: TMDB API 키
            base_url: TMDB API 베이스 URL (기본값: https://api.themoviedb.org/3)

        Raises:
            ValueError: API 키가 비어있는 경우
        """
        if not api_key or not api_key.strip():
            raise ValueError("TMDB API 키가 비어있습니다.")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip('/')
        self.timeout = 10  # seconds
        self.max_retries = 3
        logger.info("TMDBClient 초기화 완료")

    def _api_call_with_retry(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_retries: Optional[int] = None
    ) -> Dict:
        """
        재시도 로직을 포함한 API 호출

        Args:
            endpoint: API 엔드포인트 (예: "/search/movie")
            params: 쿼리 파라미터 (기본값: None)
            max_retries: 최대 재시도 횟수 (기본값: self.max_retries)

        Returns:
            API 응답 JSON

        Raises:
            requests.exceptions.Timeout: 타임아웃 발생 시
            requests.exceptions.HTTPError: HTTP 에러 발생 시
            requests.exceptions.RequestException: 기타 요청 에러 발생 시
        """
        if params is None:
            params = {}

        if max_retries is None:
            max_retries = self.max_retries

        # API 키 추가
        params['api_key'] = self.api_key

        # 한국어 우선 설정
        if 'language' not in params:
            params['language'] = 'ko-KR'

        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                logger.debug(f"API 호출 시도 {attempt + 1}/{max_retries}: {endpoint}")

                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                logger.debug(f"API 호출 성공: {endpoint}")
                return response.json()

            except requests.exceptions.Timeout as e:
                logger.warning(f"API 타임아웃 (시도 {attempt + 1}/{max_retries}): {endpoint}")
                if attempt == max_retries - 1:
                    logger.error(f"API 타임아웃 최종 실패: {endpoint}")
                    raise

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                logger.error(f"HTTP 에러 {status_code}: {endpoint}")

                # 재시도 불가능한 에러 (4xx)
                if 400 <= status_code < 500:
                    raise

                # 재시도 가능한 에러 (5xx)
                if attempt == max_retries - 1:
                    logger.error(f"HTTP 에러 최종 실패: {endpoint}")
                    raise

            except requests.exceptions.RequestException as e:
                logger.error(f"요청 에러 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"요청 에러 최종 실패: {endpoint}")
                    raise

            # Exponential backoff
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.debug(f"재시도 대기 중: {sleep_time}초")
                time.sleep(sleep_time)

        # 이 지점에 도달하면 안 되지만 안전을 위해
        raise requests.exceptions.RequestException("API 호출 실패")

    def search_movies(self, query: str, page: int = 1) -> List[Dict]:
        """
        영화 제목으로 검색합니다.

        Args:
            query: 검색할 영화 제목
            page: 페이지 번호 (기본값: 1)

        Returns:
            영화 정보 딕셔너리 리스트

        Raises:
            ValueError: 쿼리가 비어있는 경우
            requests.exceptions.RequestException: API 호출 실패 시
        """
        if not query or not query.strip():
            raise ValueError("검색 쿼리가 비어있습니다.")

        logger.info(f"영화 검색 시작: query='{query}', page={page}")

        try:
            params = {
                'query': query.strip(),
                'page': page
            }

            response = self._api_call_with_retry('/search/movie', params)
            results = response.get('results', [])

            logger.info(f"영화 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"영화 검색 실패: query='{query}', error={e}")
            raise

    def discover_by_genre(
        self,
        genre_ids: List[int],
        year: Optional[int] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        장르로 영화를 검색합니다.

        Args:
            genre_ids: 장르 ID 리스트
            year: 개봉 연도 (선택적)
            page: 페이지 번호 (기본값: 1)

        Returns:
            영화 정보 딕셔너리 리스트

        Raises:
            ValueError: 장르 ID 리스트가 비어있는 경우
            requests.exceptions.RequestException: API 호출 실패 시
        """
        if not genre_ids:
            raise ValueError("장르 ID 리스트가 비어있습니다.")

        logger.info(f"장르별 영화 검색 시작: genre_ids={genre_ids}, year={year}, page={page}")

        try:
            params = {
                'with_genres': ','.join(map(str, genre_ids)),
                'page': page,
                'sort_by': 'popularity.desc'
            }

            if year is not None:
                params['primary_release_year'] = year

            response = self._api_call_with_retry('/discover/movie', params)
            results = response.get('results', [])

            logger.info(f"장르별 영화 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"장르별 영화 검색 실패: genre_ids={genre_ids}, error={e}")
            raise

    def get_movie_details(self, movie_id: int) -> Dict:
        """
        영화 상세 정보를 조회합니다.

        Args:
            movie_id: 영화 ID

        Returns:
            영화 상세 정보 딕셔너리

        Raises:
            ValueError: 영화 ID가 유효하지 않은 경우
            requests.exceptions.RequestException: API 호출 실패 시
        """
        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError(f"유효하지 않은 영화 ID: {movie_id}")

        logger.info(f"영화 상세 정보 조회 시작: movie_id={movie_id}")

        try:
            params = {
                'append_to_response': 'credits,keywords,videos'
            }

            response = self._api_call_with_retry(f'/movie/{movie_id}', params)

            logger.info(f"영화 상세 정보 조회 완료: movie_id={movie_id}, title={response.get('title', 'N/A')}")
            return response

        except Exception as e:
            logger.error(f"영화 상세 정보 조회 실패: movie_id={movie_id}, error={e}")
            raise

    def get_similar_movies(self, movie_id: int, limit: int = 20) -> List[Dict]:
        """
        유사한 영화를 조회합니다.

        Args:
            movie_id: 기준 영화 ID
            limit: 반환할 최대 영화 수 (기본값: 20)

        Returns:
            유사한 영화 정보 딕셔너리 리스트

        Raises:
            ValueError: 영화 ID가 유효하지 않거나 limit이 음수인 경우
            requests.exceptions.RequestException: API 호출 실패 시
        """
        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError(f"유효하지 않은 영화 ID: {movie_id}")

        if limit <= 0:
            raise ValueError(f"limit은 양수여야 합니다: {limit}")

        logger.info(f"유사 영화 조회 시작: movie_id={movie_id}, limit={limit}")

        try:
            results = []
            page = 1
            max_pages = (limit // 20) + 1  # TMDB는 페이지당 최대 20개

            while len(results) < limit and page <= max_pages:
                params = {'page': page}
                response = self._api_call_with_retry(f'/movie/{movie_id}/similar', params)

                page_results = response.get('results', [])
                if not page_results:
                    break

                results.extend(page_results)
                page += 1

            # limit 만큼만 반환
            results = results[:limit]

            logger.info(f"유사 영화 조회 완료: movie_id={movie_id}, {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"유사 영화 조회 실패: movie_id={movie_id}, error={e}")
            raise
