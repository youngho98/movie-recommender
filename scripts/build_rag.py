"""
RAG 데이터베이스 구축 스크립트

TMDB API로부터 영화 정보를 수집하고 ChromaDB에 저장합니다.
"""

# 표준 라이브러리
import argparse
import logging
import time
from typing import List, Dict, Optional
from collections import Counter
from datetime import datetime

# 서드파티 라이브러리
from tqdm import tqdm

# 로컬 모듈
from src.config import TMDB_API_KEY, TMDB_BASE_URL, CHROMA_DB_PATH
from src.tmdb.api import TMDBClient
from src.rag.vector_store import MovieVectorStore

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    명령줄 인자를 파싱합니다.

    Returns:
        파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(
        description='TMDB 영화 데이터를 수집하여 RAG 데이터베이스를 구축합니다.'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='수집할 최대 영화 수 (기본값: 100, 실제 운영: 5000)'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2000,
        help='수집 시작 연도 (기본값: 2000)'
    )

    parser.add_argument(
        '--min-rating',
        type=float,
        default=6.0,
        help='최소 평점 (기본값: 6.0)'
    )

    parser.add_argument(
        '--persist-dir',
        type=str,
        default=CHROMA_DB_PATH,
        help=f'ChromaDB 저장 경로 (기본값: {CHROMA_DB_PATH})'
    )

    return parser.parse_args()


def collect_movies(
    client: TMDBClient,
    limit: int,
    start_year: int,
    min_rating: float
) -> List[Dict]:
    """
    TMDB API로부터 영화 목록을 수집합니다.

    Args:
        client: TMDB API 클라이언트
        limit: 수집할 최대 영화 수
        start_year: 수집 시작 연도
        min_rating: 최소 평점

    Returns:
        영화 기본 정보 딕셔너리 리스트

    Raises:
        Exception: 영화 수집 실패 시
    """
    logger.info(f"영화 수집 시작: limit={limit}, start_year={start_year}, min_rating={min_rating}")

    movies = []
    page = 1
    max_pages = 500  # TMDB API 제한

    try:
        with tqdm(total=limit, desc="영화 목록 수집 중") as pbar:
            while len(movies) < limit and page <= max_pages:
                try:
                    # discover API 호출
                    response = client._api_call_with_retry(
                        '/discover/movie',
                        params={
                            'sort_by': 'popularity.desc',
                            'primary_release_date.gte': f'{start_year}-01-01',
                            'vote_average.gte': min_rating,
                            'vote_count.gte': 100,  # 충분한 투표 수
                            'page': page
                        }
                    )

                    results = response.get('results', [])

                    if not results:
                        logger.info(f"더 이상 결과 없음 (페이지 {page})")
                        break

                    # 영화 추가
                    for movie in results:
                        if len(movies) >= limit:
                            break

                        movies.append(movie)
                        pbar.update(1)

                    page += 1

                    # Rate limit 대응
                    time.sleep(0.25)

                except Exception as e:
                    logger.error(f"페이지 {page} 수집 실패: {e}")
                    page += 1
                    continue

        logger.info(f"영화 목록 수집 완료: {len(movies)}개")
        return movies

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        logger.info(f"현재까지 수집된 영화: {len(movies)}개")
        return movies

    except Exception as e:
        logger.error(f"영화 수집 실패: {e}")
        raise


def fetch_movie_details(
    client: TMDBClient,
    movies: List[Dict]
) -> List[Dict]:
    """
    각 영화의 상세 정보를 가져옵니다.

    Args:
        client: TMDB API 클라이언트
        movies: 영화 기본 정보 리스트

    Returns:
        상세 정보가 포함된 영화 딕셔너리 리스트

    Raises:
        Exception: 상세 정보 수집 실패 시
    """
    logger.info(f"영화 상세 정보 수집 시작: {len(movies)}개")

    detailed_movies = []
    failed_count = 0

    try:
        with tqdm(total=len(movies), desc="상세 정보 수집 중") as pbar:
            for movie in movies:
                movie_id = movie.get('id')

                try:
                    # 상세 정보 가져오기 (credits, keywords 포함)
                    details = client.get_movie_details(movie_id)

                    # 필요한 정보 추출
                    processed_movie = {
                        'id': details.get('id'),
                        'title': details.get('title', ''),
                        'genres': [g['name'] for g in details.get('genres', [])],
                        'overview': details.get('overview', ''),
                        'keywords': [k['name'] for k in details.get('keywords', {}).get('keywords', [])],
                        'vote_average': details.get('vote_average', 0.0),
                        'release_date': details.get('release_date', ''),
                        'poster_path': details.get('poster_path', '')
                    }

                    detailed_movies.append(processed_movie)

                except Exception as e:
                    logger.error(f"영화 {movie_id} 상세 정보 가져오기 실패: {e}")
                    failed_count += 1

                pbar.update(1)

                # Rate limit 대응
                time.sleep(0.1)

        logger.info(f"상세 정보 수집 완료: 성공={len(detailed_movies)}, 실패={failed_count}")
        return detailed_movies

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        logger.info(f"현재까지 수집된 상세 정보: {len(detailed_movies)}개")
        return detailed_movies

    except Exception as e:
        logger.error(f"상세 정보 수집 실패: {e}")
        raise


def save_to_vector_store(
    movies: List[Dict],
    persist_directory: str,
    batch_size: int = 100
) -> None:
    """
    영화 정보를 벡터 저장소에 저장합니다.

    Args:
        movies: 영화 정보 딕셔너리 리스트
        persist_directory: ChromaDB 저장 경로
        batch_size: 배치 크기 (기본값: 100)

    Raises:
        Exception: 저장 실패 시
    """
    logger.info(f"벡터 저장소 저장 시작: {len(movies)}개, batch_size={batch_size}")

    try:
        # MovieVectorStore 초기화
        vector_store = MovieVectorStore(persist_directory=persist_directory)

        # 배치 저장
        vector_store.add_movies_batch(movies, batch_size=batch_size)

        logger.info("벡터 저장소 저장 완료")

    except Exception as e:
        logger.error(f"벡터 저장소 저장 실패: {e}")
        raise


def calculate_statistics(movies: List[Dict]) -> Dict:
    """
    영화 통계를 계산합니다.

    Args:
        movies: 영화 정보 딕셔너리 리스트

    Returns:
        통계 정보 딕셔너리
    """
    logger.info("통계 계산 시작")

    # 총 영화 수
    total_count = len(movies)

    # 장르별 분포
    all_genres = []
    for movie in movies:
        all_genres.extend(movie.get('genres', []))
    genre_counter = Counter(all_genres)

    # 평균 평점
    ratings = [movie.get('vote_average', 0.0) for movie in movies]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

    # 연도별 분포
    years = []
    for movie in movies:
        release_date = movie.get('release_date', '')
        if release_date and len(release_date) >= 4:
            try:
                year = int(release_date[:4])
                years.append(year)
            except ValueError:
                pass
    year_counter = Counter(years)

    statistics = {
        'total_count': total_count,
        'genre_distribution': dict(genre_counter.most_common(10)),
        'avg_rating': round(avg_rating, 2),
        'year_distribution': dict(year_counter.most_common(10))
    }

    logger.info("통계 계산 완료")
    return statistics


def print_statistics(statistics: Dict, elapsed_time: float) -> None:
    """
    통계를 출력합니다.

    Args:
        statistics: 통계 정보 딕셔너리
        elapsed_time: 소요 시간 (초)
    """
    print("\n" + "=" * 60)
    print("RAG 데이터베이스 구축 완료")
    print("=" * 60)

    print(f"\n총 영화 수: {statistics['total_count']}개")
    print(f"평균 평점: {statistics['avg_rating']}")
    print(f"소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

    print("\n장르별 분포 (상위 10개):")
    for genre, count in statistics['genre_distribution'].items():
        print(f"  - {genre}: {count}개")

    print("\n연도별 분포 (상위 10개):")
    for year, count in sorted(statistics['year_distribution'].items(), reverse=True):
        print(f"  - {year}년: {count}개")

    print("\n" + "=" * 60)


def main() -> None:
    """메인 함수"""
    start_time = time.time()

    try:
        # 인자 파싱
        args = parse_arguments()

        logger.info("=" * 60)
        logger.info("RAG 데이터베이스 구축 시작")
        logger.info("=" * 60)
        logger.info(f"설정: limit={args.limit}, start_year={args.start_year}, min_rating={args.min_rating}")
        logger.info(f"저장 경로: {args.persist_dir}")

        # TMDB 클라이언트 초기화
        logger.info("TMDB 클라이언트 초기화 중...")
        client = TMDBClient(api_key=TMDB_API_KEY, base_url=TMDB_BASE_URL)
        logger.info("TMDB 클라이언트 초기화 완료")

        # 1. 영화 목록 수집
        print("\n[1/4] 영화 목록 수집 중...")
        movies = collect_movies(
            client=client,
            limit=args.limit,
            start_year=args.start_year,
            min_rating=args.min_rating
        )

        if not movies:
            logger.error("수집된 영화가 없습니다.")
            return

        # 2. 상세 정보 수집
        print("\n[2/4] 영화 상세 정보 수집 중...")
        detailed_movies = fetch_movie_details(client=client, movies=movies)

        if not detailed_movies:
            logger.error("상세 정보를 가져온 영화가 없습니다.")
            return

        # 3. 벡터 저장소에 저장
        print("\n[3/4] 벡터 저장소에 저장 중...")
        save_to_vector_store(
            movies=detailed_movies,
            persist_directory=args.persist_dir,
            batch_size=100
        )

        # 4. 통계 계산 및 출력
        print("\n[4/4] 통계 계산 중...")
        statistics = calculate_statistics(detailed_movies)

        elapsed_time = time.time() - start_time
        print_statistics(statistics, elapsed_time)

        logger.info(f"RAG 데이터베이스 구축 완료 (소요 시간: {elapsed_time:.2f}초)")

    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단됨")
        elapsed_time = time.time() - start_time
        logger.info(f"소요 시간: {elapsed_time:.2f}초")

    except Exception as e:
        logger.error(f"RAG 데이터베이스 구축 실패: {e}")
        raise


if __name__ == '__main__':
    main()
