"""
RAG 검색 모듈

벡터 저장소를 사용하여 다양한 방식으로 영화를 검색합니다.
"""

# 표준 라이브러리
import logging
from typing import List, Dict, Optional

# 로컬 모듈
from src.config import CHROMA_DB_PATH
from src.rag.vector_store import MovieVectorStore

# 로깅 설정
logger = logging.getLogger(__name__)


class MovieRAG:
    """영화 RAG 검색 클래스"""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        """
        MovieRAG를 초기화합니다.

        Args:
            persist_directory: ChromaDB 저장 경로 (기본값: config의 CHROMA_DB_PATH)

        Raises:
            Exception: MovieVectorStore 초기화 실패 시
        """
        logger.info(f"MovieRAG 초기화 시작: persist_directory={persist_directory}")

        try:
            # MovieVectorStore 초기화
            self.vector_store = MovieVectorStore(persist_directory=persist_directory)
            logger.info("MovieRAG 초기화 완료")

        except Exception as e:
            logger.error(f"MovieRAG 초기화 실패: {e}")
            raise

    def search_similar(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        쿼리 텍스트와 유사한 영화를 검색합니다.

        Args:
            query: 검색 쿼리 텍스트
            k: 반환할 최대 영화 수 (기본값: 10)
            min_similarity: 최소 유사도 임계값 (기본값: 0.7, 범위: 0.0~1.0)

        Returns:
            영화 정보 딕셔너리 리스트 (유사도 점수 포함)
            각 딕셔너리 형식:
            {
                "movie_id": int,
                "title": str,
                "genres": str,
                "vote_average": float,
                "release_date": str,
                "poster_path": str,
                "similarity_score": float
            }

        Raises:
            ValueError: 쿼리가 비어있거나 k/min_similarity가 유효하지 않은 경우
            Exception: 검색 실패 시
        """
        # 입력 검증
        if not query or not query.strip():
            error_msg = "검색 쿼리가 비어있습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if k <= 0:
            error_msg = f"k는 양수여야 합니다: {k}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not 0.0 <= min_similarity <= 1.0:
            error_msg = f"min_similarity는 0.0~1.0 사이여야 합니다: {min_similarity}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"영화 검색 시작: query='{query}', k={k}, min_similarity={min_similarity}")

        try:
            # 유사도 점수와 함께 검색 (k*2로 여유있게 가져온 후 필터링)
            results_with_scores = self.vector_store.vectorstore.similarity_search_with_score(
                query.strip(),
                k=k * 2
            )

            movies = []
            for doc, score in results_with_scores:
                # ChromaDB는 거리를 반환 (낮을수록 유사)
                # 유사도로 변환: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + score)

                # 최소 유사도 필터링
                if similarity < min_similarity:
                    logger.debug(f"영화 제외 (낮은 유사도): title={doc.metadata.get('title')}, similarity={similarity:.3f}")
                    continue

                movie_info = {
                    "movie_id": doc.metadata.get("movie_id", 0),
                    "title": doc.metadata.get("title", ""),
                    "genres": doc.metadata.get("genres", ""),
                    "vote_average": doc.metadata.get("vote_average", 0.0),
                    "release_date": doc.metadata.get("release_date", ""),
                    "poster_path": doc.metadata.get("poster_path", ""),
                    "similarity_score": round(similarity, 3)
                }

                movies.append(movie_info)

                # k개까지만
                if len(movies) >= k:
                    break

            logger.info(f"영화 검색 완료: {len(movies)}개 결과 (필터링 전: {len(results_with_scores)}개)")
            return movies

        except Exception as e:
            logger.error(f"영화 검색 실패: query='{query}', error={e}")
            raise

    def search_by_movie_id(
        self,
        movie_id: int,
        k: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        특정 영화와 비슷한 영화를 검색합니다.

        Args:
            movie_id: 기준 영화 ID
            k: 반환할 최대 영화 수 (기본값: 5)
            min_similarity: 최소 유사도 임계값 (기본값: 0.7)

        Returns:
            유사한 영화 정보 딕셔너리 리스트 (유사도 점수 포함)

        Raises:
            ValueError: movie_id가 유효하지 않은 경우
            Exception: 검색 실패 시
        """
        if not isinstance(movie_id, int) or movie_id <= 0:
            error_msg = f"유효하지 않은 영화 ID: {movie_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"영화 ID 기반 검색 시작: movie_id={movie_id}, k={k}")

        try:
            # 벡터 저장소에서 해당 영화 찾기
            # movie_id로 필터링하여 검색
            collection = self.vector_store.vectorstore._collection
            results = collection.get(
                where={"movie_id": movie_id},
                include=["documents", "metadatas"]
            )

            if not results['documents'] or len(results['documents']) == 0:
                logger.warning(f"영화를 찾을 수 없음: movie_id={movie_id}")
                return []

            # 첫 번째 결과 사용
            movie_text = results['documents'][0]
            movie_metadata = results['metadatas'][0]

            logger.info(f"기준 영화 찾음: title={movie_metadata.get('title', 'Unknown')}")

            # 해당 영화의 텍스트를 쿼리로 사용하여 유사 영화 검색
            # k+1로 검색 (자기 자신 제외)
            similar_movies = self.search_similar(
                query=movie_text,
                k=k + 1,
                min_similarity=min_similarity
            )

            # 자기 자신 제외
            filtered_movies = [
                movie for movie in similar_movies
                if movie["movie_id"] != movie_id
            ][:k]

            logger.info(f"영화 ID 기반 검색 완료: {len(filtered_movies)}개 결과")
            return filtered_movies

        except Exception as e:
            logger.error(f"영화 ID 기반 검색 실패: movie_id={movie_id}, error={e}")
            raise

    def search_by_profile(
        self,
        profile: Dict,
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        사용자 프로필 기반으로 영화를 검색합니다.

        Args:
            profile: 사용자 프로필 딕셔너리
                - favorite_genres: List[str] (선택)
                - mood: str (선택, 예: "감동적인", "스릴있는")
                - preferences: str (선택, 자유 형식 텍스트)
            k: 반환할 최대 영화 수 (기본값: 10)
            min_similarity: 최소 유사도 임계값 (기본값: 0.7)

        Returns:
            추천 영화 정보 딕셔너리 리스트 (유사도 점수 포함)

        Raises:
            ValueError: profile이 비어있는 경우
            Exception: 검색 실패 시

        Examples:
            >>> profile = {
            ...     "favorite_genres": ["SF", "드라마"],
            ...     "mood": "감동적이고 생각할 거리가 많은",
            ...     "preferences": "우주를 배경으로 한 영화"
            ... }
            >>> results = rag.search_by_profile(profile, k=5)
        """
        # 입력 검증
        if not profile:
            error_msg = "프로필이 비어있습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"프로필 기반 검색 시작: profile={profile}, k={k}")

        try:
            # 프로필을 자연어 쿼리로 변환
            query_parts = []

            # 선호 장르
            favorite_genres = profile.get("favorite_genres", [])
            if favorite_genres:
                genres_text = ", ".join(favorite_genres)
                query_parts.append(f"{genres_text} 장르")

            # 분위기/무드
            mood = profile.get("mood", "")
            if mood:
                query_parts.append(f"{mood} 분위기")

            # 기타 선호도
            preferences = profile.get("preferences", "")
            if preferences:
                query_parts.append(preferences)

            # 최종 쿼리 생성
            if not query_parts:
                error_msg = "프로필에서 유효한 검색 조건을 추출할 수 없습니다."
                logger.error(error_msg)
                raise ValueError(error_msg)

            query = "를 좋아하고 ".join(query_parts) + "를 선호하는 영화"

            logger.info(f"생성된 쿼리: '{query}'")

            # 쿼리로 검색
            results = self.search_similar(
                query=query,
                k=k,
                min_similarity=min_similarity
            )

            logger.info(f"프로필 기반 검색 완료: {len(results)}개 결과")
            return results

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"프로필 기반 검색 실패: profile={profile}, error={e}")
            raise

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """
        영화 ID로 영화 정보를 조회합니다.

        Args:
            movie_id: 영화 ID

        Returns:
            영화 정보 딕셔너리 (찾지 못하면 None)

        Raises:
            ValueError: movie_id가 유효하지 않은 경우
            Exception: 조회 실패 시
        """
        if not isinstance(movie_id, int) or movie_id <= 0:
            error_msg = f"유효하지 않은 영화 ID: {movie_id}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"영화 조회 시작: movie_id={movie_id}")

        try:
            collection = self.vector_store.vectorstore._collection
            results = collection.get(
                where={"movie_id": movie_id},
                include=["metadatas"]
            )

            if not results['metadatas'] or len(results['metadatas']) == 0:
                logger.warning(f"영화를 찾을 수 없음: movie_id={movie_id}")
                return None

            metadata = results['metadatas'][0]

            movie_info = {
                "movie_id": metadata.get("movie_id", 0),
                "title": metadata.get("title", ""),
                "genres": metadata.get("genres", ""),
                "vote_average": metadata.get("vote_average", 0.0),
                "release_date": metadata.get("release_date", ""),
                "poster_path": metadata.get("poster_path", "")
            }

            logger.info(f"영화 조회 완료: title={movie_info['title']}")
            return movie_info

        except Exception as e:
            logger.error(f"영화 조회 실패: movie_id={movie_id}, error={e}")
            raise
