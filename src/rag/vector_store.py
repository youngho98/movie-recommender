"""
벡터 저장소 관리 모듈

ChromaDB를 사용하여 영화 정보를 벡터로 저장하고 관리합니다.
"""

# 표준 라이브러리
import logging
from typing import List, Dict, Optional

# 서드파티 라이브러리
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 로컬 모듈
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DB_PATH
from src.rag.embeddings import create_embedding, batch_create_embeddings

# 로깅 설정
logger = logging.getLogger(__name__)


class MovieVectorStore:
    """영화 정보를 벡터로 저장하고 관리하는 클래스"""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        """
        MovieVectorStore를 초기화합니다.

        Args:
            persist_directory: ChromaDB 저장 경로 (기본값: config의 CHROMA_DB_PATH)

        Raises:
            Exception: ChromaDB 초기화 실패 시
        """
        logger.info(f"MovieVectorStore 초기화 시작: persist_directory={persist_directory}")

        try:
            # OpenAI Embeddings 초기화
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY
            )
            logger.info(f"Embeddings 초기화 완료: model={EMBEDDING_MODEL}")

            # ChromaDB 초기화 (컬렉션 자동 로드 또는 생성)
            self.vectorstore = Chroma(
                collection_name="movies",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info("ChromaDB 컬렉션 'movies' 초기화 완료")

            # 현재 저장된 영화 수 확인
            collection_size = self.get_collection_size()
            logger.info(f"현재 저장된 영화 수: {collection_size}개")

        except Exception as e:
            logger.error(f"MovieVectorStore 초기화 실패: {e}")
            raise

    def _create_document_text(self, movie_data: Dict) -> str:
        """
        영화 데이터로부터 임베딩할 텍스트를 생성합니다.

        Args:
            movie_data: 영화 정보 딕셔너리

        Returns:
            임베딩할 텍스트 문자열

        Raises:
            ValueError: 필수 필드가 누락된 경우
        """
        # 필수 필드 검증
        required_fields = ["title", "overview"]
        for field in required_fields:
            if field not in movie_data:
                raise ValueError(f"필수 필드 누락: {field}")

        # 텍스트 구성
        title = movie_data.get("title", "")
        genres = ", ".join(movie_data.get("genres", []))
        overview = movie_data.get("overview", "")
        keywords = ", ".join(movie_data.get("keywords", []))

        text = f"제목: {title}\n장르: {genres}\n줄거리: {overview}\n키워드: {keywords}"

        return text.strip()

    def _create_metadata(self, movie_data: Dict) -> Dict:
        """
        영화 데이터로부터 메타데이터를 생성합니다.

        Args:
            movie_data: 영화 정보 딕셔너리

        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "movie_id": movie_data.get("id", 0),
            "title": movie_data.get("title", ""),
            "genres": ", ".join(movie_data.get("genres", [])),
            "vote_average": movie_data.get("vote_average", 0.0),
            "release_date": movie_data.get("release_date", ""),
            "poster_path": movie_data.get("poster_path", "")
        }

        return metadata

    def add_movie(self, movie_data: Dict) -> None:
        """
        영화 1개를 벡터 저장소에 추가합니다.

        Args:
            movie_data: 영화 정보 딕셔너리
                - id: int (필수)
                - title: str (필수)
                - genres: List[str] (선택)
                - overview: str (필수)
                - keywords: List[str] (선택)
                - vote_average: float (선택)
                - release_date: str (선택)
                - poster_path: str (선택)

        Raises:
            ValueError: 필수 필드가 누락된 경우
            Exception: 영화 추가 실패 시
        """
        try:
            movie_id = movie_data.get("id")
            movie_title = movie_data.get("title", "Unknown")

            logger.debug(f"영화 추가 시작: id={movie_id}, title={movie_title}")

            # 텍스트 생성
            text = self._create_document_text(movie_data)

            # 메타데이터 생성
            metadata = self._create_metadata(movie_data)

            # Document 생성
            document = Document(
                page_content=text,
                metadata=metadata
            )

            # 벡터 저장소에 추가
            self.vectorstore.add_documents([document])

            logger.info(f"영화 추가 완료: id={movie_id}, title={movie_title}")

        except ValueError as e:
            logger.error(f"영화 데이터 검증 실패: {e}")
            raise

        except Exception as e:
            logger.error(f"영화 추가 실패: movie_id={movie_data.get('id')}, error={e}")
            raise

    def add_movies_batch(
        self,
        movies: List[Dict],
        batch_size: int = 100
    ) -> None:
        """
        여러 영화를 배치로 벡터 저장소에 추가합니다.

        Args:
            movies: 영화 정보 딕셔너리 리스트
            batch_size: 배치 크기 (기본값: 100)

        Raises:
            ValueError: movies가 비어있거나 batch_size가 유효하지 않은 경우
            Exception: 배치 추가 실패 시
        """
        # 입력 검증
        if not movies:
            error_msg = "영화 리스트가 비어있습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if batch_size <= 0:
            error_msg = f"batch_size는 양수여야 합니다: {batch_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"배치 영화 추가 시작: 총 {len(movies)}개 영화, 배치 크기={batch_size}")

        failed_movies = []
        success_count = 0

        try:
            # tqdm progress bar
            with tqdm(total=len(movies), desc="영화 추가 중") as pbar:
                for batch_idx in range(0, len(movies), batch_size):
                    batch_movies = movies[batch_idx:batch_idx + batch_size]
                    batch_num = (batch_idx // batch_size) + 1
                    total_batches = (len(movies) + batch_size - 1) // batch_size

                    logger.debug(f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_movies)}개 영화")

                    batch_documents = []

                    # 배치 내 각 영화 처리
                    for movie_data in batch_movies:
                        try:
                            # 텍스트 생성
                            text = self._create_document_text(movie_data)

                            # 메타데이터 생성
                            metadata = self._create_metadata(movie_data)

                            # Document 생성
                            document = Document(
                                page_content=text,
                                metadata=metadata
                            )

                            batch_documents.append(document)

                        except Exception as e:
                            movie_id = movie_data.get("id", "Unknown")
                            movie_title = movie_data.get("title", "Unknown")
                            logger.error(f"영화 처리 실패: id={movie_id}, title={movie_title}, error={e}")
                            failed_movies.append({
                                "id": movie_id,
                                "title": movie_title,
                                "error": str(e)
                            })

                        pbar.update(1)

                    # 배치 추가
                    if batch_documents:
                        try:
                            self.vectorstore.add_documents(batch_documents)
                            success_count += len(batch_documents)
                            logger.debug(f"배치 {batch_num}/{total_batches} 추가 완료: {len(batch_documents)}개")
                        except Exception as e:
                            logger.error(f"배치 {batch_num} 추가 실패: {e}")
                            for doc in batch_documents:
                                failed_movies.append({
                                    "id": doc.metadata.get("movie_id"),
                                    "title": doc.metadata.get("title"),
                                    "error": str(e)
                                })

        except KeyboardInterrupt:
            logger.warning("사용자에 의해 중단됨")
            raise

        except Exception as e:
            logger.error(f"배치 영화 추가 중 예상치 못한 에러: {e}")
            raise

        # 결과 요약
        logger.info(f"배치 영화 추가 완료: 성공={success_count}/{len(movies)}, 실패={len(failed_movies)}")

        if failed_movies:
            logger.warning(f"실패한 영화 (처음 10개): {failed_movies[:10]}")

    def get_collection_size(self) -> int:
        """
        벡터 저장소에 저장된 영화 수를 반환합니다.

        Returns:
            저장된 영화 수

        Raises:
            Exception: 컬렉션 크기 조회 실패 시
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            logger.debug(f"컬렉션 크기: {count}개")
            return count

        except Exception as e:
            logger.error(f"컬렉션 크기 조회 실패: {e}")
            raise

    def search_similar_movies(
        self,
        query: str,
        k: int = 10
    ) -> List[Document]:
        """
        쿼리와 유사한 영화를 검색합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 영화 수 (기본값: 10)

        Returns:
            유사한 영화 Document 리스트

        Raises:
            ValueError: 쿼리가 비어있거나 k가 유효하지 않은 경우
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

        logger.info(f"영화 검색 시작: query='{query}', k={k}")

        try:
            results = self.vectorstore.similarity_search(query.strip(), k=k)
            logger.info(f"영화 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"영화 검색 실패: query='{query}', error={e}")
            raise
