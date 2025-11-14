"""
LangGraph 노드 함수들

Agent의 각 단계를 수행하는 노드 함수들을 정의합니다.
각 노드는 AgentState를 받아 처리 후 업데이트된 state를 반환합니다.
"""

# 표준 라이브러리
import logging
import time
import json
from typing import List, Dict, Any

# 서드파티 라이브러리
from langchain_openai import ChatOpenAI

# 로컬 모듈
from src.config import OPENAI_API_KEY, GPT_MODEL, TMDB_API_KEY, TMDB_BASE_URL
from src.agent.state import AgentState
from src.tmdb.api import TMDBClient
from src.rag.search import MovieRAG

# 로깅 설정
logger = logging.getLogger(__name__)


def analyze_intent_node(state: AgentState) -> AgentState:
    """
    사용자 입력의 의도를 분석하고 키워드를 추출합니다.

    GPT-4o-mini를 사용하여:
    - 의도 분류: "search", "recommend", "similar"
    - 키워드 추출: 장르, 테마, 감정 등

    Args:
        state: 현재 Agent 상태

    Returns:
        업데이트된 Agent 상태 (intent, keywords 필드 수정)

    Raises:
        Exception: 의도 분석 실패 시
    """
    start_time = time.time()
    logger.info("=== analyze_intent_node 시작 ===")
    logger.info(f"사용자 입력: {state.get('user_input', '')}")

    try:
        user_input = state.get("user_input", "")

        if not user_input:
            logger.warning("사용자 입력이 비어있음")
            state["intent"] = "casual"
            state["keywords"] = []
            return state

        # LLM 초기화
        llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        # 프롬프트 생성
        prompt = f"""사용자의 영화 추천 요청을 분석하세요.

사용자 입력: "{user_input}"

다음 JSON 형식으로 응답하세요:
{{
    "intent": "search" | "recommend" | "similar",
    "keywords": ["키워드1", "키워드2", ...]
}}

의도 분류 기준:
- "search": 특정 영화를 검색 (예: "타이타닉 찾아줘")
- "recommend": 조건에 맞는 영화 추천 (예: "SF 영화 추천해줘")
- "similar": 특정 영화와 비슷한 영화 추천 (예: "인터스텔라 같은 영화")

키워드는 장르, 테마, 감정, 영화 제목 등을 추출하세요.

JSON만 출력하세요:"""

        # LLM 호출
        logger.debug("LLM 호출 중...")
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        logger.debug(f"LLM 응답: {response_text}")

        # JSON 파싱
        try:
            # JSON 블록 추출 (```json ... ``` 처리)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            intent = result.get("intent", "recommend")
            keywords = result.get("keywords", [])

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}, 응답: {response_text}")
            # 기본값 사용
            intent = "recommend"
            keywords = []

        # State 업데이트
        state["intent"] = intent
        state["keywords"] = keywords

        elapsed_time = time.time() - start_time
        logger.info(f"의도 분석 완료: intent={intent}, keywords={keywords}")
        logger.info(f"=== analyze_intent_node 완료 ({elapsed_time:.2f}초) ===\n")

        return state

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"의도 분석 실패: {e}")
        logger.error(f"=== analyze_intent_node 실패 ({elapsed_time:.2f}초) ===\n")

        # 에러 발생 시 기본값 설정
        state["intent"] = "recommend"
        state["keywords"] = []
        return state


def search_tmdb_node(state: AgentState) -> AgentState:
    """
    TMDB API를 사용하여 영화를 검색합니다.

    keywords를 기반으로:
    - 장르가 포함된 경우: discover API 사용
    - 영화 제목인 경우: search API 사용

    Args:
        state: 현재 Agent 상태

    Returns:
        업데이트된 Agent 상태 (tmdb_results 필드 수정)

    Raises:
        Exception: TMDB 검색 실패 시
    """
    start_time = time.time()
    logger.info("=== search_tmdb_node 시작 ===")

    try:
        keywords = state.get("keywords", [])
        intent = state.get("intent", "recommend")

        logger.info(f"검색 조건: intent={intent}, keywords={keywords}")

        # TMDB 클라이언트 초기화
        client = TMDBClient(api_key=TMDB_API_KEY, base_url=TMDB_BASE_URL)

        results = []

        # 장르 매핑 (한국어 -> 영어)
        genre_mapping = {
            "SF": 878,
            "액션": 28,
            "모험": 12,
            "애니메이션": 16,
            "코미디": 35,
            "범죄": 80,
            "다큐멘터리": 99,
            "드라마": 18,
            "가족": 10751,
            "판타지": 14,
            "역사": 36,
            "공포": 27,
            "음악": 10402,
            "미스터리": 9648,
            "로맨스": 10749,
            "스릴러": 53,
            "전쟁": 10752,
            "서부": 37
        }

        # 장르 추출
        genre_ids = []
        for keyword in keywords:
            if keyword in genre_mapping:
                genre_ids.append(genre_mapping[keyword])

        # 장르 기반 검색
        if genre_ids:
            logger.info(f"장르 기반 검색: genre_ids={genre_ids}")
            try:
                # 여러 페이지 수집 (최대 30개)
                for page in range(1, 3):  # 2페이지 = 40개
                    page_results = client.discover_by_genre(
                        genre_ids=genre_ids,
                        page=page
                    )
                    results.extend(page_results)

                    if len(results) >= 30:
                        break

                results = results[:30]
                logger.info(f"장르 검색 결과: {len(results)}개")

            except Exception as e:
                logger.error(f"장르 검색 실패: {e}")

        # 제목 기반 검색 (intent가 "search"이거나 장르 결과가 없을 때)
        if intent == "search" or (not results and keywords):
            logger.info(f"제목 기반 검색: keywords={keywords}")
            for keyword in keywords:
                try:
                    search_results = client.search_movies(query=keyword, page=1)
                    results.extend(search_results)

                    if len(results) >= 30:
                        break

                except Exception as e:
                    logger.error(f"제목 검색 실패 (keyword={keyword}): {e}")

            results = results[:30]
            logger.info(f"제목 검색 결과: {len(results)}개")

        # 기본 검색 (결과가 없을 때)
        if not results:
            logger.warning("검색 결과 없음, 인기 영화 반환")
            try:
                results = client.discover_by_genre(
                    genre_ids=[],  # 전체
                    page=1
                )[:30]
                logger.info(f"인기 영화 결과: {len(results)}개")

            except Exception as e:
                logger.error(f"인기 영화 검색 실패: {e}")
                results = []

        # State 업데이트
        state["tmdb_results"] = results

        elapsed_time = time.time() - start_time
        logger.info(f"TMDB 검색 완료: {len(results)}개 결과")
        logger.info(f"=== search_tmdb_node 완료 ({elapsed_time:.2f}초) ===\n")

        return state

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"TMDB 검색 실패: {e}")
        logger.error(f"=== search_tmdb_node 실패 ({elapsed_time:.2f}초) ===\n")

        # 에러 발생 시 빈 결과
        state["tmdb_results"] = []
        return state


def search_rag_node(state: AgentState) -> AgentState:
    """
    RAG 벡터 검색을 사용하여 영화를 검색합니다.

    intent에 따라:
    - "similar": 특정 영화 기반 검색
    - 그 외: 사용자 입력 + 프로필 기반 검색

    Args:
        state: 현재 Agent 상태

    Returns:
        업데이트된 Agent 상태 (rag_results 필드 수정)

    Raises:
        Exception: RAG 검색 실패 시
    """
    start_time = time.time()
    logger.info("=== search_rag_node 시작 ===")

    try:
        user_input = state.get("user_input", "")
        intent = state.get("intent", "recommend")
        keywords = state.get("keywords", [])
        user_profile = state.get("user_profile", {})

        logger.info(f"검색 조건: intent={intent}, user_input='{user_input}'")

        # MovieRAG 초기화
        rag = MovieRAG()

        results = []

        # "similar" 의도: 특정 영화 기반 검색
        if intent == "similar":
            logger.info("유사 영화 검색 모드")

            # 키워드에서 영화 제목 추출
            movie_title = None
            for keyword in keywords:
                # 장르가 아닌 키워드를 영화 제목으로 간주
                genre_list = ["SF", "액션", "모험", "애니메이션", "코미디", "범죄",
                              "다큐멘터리", "드라마", "가족", "판타지", "역사",
                              "공포", "음악", "미스터리", "로맨스", "스릴러", "전쟁", "서부"]
                if keyword not in genre_list:
                    movie_title = keyword
                    break

            if movie_title:
                logger.info(f"영화 제목으로 검색: '{movie_title}'")
                try:
                    # 제목으로 영화 찾기
                    search_results = rag.search_similar(
                        query=f"제목: {movie_title}",
                        k=1,
                        min_similarity=0.5
                    )

                    if search_results:
                        movie_id = search_results[0]["movie_id"]
                        logger.info(f"영화 발견: movie_id={movie_id}, title={search_results[0]['title']}")

                        # 유사 영화 검색
                        results = rag.search_by_movie_id(
                            movie_id=movie_id,
                            k=20,
                            min_similarity=0.7
                        )
                        logger.info(f"유사 영화 검색 결과: {len(results)}개")

                except Exception as e:
                    logger.error(f"유사 영화 검색 실패: {e}")

        # 일반 검색: 사용자 입력 기반
        if not results:
            logger.info("일반 RAG 검색 모드")

            # 사용자 프로필 반영
            query = user_input

            preferred_genres = user_profile.get("preferred_genres", [])
            mood_preferences = user_profile.get("mood_preferences", [])

            if preferred_genres or mood_preferences:
                query_parts = [user_input]

                if preferred_genres:
                    query_parts.append(f"선호 장르: {', '.join(preferred_genres)}")

                if mood_preferences:
                    query_parts.append(f"선호 분위기: {', '.join(mood_preferences)}")

                query = " ".join(query_parts)
                logger.info(f"프로필 반영 쿼리: '{query}'")

            try:
                results = rag.search_similar(
                    query=query,
                    k=20,
                    min_similarity=0.7
                )
                logger.info(f"RAG 검색 결과: {len(results)}개")

            except Exception as e:
                logger.error(f"RAG 검색 실패: {e}")
                results = []

        # State 업데이트
        state["rag_results"] = results

        elapsed_time = time.time() - start_time
        logger.info(f"RAG 검색 완료: {len(results)}개 결과")
        logger.info(f"=== search_rag_node 완료 ({elapsed_time:.2f}초) ===\n")

        return state

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG 검색 실패: {e}")
        logger.error(f"=== search_rag_node 실패 ({elapsed_time:.2f}초) ===\n")

        # 에러 발생 시 빈 결과
        state["rag_results"] = []
        return state


def rank_results_node(state: AgentState) -> AgentState:
    """
    TMDB와 RAG 검색 결과를 병합하고 순위를 매깁니다.

    스코어 계산 로직:
    - TMDB 평점 × 0.5
    - 인기도 × 0.3
    - RAG 유사도 × 2.0
    - 장르 매칭 × 1.5
    - 회피 장르 -5.0

    Args:
        state: 현재 Agent 상태

    Returns:
        업데이트된 Agent 상태 (final_movies, reasoning 필드 수정)

    Raises:
        Exception: 순위 계산 실패 시
    """
    start_time = time.time()
    logger.info("=== rank_results_node 시작 ===")

    try:
        tmdb_results = state.get("tmdb_results", [])
        rag_results = state.get("rag_results", [])
        user_profile = state.get("user_profile", {})
        keywords = state.get("keywords", [])

        logger.info(f"결과 병합: TMDB={len(tmdb_results)}개, RAG={len(rag_results)}개")

        # 영화 딕셔너리 (중복 제거용)
        movies_dict: Dict[int, Dict] = {}

        # TMDB 결과 처리
        for movie in tmdb_results:
            movie_id = movie.get("id")
            if not movie_id:
                continue

            movies_dict[movie_id] = {
                "movie_id": movie_id,
                "title": movie.get("title", ""),
                "genres": [g.get("name", "") for g in movie.get("genres", [])] if isinstance(movie.get("genres", []), list) and movie.get("genres") and isinstance(movie["genres"][0], dict) else movie.get("genres", []),
                "overview": movie.get("overview", ""),
                "vote_average": movie.get("vote_average", 0.0),
                "popularity": movie.get("popularity", 0.0),
                "release_date": movie.get("release_date", ""),
                "poster_path": movie.get("poster_path", ""),
                "rag_similarity": 0.0,  # 기본값
                "source": "tmdb"
            }

        # RAG 결과 처리 (덮어쓰기 또는 추가)
        for movie in rag_results:
            movie_id = movie.get("movie_id")
            if not movie_id:
                continue

            # 장르 문자열을 리스트로 변환
            genres = movie.get("genres", "")
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(",") if g.strip()]

            if movie_id in movies_dict:
                # 이미 존재하면 RAG 정보 추가
                movies_dict[movie_id]["rag_similarity"] = movie.get("similarity_score", 0.0)
                movies_dict[movie_id]["source"] = "both"
            else:
                # 새로 추가
                movies_dict[movie_id] = {
                    "movie_id": movie_id,
                    "title": movie.get("title", ""),
                    "genres": genres,
                    "overview": "",
                    "vote_average": movie.get("vote_average", 0.0),
                    "popularity": 0.0,
                    "release_date": movie.get("release_date", ""),
                    "poster_path": movie.get("poster_path", ""),
                    "rag_similarity": movie.get("similarity_score", 0.0),
                    "source": "rag"
                }

        logger.info(f"중복 제거 후: {len(movies_dict)}개 영화")

        # 스코어 계산
        preferred_genres = user_profile.get("preferred_genres", [])
        disliked_genres = []  # 향후 확장 가능

        scored_movies = []

        for movie in movies_dict.values():
            score = 0.0

            # TMDB 평점 × 0.5
            vote_average = movie.get("vote_average", 0.0)
            score += vote_average * 0.5

            # 인기도 × 0.3 (정규화: 0-10 범위로)
            popularity = movie.get("popularity", 0.0)
            normalized_popularity = min(popularity / 100, 10.0)
            score += normalized_popularity * 0.3

            # RAG 유사도 × 2.0
            rag_similarity = movie.get("rag_similarity", 0.0)
            score += rag_similarity * 2.0

            # 장르 매칭 × 1.5
            movie_genres = movie.get("genres", [])
            genre_match_count = 0

            for genre in movie_genres:
                if genre in preferred_genres or genre in keywords:
                    genre_match_count += 1

            score += genre_match_count * 1.5

            # 회피 장르 -5.0
            for genre in movie_genres:
                if genre in disliked_genres:
                    score -= 5.0

            movie["score"] = round(score, 2)
            scored_movies.append(movie)

        # 점수 순 정렬
        scored_movies.sort(key=lambda x: x["score"], reverse=True)

        # 상위 5개 선택
        final_movies = scored_movies[:5]

        logger.info(f"최종 영화 선택: {len(final_movies)}개")
        for idx, movie in enumerate(final_movies, 1):
            logger.info(f"  {idx}. {movie['title']} (점수: {movie['score']}, 출처: {movie['source']})")

        # Reasoning 생성
        reasoning_parts = []
        reasoning_parts.append(f"총 {len(movies_dict)}개 영화 중에서 상위 {len(final_movies)}개를 선택했습니다.")

        if preferred_genres:
            reasoning_parts.append(f"선호 장르 ({', '.join(preferred_genres)})를 고려했습니다.")

        if keywords:
            reasoning_parts.append(f"키워드 ({', '.join(keywords)})와의 연관성을 평가했습니다.")

        reasoning = " ".join(reasoning_parts)

        # State 업데이트
        state["final_movies"] = final_movies
        state["reasoning"] = reasoning

        elapsed_time = time.time() - start_time
        logger.info(f"순위 계산 완료")
        logger.info(f"=== rank_results_node 완료 ({elapsed_time:.2f}초) ===\n")

        return state

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"순위 계산 실패: {e}")
        logger.error(f"=== rank_results_node 실패 ({elapsed_time:.2f}초) ===\n")

        # 에러 발생 시 빈 결과
        state["final_movies"] = []
        state["reasoning"] = "영화 순위를 계산하는 중 오류가 발생했습니다."
        return state


def generate_response_node(state: AgentState) -> AgentState:
    """
    최종 추천 결과를 자연어로 생성합니다.

    GPT-4o를 사용하여:
    - 각 영화별 추천 이유 설명
    - 친근하고 자연스러운 톤
    - 사용자 프로필 고려

    Args:
        state: 현재 Agent 상태

    Returns:
        업데이트된 Agent 상태 (messages 필드에 응답 추가)

    Raises:
        Exception: 응답 생성 실패 시
    """
    start_time = time.time()
    logger.info("=== generate_response_node 시작 ===")

    try:
        final_movies = state.get("final_movies", [])
        reasoning = state.get("reasoning", "")
        user_input = state.get("user_input", "")

        if not final_movies:
            logger.warning("추천할 영화가 없음")
            response = "죄송합니다. 요청하신 조건에 맞는 영화를 찾지 못했습니다. 다른 조건으로 검색해보시겠어요?"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": response}]
            return state

        logger.info(f"응답 생성 중: {len(final_movies)}개 영화")

        # LLM 초기화 (GPT-4o)
        from src.config import GPT_RESPONSE_MODEL
        llm = ChatOpenAI(
            model=GPT_RESPONSE_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY
        )

        # 영화 정보 포맷팅
        movies_text = []
        for idx, movie in enumerate(final_movies, 1):
            movie_info = f"{idx}. {movie['title']}"

            if movie.get('genres'):
                genres = movie['genres'] if isinstance(movie['genres'], list) else [movie['genres']]
                movie_info += f" ({', '.join(genres)})"

            movie_info += f"\n   평점: {movie.get('vote_average', 'N/A')}"
            movie_info += f" | 점수: {movie.get('score', 'N/A')}"

            if movie.get('overview'):
                overview = movie['overview'][:100] + "..." if len(movie['overview']) > 100 else movie['overview']
                movie_info += f"\n   줄거리: {overview}"

            movies_text.append(movie_info)

        movies_formatted = "\n\n".join(movies_text)

        # 프롬프트 생성
        prompt = f"""사용자에게 영화를 추천하는 친근한 어시스턴트입니다.

사용자 요청: "{user_input}"

추천 영화 목록:
{movies_formatted}

추천 근거: {reasoning}

위 영화들을 사용자에게 추천하는 자연스러운 응답을 작성하세요:
- 각 영화가 왜 추천되었는지 간단히 설명
- 친근하고 대화하는 듯한 톤
- 사용자의 요청과 연결
- 3-5문장으로 간결하게

응답:"""

        # LLM 호출
        logger.debug("LLM 호출 중...")
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        logger.debug(f"LLM 응답 생성 완료: {len(response_text)} 글자")

        # State 업데이트 (messages에 추가)
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ]

        elapsed_time = time.time() - start_time
        logger.info(f"응답 생성 완료")
        logger.info(f"=== generate_response_node 완료 ({elapsed_time:.2f}초) ===\n")

        return state

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"응답 생성 실패: {e}")
        logger.error(f"=== generate_response_node 실패 ({elapsed_time:.2f}초) ===\n")

        # 에러 발생 시 기본 응답
        final_movies = state.get("final_movies", [])
        if final_movies:
            response = f"다음 영화들을 추천드립니다:\n\n"
            for idx, movie in enumerate(final_movies, 1):
                response += f"{idx}. {movie['title']}"
                if movie.get('vote_average'):
                    response += f" (평점: {movie['vote_average']})"
                response += "\n"
        else:
            response = "영화 추천 중 오류가 발생했습니다."

        state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": response}]
        return state
