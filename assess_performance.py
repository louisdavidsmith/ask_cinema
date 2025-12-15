import json
from src.config import get_config
from src.agent_tools import AgentTools
from openai import OpenAI
from src.cinema_expert import CinemaExpert
from typing import Dict, Any
from src.models import CinemaExpertRequest
from tqdm import tqdm
from structlog import get_logger
import polars as pl
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import duckdb

logger = get_logger("performance-assessment")


def startup_application() -> CinemaExpert:
    config = get_config()
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


def load_calicut_test():
    with open("data/unv_calicult_eng410.json") as f:
        data = json.loads(f.read())
    return data


def take_calicut_test(expert: CinemaExpert, test: Dict[str, Any]) -> float:
    n_questions = len(test["questions"])
    n_correct = 0
    for question in tqdm(test["questions"]):
        q = question["question"]
        o = question["options"]
        prompt = f"""Please fill in the blank with the correction option. Only
        respond with the option. Question: {q} Options:
            {o}"""
        request = CinemaExpertRequest(user_input=prompt)
        response = expert.invoke(request)
        if question["answer"].lower() in response.generated_response.lower():
            n_correct += 1
    return n_correct / n_questions


def sample_users(n_users: int, database_path: str) -> pl.DataFrame():
    sql = f"""WITH user_review_counts AS (
        SELECT
            userId,
            COUNT(*) AS review_count
        FROM user
        GROUP BY userId
        HAVING COUNT(*) >= 5
        LIMIT {n_users}
    ),

    user_movies AS (
        SELECT
            urc.userId,
            u.movieId,
            u.rating,
            m.title,
            ROW_NUMBER() OVER (PARTITION BY urc.userId ORDER BY RANDOM()) AS rn
        FROM user_review_counts urc
        JOIN user u ON urc.userId = u.userId
        JOIN movie m ON u.movieId = m.movieId  -- Fixed join condition
    ),

    holdout_movies AS (
        SELECT
            userId,
            movieId AS holdout_movie_id,
            rating AS holdout_rating,
            title AS holdout_title
        FROM user_movies
        WHERE rn = 1
    )

    SELECT
        urc.userId,
        h.holdout_movie_id,
        h.holdout_rating,
        h.holdout_title,
        ARRAY_AGG(
            CASE WHEN u.movieId != h.holdout_movie_id THEN STRUCT_PACK(u.movieId, u.rating) END
        ) AS all_movie_ratings_excluding_holdout,
        STRING_AGG(
            CASE WHEN u.movieId != h.holdout_movie_id AND u.rating > 3 THEN m.title END,
            ', '
        ) AS liked_movies_excluding_holdout,
        COUNT(CASE WHEN u.movieId != h.holdout_movie_id THEN 1 END) AS total_movies_excluding_holdout,
        COUNT(CASE WHEN u.rating > 3 AND u.movieId != h.holdout_movie_id THEN 1 END) AS liked_movie_count_excluding_holdout
    FROM user_review_counts urc
    JOIN user u ON urc.userId = u.userId
    JOIN movie m ON u.movieId = m.movieId  -- Fixed join condition
    JOIN holdout_movies h ON urc.userId = h.userId
    GROUP BY urc.userId, h.holdout_movie_id, h.holdout_rating, h.holdout_title
    """
    with duckdb.connect(database_path) as conn:
        df = conn.sql(sql).pl()
    return df


if __name__ == "__main__":
    expert = startup_application()
    test = load_calicut_test()
    logger.info("LoadedDomainKnowledgeTest", n_questions=len(test["questions"]))
    performance = take_calicut_test(expert, test)
    logger.info("AssessedAgentDomainKnowledge", score=performance)

    logger.info("AssessingRecommendationAlgorithm")

    n = 10000
    database_path = "data/cinemastore"

    users = sample_users(n, database_path)

    user_embeddings = []
    holdout_embeddings = []
    holdout_ratings = []

    for user in tqdm(users.iter_rows(named=True)):
        user_prompt = f"I love {user['liked_movies_excluding_holdout']}. What should I watch tonight?"

        user_embedding = expert.tools.model.encode(user_prompt)
        user_embeddings.append(user_embedding)

        with duckdb.connect(database_path) as conn:
            holdout_embedding = (
                conn.execute(
                    """
                SELECT embedding
                FROM movie
                WHERE movieId = ?
            """,
                    [user["holdout_movie_id"]],
                )
                .pl()
                .item()
            )

        holdout_embeddings.append(holdout_embedding)
        holdout_ratings.append(user["holdout_rating"])

    user_embeddings = np.array(user_embeddings)
    holdout_embeddings = np.array(holdout_embeddings)
    holdout_ratings = np.array(holdout_ratings)

    similarities = cosine_similarity(user_embeddings, holdout_embeddings)
    similarities = similarities.diagonal()

    r_value, p_value = pearsonr(similarities, holdout_ratings)

    logger.info(
        "RecommendationAlgorithmPerformance",
        n_users=n,
        pearson_r=r_value,
        p_value=p_value,
    )
