import json
import argparse
from typing import Dict, Any

import duckdb
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from structlog import get_logger
from openai import OpenAI

from src.config import get_config
from src.agent_tools import AgentTools
from src.cinema_expert import CinemaExpert
from src.models import CinemaExpertRequest

logger = get_logger("performance-assessment")


def startup_application() -> CinemaExpert:
    config = get_config()
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


def load_calicut_test(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def run_domain_knowledge_test(
    expert: CinemaExpert, test: Dict[str, Any]
) -> float:
    n_correct = 0

    for question in tqdm(test["questions"], desc="Domain Knowledge Test"):
        prompt = (
            "Please fill in the blank with the correct option. "
            "Only respond with the option.\n"
            f"Question: {question['question']}\n"
            f"Options: {question['options']}"
        )

        request = CinemaExpertRequest(user_input=prompt)
        response = expert.invoke(request)

        if question["answer"].lower() in response.generated_response.lower():
            n_correct += 1

    return n_correct / len(test["questions"])


def sample_users(n_users: int, database_path: str) -> pl.DataFrame:
    sql = f"""
    WITH user_review_counts AS (
        SELECT userId
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
        JOIN movie m ON u.movieId = m.movieId
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
        h.*,
        STRING_AGG(
            CASE WHEN u.rating > 3 AND u.movieId != h.holdout_movie_id
            THEN m.title END,
            ', '
        ) AS liked_movies_excluding_holdout
    FROM holdout_movies h
    JOIN user u ON h.userId = u.userId
    JOIN movie m ON u.movieId = m.movieId
    GROUP BY h.userId, h.holdout_movie_id, h.holdout_rating, h.holdout_title
    """

    with duckdb.connect(database_path) as conn:
        return conn.sql(sql).pl()


def run_embedding_recommendation_test(
    expert: CinemaExpert, users: pl.DataFrame, database_path: str
) -> Dict[str, float]:
    user_embeddings = []
    holdout_embeddings = []
    holdout_ratings = []

    for user in tqdm(users.iter_rows(named=True), desc="Embedding Eval"):
        user_prompt = f"I love {user['liked_movies_excluding_holdout']}. What should I watch tonight?"
        user_embeddings.append(expert.tools.model.encode(user_prompt))

        with duckdb.connect(database_path) as conn:
            emb = (
                conn.execute(
                    "SELECT embedding FROM movie WHERE movieId = ?",
                    [user["holdout_movie_id"]],
                )
                .fetchone()[0]
            )

        holdout_embeddings.append(np.array(emb))
        holdout_ratings.append(user["holdout_rating"])

    user_embeddings = np.array(user_embeddings)
    holdout_embeddings = np.array(holdout_embeddings)
    holdout_ratings = np.array(holdout_ratings)

    similarities = cosine_similarity(user_embeddings, holdout_embeddings).diagonal()
    r_value, p_value = pearsonr(similarities, holdout_ratings)

    return {
        "pearson_r": float(r_value),
        "p_value": float(p_value),
    }


def run_taste_classification_test(
    expert: CinemaExpert, users: pl.DataFrame
) -> float:
    n_sampled = 0
    n_correct = 0

    for user in tqdm(users.iter_rows(named=True), desc="Taste Classification"):
        prompt = (
            f"I love {user['liked_movies_excluding_holdout']}. "
            f"Would I like {user['holdout_title']}? "
            "Respond with only 'yes' or 'no'."
        )

        request = CinemaExpertRequest(user_input=prompt)
        response = expert.invoke(request)
        output = response.generated_response.strip().lower()

        if output not in {"yes", "no"}:
            continue

        n_sampled += 1
        ground_truth = "yes" if user["holdout_rating"] >= 3 else "no"

        if output == ground_truth:
            n_correct += 1

    return n_correct / n_sampled if n_sampled else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-test", action="store_true")
    parser.add_argument("--recommendation-test", action="store_true")
    parser.add_argument("--taste-test", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="run_results.json")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--db-path", default="data/cinemastore")
    parser.add_argument("--calicut-path", default="data/unv_calicult_eng410.json")
    args = parser.parse_args()

    expert = startup_application()
    results = {}

    if args.domain_test or args.all:
        test = load_calicut_test(args.calicut_path)
        score = run_domain_knowledge_test(expert, test)
        results["domain_knowledge_accuracy"] = score
        logger.info("DomainKnowledgeComplete", score=score)

    if args.recommendation_test or args.taste_test or args.all:
        users = sample_users(args.n_users, args.db_path)

    if args.recommendation_test or args.all:
        rec_results = run_embedding_recommendation_test(
            expert, users, args.db_path
        )
        results["embedding_recommendation"] = rec_results
        logger.info("RecommendationEvalComplete", **rec_results)

    if args.taste_test or args.all:
        score = run_taste_classification_test(expert, users)
        results["taste_classification_accuracy"] = score
        logger.info("TasteClassificationComplete", score=score)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("RunComplete", output=args.output)


if __name__ == "__main__":
    main()

