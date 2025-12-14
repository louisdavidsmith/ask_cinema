import argparse
from typing import Tuple

import duckdb
import polars as pl
from sentence_transformers import SentenceTransformer
from structlog import get_logger
from dotenv import load_dotenv

load_dotenv()

embedding_model_id = os.getenv("EMBEDDING_MODEL")
DB_PATH = os.getenv("DB_PATH")

logger = get_logger("data-ingest")

# lets start out with a very small sentence transformer.
# I belive in starting with the smallest model first and then scaling only once you can
# measure performance.
# this is likely a place to start increasing performance
# TODO want to make sure this pulls from a central config at the root level.
model = SentenceTransformer(embedding_model_id)


def load_data(data_dir: str) -> Tuple[pl.DataFrame]:
    link_df = pl.read_csv(
        f"{data_dir}/links.csv", schema_overrides={"imdbId": pl.String}
    )
    movie_df = pl.read_csv(f"{data_dir}/movies.csv")
    rating_df = pl.read_csv(f"{data_dir}/ratings.csv")
    tags_df = pl.read_csv(f"{data_dir}/tags.csv")
    return link_df, movie_df, rating_df, tags_df


def generate_description_field(tags: pl.DataFrame, movies: pl.DataFrame):
    """
    This dataset contains the following fields:
        1. the titles of movies
        2. user generated tags
        3. curated genre metadata
    My idea is to combine them into a single feature string and then use
    an embedding model to embed.  i
    I concatenated together all of the tags for a given movie This works because of
    super position. The encoded representation of for example "pirate" and "zombie" are     unlikely to be mutually exclusive.
    So when I concat it all together it can contain information about both, and
    allow users to search for both 'vibe' and 'genre.'
    """
    movies_by_tags = tags.group_by("movieId").agg(pl.col("tag"))
    movies_by_tags = movies_by_tags.with_columns(pl.col("tag").list.unique())
    movies_by_tags = movies_by_tags.with_columns(pl.col("tag").list.join(", "))
    movies_by_tags = movies_by_tags.join(movies, on="movieId")
    movies_by_tags = movies_by_tags.with_columns(
        pl.concat_str(
            [
                pl.col("title"),
                pl.col("genres").str.split("|").list.join(", "),
                pl.col("tag"),
            ],
            separator=" ",
        ).alias("description")
    )
    return movies_by_tags


def bayesian_average(
    df: pl.DataFrame,
    rating_col: str = "mean_rating",
    n_rating_col: str = "n_rating",
    min_n_ratings: int = 1,
    prior_weight: float = 0.5,
) -> pl.DataFrame:
    """The idea here is that when we have very few reviews about movie, our
    uncertainty about its actually quailty is very high. If a movie has a
    single 5 star rating, and no other ratings, we want to pull that value
    toward the mean, modulated by how many reviews we have."""
    global_mean = (
        df.filter(pl.col(n_rating_col) >= min_n_ratings)
        .select(pl.col(rating_col).mean())
        .item()
    )

    bayesian_avg_expr = (
        pl.col(rating_col) * pl.col(n_rating_col) + global_mean * prior_weight
    ) / (pl.col(n_rating_col) + prior_weight)

    return df.with_columns(bayesian_avg=bayesian_avg_expr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process movie data and create a database."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the CSV files (links.csv, movies.csv, etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    db_name = args.db_name
    logger.info("DataIngestionConfig", data_dir=data_dir, db_name=db_name)
    link_df, movie_df, rating_df, tag_df = load_data(data_dir)
    logger.info("LoadedData")
    ratings_by_movie = rating_df.group_by("movieId").agg(
        [
            pl.col("rating").mean().alias("mean_rating"),
            pl.col("rating").count().alias("n_rating"),
        ]
    )
    movies_by_tags = generate_description_field(tag_df, movie_df)

    descriptions = movies_by_tags["description"].to_list()
    description_embeddings = model.encode(descriptions, show_progress_bar=True)
    logger.info("ScoredEmbeddings")
    movies_by_tags = movies_by_tags.with_columns(
        pl.lit(description_embeddings).alias("embedding")
    )

    movies_with_metadata = (
        movie_df.join(link_df, on="movieId")
        .join(movies_by_tags, on="movieId")
        .join(ratings_by_movie, on="movieId")
        .sort("n_rating")
    )
    movies_with_metadata = bayesian_average(movies_with_metadata)

    conn = duckdb.connect(db_name)

    conn.sql("install vss")
    # here i used create or replace to make sure this process is idempotent
    conn.sql("create or replace Table movie as select * from movies_with_metadata")
    conn.sql("create or replace Table user as select * from rating_df")
    conn.close()
    logger.info("DataIngested")


if __name__ == "__main__":
    main()
