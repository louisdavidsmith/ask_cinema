from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
TMDB_KEY = os.getenv("TMDB_KEY")
GENERATIVE_MODEL_ID = os.getenv("GENERATIVE_MODEL_ID")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MOVIE_SEARCH_CUTOFF_HIGH = os.getenv("MOVIE_SEARCH_MIN_BAYESIAN_AVG_HIGH")
MOVIE_SEARCH_CUTOFF_LOW = os.getenv("MOVIE_SEARCH_MIN_BAYESIAN_AVG_LOW")
TOOL_CHOICE = os.getenv("TOOL_CHOICE")
DB_PATH = os.getenv("DB_PATH")


class Config(BaseModel):
    open_ai_key: str
    tmdb_api_key: str
    generative_model_id: str
    embedding_model: str
    movie_search_cutoff_high: float
    movie_search_cutoff_low: float
    tool_choice: str  # TODO: make enum
    db_path: str


def get_config() -> Config:
    return Config(
        open_ai_key=OPEN_AI_KEY,
        tmdb_api_key=TMDB_KEY,
        generative_model_id=GENERATIVE_MODEL_ID,
        embedding_model=EMBEDDING_MODEL,
        movie_search_cutoff_high=MOVIE_SEARCH_CUTOFF_HIGH,
        movie_search_cutoff_low=MOVIE_SEARCH_CUTOFF_LOW,
        tool_choice=TOOL_CHOICE,
        db_path=DB_PATH,
    )
