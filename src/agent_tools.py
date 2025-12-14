# my decision to seperate this out is to allow for tool usage to be
# decoupled from the rest of the application. I can add new tools
# without breaking my contract
import json
from typing import Dict, List, Any
from .config import Config
import tmdbsimple as tmdb
import duckdb
from sentence_transformers import SentenceTransformer
from structlog import get_logger
import os

logger = get_logger("agent-tools")

TOOLS = [
    {
        "type": "function",
        "name": "get_movie_recommendation",
        "description": "Get a film recommendation",
        "parameters": {
            "type": "object",
            "properties": {
                "user_request": {"type": "string"},
                "k": {"type": "integer"},
                "user_desires_critically_acclaimed": {"type": "boolean"},
            },
            "required": ["user_request"],
        },
    },
    {
        "type": "function",
        "name": "get_movie_information",
        "description": """Get more information about a certain movie. Only search
        the name of the movie. For example: 'The Matrix'""",
        "parameters": {
            "type": "object",
            "properties": {
                "movie": {"type": "string"},
                "n_results": {"type": "integer"},
            },
            "required": ["movie"],
        },
    },
    {
        "type": "web_search",
        "filters": {
            "allowed_domains": [
                "cinephiliabeyond.org",
                "rightwalldarkroom.com",
                "filmanalysis.yale.edu",
                "researchguides.dartmouth.edu",
                "wikipedia.com",
            ]
        },
    },
]


class AgentTools:
    def __init__(self, config: Config):
        if not os.path.exists(config.db_path):
            raise FileNotFoundError("""The database does not exist.
                                    Please run make install before running this
                                    server""")
        self.tools = TOOLS
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        tmdb.API_KEY = config.tmdb_api_key

    def get_tools(self):
        # TODO: make web filters configurable without changing code
        return self.tools

    def _search(self, query, bayesian_avg, n_rating, k=10):
        embedding = self.model.encode(query)
        embedding = [[float(x) for x in embedding]]
        with duckdb.connect(self.config.db_path) as conn:
            out = conn.execute(
                f"""
                SELECT
                    movieId,
                    title,
                    bayesian_avg,
                    n_rating,
                    array_cosine_similarity(embedding, ?::FLOAT[384]) AS similarity
                FROM movie
                WHERE
                    bayesian_avg >= {bayesian_avg} and
                    n_rating >= {n_rating}
                ORDER BY similarity DESC
                LIMIT {k}
                """,
                embedding,
            ).fetchall()
        movie_recs = [x[1] for x in out]
        logger.info("MadeVectorSearch", query=query, returned_movies=movie_recs)
        return movie_recs

    def _get_movie_recommendation(
        self, user_request: str, k=10, user_desires_critically_acclaimed=False
    ) -> List[str]:
        # this is a place where the code and method could improve. I am leaving
        # the decision up to the llm on if the use wants a well known
        # or more niche movie. I think down the line you could
        # train a model using cross encoders to help the llm make
        # that determination. It would just be another tool.
        # TODO: make these values configurable
        if user_desires_critically_acclaimed:
            return self._search(
                user_request, self.config.movie_search_cutoff_high, 50, k
            )
        else:
            return self._search(user_request, self.config.movie_search_cutoff_low, 1, k)

    def _get_movie_information(self, movie: str, n_results=3) -> Dict[str, Any]:
        search = tmdb.Search()
        result = search.movie(query=movie)
        out = []
        ids = [result["results"][x]["id"] for x in range(n_results)]
        for id in ids:
            movie_info = {}
            movie = tmdb.Movies(id)
            movie_info["info"] = movie.info()
            movie_info["credits"] = movie.credits()
            out.append(movie_info)
        logger.info("CompletedMovieSearch", movie=movie)
        return out

    def handle_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        # TODO: come back and clean up this logic flow. Looking at this this is
        # pretty brittle and not as extensible as I'd like. Probably want a
        # general abstraction over tool calls.
        for item in messages:
            if item.type == "function_call":
                if item.name == "get_movie_recommendation":
                    result = self._get_movie_recommendation(
                        **json.loads(item.arguments)
                    )
                    output.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"movie_recommendation": result}),
                        }
                    )
                elif item.name == "get_movie_information":
                    result = self._get_movie_information(**json.loads(item.arguments))
                    output.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"movie_information": result}),
                        }
                    )

        return output
