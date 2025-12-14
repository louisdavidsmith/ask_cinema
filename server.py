from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from uuid import UUID
import json
from src.cinema_expert import CinemaExpert
from src.models import CinemaExpertRequest
from src.config import get_config
from src.agent_tools import AgentTools
from openai import OpenAI


def startup_application() -> CinemaExpert:
    config = get_config()
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


expert = startup_application()


async def cinema_expert_endpoint(request: Request):
    try:
        body = await request.json()
        expert_request = CinemaExpertRequest(**body)

        response = expert.invoke(expert_request)

        response_dict = response.model_dump()

        # Convert UUID fields to strings
        for key, value in response_dict.items():
            if isinstance(value, UUID):
                response_dict[key] = str(value)

        return JSONResponse(response_dict)
    except ValidationError as e:
        return JSONResponse(
            {"error": "Invalid request format", "details": str(e)}, status_code=422
        )
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON format"}, status_code=400)
    except Exception as e:
        return JSONResponse(
            {"error": "Internal server error", "details": str(e)}, status_code=500
        )


async def health_check(request: Request):
    return JSONResponse({"status": "healthy"})


middleware = [
    Middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
]

app = Starlette(
    routes=[
        Route("/cinema-expert", cinema_expert_endpoint, methods=["POST"]),
        Route("/health", health_check, methods=["GET"]),
    ],
    middleware=middleware,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
