import json
from src.cinema_expert import CinemaExpert
from src.models import CinemaExpertRequest
from src.config import get_config
from src.agent_tools import AgentTools
from openai import OpenAI


def startup_application() -> CinemaExpert:
    config = get_config()
    print(f"Using config: {config.db_path}")
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


def test_cinema_expert_flow():
    print("Starting smoke test for Cinema Expert...")

    # Initialize the expert
    expert = startup_application()

    # Create a sample request
    sample_request = {
        "user_input": "Recommend a good sci-fi movie from the 1980s",
    }

    expert_request = CinemaExpertRequest(**sample_request)
    response = expert.invoke(expert_request)
    print(response)

    sample_request = {
        "user_input": "What was the budget for the first matrix movie?",
    }

    expert_request = CinemaExpertRequest(**sample_request)
    response = expert.invoke(expert_request)
    print(response)

if __name__ == "__main__":
    import uuid

    success = test_cinema_expert_flow()
    exit(0 if success else 1)
