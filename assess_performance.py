import json
from src.config import get_config
from src.agent_tools import AgentTools
from openai import OpenAI
from src.cinema_expert import CinemaExpert
from typing import Dict, Any
from src.models import CinemaExpertRequest
from tqdm import tqdm


def startup_application() -> CinemaExpert:
    config = get_config()
    client = OpenAI(api_key=config.open_ai_key)
    tools = AgentTools(config)
    return CinemaExpert(config, client, tools)


def load_calicut_test():
    with open("performance_report/unv_calicult_eng410.json") as f:
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


if __name__ == "__main__":
    performance = {}
    expert = startup_application()
    test = load_calicut_test()
    performance = take_calicut_test(expert, test)
    calicut_test["score"] = performance

