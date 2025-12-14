from openai import OpenAI
from typing import List, Dict, Any
from .models import CinemaExpertRequest, CinemaExpertResponse
from .config import Config
from .agent_tools import AgentTools
from .prompt import SYSTEM_PROMPT
from structlog import get_logger

logger = get_logger("app")


class CinemaExpert:
    def __init__(self, config: Config, openai_client: OpenAI, agent_tools: AgentTools):
        self.llm_client = openai_client
        self.tools = agent_tools
        self.config = config
        logger.info("CinemaExpertInitalized")

    def _format_message(self, request: CinemaExpertRequest) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.user_input},
        ]
        return messages

    def invoke(self, request: CinemaExpertRequest) -> CinemaExpertResponse:
        # TODO: would eventually like this to be llm provider agnostic. I would
        # abtract out the llm client so you could swap between bedrock,
        # anthropic ext.
        initial_messages = self._format_message(request)
        resp1 = self.llm_client.responses.create(
            model=self.config.generative_model_id,
            tools=self.tools.get_tools(),
            tool_choice=self.config.tool_choice,
            input=initial_messages,
        )

        messages = initial_messages + resp1.output

        tool_output = self.tools.handle_tool_calls(resp1.output)

        messages += tool_output

        resp2 = self.llm_client.responses.create(
            model=self.config.generative_model_id,
            tools=self.tools.get_tools(),
            input=messages,
        )

        return CinemaExpertResponse(
            generated_response=resp2.output_text,
            conversation_id=request.conversation_id,
            user_input=request.user_input,
        )
