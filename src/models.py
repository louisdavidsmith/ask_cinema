from pydantic import BaseModel, Field
import uuid
from uuid import UUID
from dotenv import load_dotenv

load_dotenv()


class CinemaExpertRequest(BaseModel):
    user_input: str
    request_id: UUID = Field(default_factory=uuid.uuid4)
    conversation_id: UUID = Field(default_factory=uuid.uuid4)


class CinemaExpertResponse(BaseModel):
    generated_response: str
    user_input: str
    response_id: UUID = Field(default_factory=uuid.uuid4)
    conversation_id: UUID = Field(default_factory=uuid.uuid4)
