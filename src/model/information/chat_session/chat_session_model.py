import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel
from datetime import datetime as datetime_type

class ChatRole(str, Enum):
    USER = "user"
    MODEL = "model"

class ChatRequest(BaseModel):
    message: str

class SessionCreationRequest(BaseModel):
    document: str

class ChatSessionDTO(BaseModel):
    session_id: str
    history: List[ChatMessageDTO]

class ChatMessageDTO(BaseModel):
    role: ChatRole
    content: str
    timestamp: datetime_type
    metadata: dict