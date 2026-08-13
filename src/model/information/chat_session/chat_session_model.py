import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel
from datetime import datetime as datetime_type

class StreamSequence(str, Enum):
    START = "start"
    CHUNK = "chunk"
    END = "end"

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
    
class ChatMessageStreamDTO(BaseModel):
    stream: StreamSequence
    role: ChatRole | None = None
    content: str | None = None
    timestamp: datetime_type | None = None
    metadata: dict | None = None