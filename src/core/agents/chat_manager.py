import uuid
from datetime import datetime

from src.model.information.chat_session.chat_session_model import ChatRole


class ChatMessage:
    role: ChatRole
    content: str
    timestamp: datetime
    metadata: dict

    def __init__(
            self,
            role: ChatRole,
            content: str,
            timestamp: datetime = datetime.now(),
            metadata=None
    ):
        if metadata is None:
            metadata = dict()

        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata

class ChatSession:
    def __init__(self, document: str):
        self.document = document
        self.history = []

    def add_message(self, message: ChatMessage):
        self.history.append(message)

class ChatManager:
    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def get_session(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def create_session(self, document) -> tuple[str, ChatSession] :
        session_id = str(uuid.uuid7())
        chat_session = ChatSession(document)
        self._sessions[session_id] = chat_session

        return session_id, chat_session

    def delete_session(self, session_id: str):
        self._sessions.pop(session_id)