from typing import Any, Generator

from google import genai
from google.genai.types import Part, Content, GenerateContentConfig
from google.oauth2 import service_account

from src.core.agents.chat_manager import ChatSession, ChatMessage
from src.core.config import settings

def build_contents(session: ChatSession, question: str) -> list[Content]:
    contents = []

    for message in session.history:
        contents.append(
            Content(
                role=message.role,
                parts=[
                    Part(
                        text=message.content
                    )
                ]
            )
        )

    # Current user question, including document context
    contents.append(
        Content(
            role="user",
            parts=[
                Part(
                    text=f"{question}"
                ),
            ]
        )
    )

    return contents

def extract_text(response) -> str:
    if response.text:
        return response.text

    if response.candidates:
        parts = response.candidates[0].content.parts
        return "".join(
            part.text
            for part in parts
            if part.text
        )

    return ""

class GeminiService:
    def __init__(self):
        self.client=genai.Client(
            vertexai=True,
            project=settings.google_project_id,
            credentials=service_account.Credentials.from_service_account_file(
                settings.gemini_api_credentials,
            )
            .with_scopes([
                "https://www.googleapis.com/auth/cloud-platform"
            ]),
            location=settings.gemini_api_location
        )

    def chat_stream(self, session: ChatSession, question: str) -> Generator[str, Any, None]:
        # Generate response
        response = self.client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=build_contents(session, question),
            config=GenerateContentConfig(
                system_instruction=f"""
                    Tu tarea es responder preguntas sobre el siguiente texto:
                    {session.document}

                    Cuando sea posible, usa la información contenida en el texto.
                    Si no se puede dar una respuesta con la información del texto,
                    haz uso de tu conocimiento para responder.
                    Se objetivo, y da respuestas breves y concisas.
                    Responde con texto plano, sin formato.
                """,
            ),
        )

        for chunk in response:
            text = extract_text(chunk)

            if text:
                yield text
