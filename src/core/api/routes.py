import logging
from datetime import datetime, timezone
from pathlib import Path

import json
import time
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from google.cloud.vision_v1 import TextAnnotation
from proto import Message

from src.core.agents.chat_manager import ChatManager, ChatMessage
from src.core.config import settings
from src.core.services.gemini_service import GeminiService
from src.core.services.google_cloud_vision_service import VisionService, get_text_paragraphs
from src.core.services.ollama_service import OllamaService, AgentType
from src.model.information.chat_session.chat_session_model import (
    ChatRequest,
    SessionCreationRequest,
    ChatSessionDTO,
    ChatMessageDTO,
    ChatRole,
)
from src.model.information.information_dto import InformationDTO, InformationType
from src.model.information.information_model import (
    InformationLabel,
    EnrichedInformationLine,
    EnrichedTitleCandidateLine,
)

router = APIRouter()

chat_manager = ChatManager()

@router.get("/")
async def status():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.post("/transcribe") #, response_model=InformationDTO)
async def transcribe(
    photo: UploadFile = File(...),
):
    if photo.content_type is None or not photo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El contenido de la petición no es una imagen."
        )

    image_content = await photo.read()

    # Obtain text from image
    extracted_text = VisionService().detect_text(image_content)

    if extracted_text is None or len(extracted_text.text) == 0:
        raise HTTPException(
            status_code=400,
            detail="No se ha extraido ningún texto de la imagen."
        )

    content = get_text_paragraphs(extracted_text)


    # Determine the title of the text
    title, classified_lines = await OllamaService().agent_fleet.get(AgentType.CLASSIFICATION).classify_lines(content)

    inventory_text = [line.text for line in classified_lines if line.label == InformationLabel.INVENTORY]

    # Join all text lines together ignoring inventory
    full_text = "\n\n".join(content_item["text"] for content_item in content if content_item["text"] not in inventory_text)

    information_dto = InformationDTO(
        title=title,
        content=full_text,
        information_type=InformationType.transcribed,
        is_simplified=False
    )

    return information_dto


@router.post("/create-session")
async def create_session(
    session_creation_request: SessionCreationRequest
):
    session_id, chat_session = chat_manager.create_session(session_creation_request.document)

    return ChatSessionDTO(
        session_id=session_id,
        history=chat_session.history
    )

@router.post("/chat/{session_id}")
async def ask_question(
    session_id: str,
    chat_request: ChatRequest,
):
    chat_session = chat_manager.get_session(session_id)

    if chat_session is None:
        raise HTTPException(status_code=404, detail="No se ha encontrado una sesión con ese id.")


    user_message = ChatMessage(role=ChatRole.USER, content=chat_request.message)
    chat_session.add_message(user_message)

    response = ChatMessage(role=ChatRole.MODEL, content=GeminiService()
                           .chat(chat_session, chat_request.message))
    
    chat_session.add_message(response)

    return ChatMessageDTO(
        role=response.role,
        content=response.content,
        timestamp=response.timestamp,
        metadata=response.metadata
    )
    