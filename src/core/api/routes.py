import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, HTTPException
from starlette.responses import StreamingResponse

from src.core.agents.chat_manager import ChatManager, ChatMessage
from src.core.agents.technicism_expert import annotate_technicisms
from src.core.services.gemini_service import GeminiService
from src.core.services.google_cloud_vision_service import VisionService, get_text_paragraphs
from src.core.services.ollama_service import OllamaService, AgentType
from src.model.information.chat_session.chat_session_model import (
    ChatRequest,
    SessionCreationRequest,
    ChatSessionDTO,
    ChatMessageDTO,
    ChatRole,
    ChatMessageStreamDTO, StreamSequence,
)
from src.model.information.information_dto import InformationDTO, InformationType, InformationSimplificationRequest, \
    InformationStreamDTO, InformationStreamProgress
from src.model.information.information_model import InformationLabel
from src.model.technicism.technicism_dto import technicism_to_DTO
from src.model.technicism.technicism_model import TechnicismTextInput

router = APIRouter()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

gemini_service = GeminiService()
ollama_service = OllamaService()
chat_manager = ChatManager()

@router.get("/")
async def status():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.post("/transcribe", response_model=InformationDTO)
async def transcribe(
    photo: UploadFile = File(...),
):
    if photo.content_type is None or not photo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El contenido de la petición no es una imagen."
        )

    image_content = await photo.read()
    
    total_time = 0
    start_time = time.perf_counter()
    
    # Obtain text from image
    extracted_text = VisionService().detect_text(image_content)
    
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    log.debug(f"Text extraction: {time_taken}s")
    total_time += time_taken

    if extracted_text is None or len(extracted_text.text) == 0:
        raise HTTPException(
            status_code=400,
            detail="No se ha extraido ningún texto de la imagen."
        )

    start_time = time.perf_counter()
    
    content = get_text_paragraphs(extracted_text)
    
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    log.debug(f"Paragraph parsing: {time_taken}s")
    total_time += time_taken

    start_time = time.perf_counter()

    # Determine the title of the text
    title, classified_lines = await ollama_service.agent_fleet.get(AgentType.CLASSIFICATION).classify_lines(content)

    non_content_text = [
        line.text for line in classified_lines if
        line.label == InformationLabel.INVENTORY or line.label == InformationLabel.TITLE
    ]

    # Join all text lines together ignoring inventory
    full_text = "\n\n".join(content_item["text"] for content_item in content if content_item["text"] not in non_content_text)

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    log.debug(f"Text classification: {time_taken}s")
    total_time += time_taken

    start_time = time.perf_counter()

    # Identify technicisms obtained from the text
    text_technicisms = await (ollama_service.agent_fleet.get(AgentType.TECHNICISM)
                              .get_technicisms(TechnicismTextInput(title=title, text=full_text)))

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    log.debug(f"Technicism identification: {time_taken}s")
    total_time += time_taken

    log.info(f"Transcribe time: {total_time}s")

    information_dto = InformationDTO(
        title=title,
        content=annotate_technicisms(full_text, text_technicisms),
        technicisms=list(map(technicism_to_DTO, text_technicisms)),
        information_type=InformationType.TRANSCRIBED,
        is_simplified=False
    )

    log.info(f"InformationDTO: {information_dto.model_dump_json()}")

    return information_dto


@router.post("/create-session", response_model=ChatSessionDTO)
async def create_session(
    session_creation_request: SessionCreationRequest
):
    session_id, chat_session = chat_manager.create_session(session_creation_request.document)

    return ChatSessionDTO(
        session_id=session_id,
        history=list(
            map(
                lambda message : ChatMessageDTO(
                    role=message.role,
                    content=message.content,
                    timestamp=message.timestamp,
                    metadata=message.metadata
                ),
                chat_session.history
            )
        )
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

@router.post("/chat_stream/{session_id}", response_model=ChatMessageStreamDTO)
async def stream_ask_question(
        session_id: str,
        chat_request: ChatRequest,
    ):
        chat_session = chat_manager.get_session(session_id)

        if chat_session is None:
            raise HTTPException(
                status_code=404, detail="No se ha encontrado una sesión con ese id."
            )

        return StreamingResponse(
            ndjson_stream(chat_stream(session_id, chat_request)),
            media_type="application/x-ndjson",
        )

@router.post("/simplify_stream")
async def simplify_text(
    information_simplification_request: InformationSimplificationRequest
):
    return StreamingResponse(
            ndjson_stream(
                simplify_stream(information_simplification_request)
            ),
            media_type="application/x-ndjson",
        )

async def simplify_stream(information_simplification_request: InformationSimplificationRequest):
    full_text=""

    # First message sent to Flutter
    yield InformationStreamDTO(
        stream=StreamSequence.START,
        stream_info=InformationStreamProgress.SIMPLIFYING,
        content=""
    )

    async for chunk in ollama_service.agent_fleet.get(AgentType.SIMPLIFICATION).simplify(
        information_simplification_request
    ):
        full_text += chunk

        yield InformationStreamDTO(
            stream=StreamSequence.CHUNK,
            stream_info=InformationStreamProgress.SIMPLIFYING,
            content=chunk
        )

    yield InformationStreamDTO(
        stream=StreamSequence.CHUNK,
        stream_info=InformationStreamProgress.FINDING_TECHNICISMS
    )

    technicism_list = await (ollama_service.agent_fleet.get(AgentType.TECHNICISM)
                .get_technicisms(
                    TechnicismTextInput(title=information_simplification_request.title, text=full_text)
                )
            )

    yield InformationStreamDTO(
        stream=StreamSequence.CHUNK,
        content=annotate_technicisms(full_text, technicism_list),
        technicisms=list(map(technicism_to_DTO, technicism_list))
    )

    yield InformationStreamDTO(
        stream=StreamSequence.END,
        is_simplified=True
    )

async def chat_stream(session_id: str, chat_request: ChatRequest):
    chat_session = chat_manager.get_session(session_id)

    if chat_session is None:
        yield {"stream": "error", "detail": "Session not found"}
        return

    user_message = ChatMessage(
        role=ChatRole.USER,
        content=chat_request.message
    )
    chat_session.add_message(user_message)

    # First message sent to Flutter
    yield ChatMessageStreamDTO(
        stream=StreamSequence.START,
        role=ChatRole.MODEL,
        timestamp=datetime.now()
    )

    model_message = ""

    # Start streaming here
    for chunk in (gemini_service
            .chat_stream(chat_session, chat_request.message)):
        model_message += chunk

        yield ChatMessageStreamDTO(
            stream=StreamSequence.CHUNK,
            content=chunk
        )

    chat_session.add_message(
        ChatMessage(
            role=ChatRole.MODEL,
            content=model_message,
        )
    )

    yield ChatMessageStreamDTO(
        stream=StreamSequence.END
    )


async def ndjson_stream(stream):
    async for event in stream:
        processed_event = event.model_dump_json() + "\n"
        logging.log(logging.INFO, processed_event)
        yield processed_event
