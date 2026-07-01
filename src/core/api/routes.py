from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, HTTPException
from src.core.services.google_cloud_vision_service import VisionService, get_text_paragraphs
from src.core.services.ollama_service import OllamaService
from src.model.information.information_dto import InformationDTO, InformationType
from src.model.information.information_model import InformationLabel

router = APIRouter()

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

    if extracted_text is None or not extracted_text.text:
        raise HTTPException(
            status_code=400,
            detail="No se ha extraido ningún texto de la imagen."
        )

    content = get_text_paragraphs(extracted_text)


    # Determine the title of the text
    title, classified_lines = await OllamaService().classify_lines(content)

    inventory_text = [line.text for line in classified_lines if line.label == InformationLabel.INVENTORY]

    # Join all text lines together ignoring inventory
    full_text = "\n".join(content_item["text"] for content_item in content if content_item["text"] not in inventory_text)

    information_dto = InformationDTO(
        title=title,
        content=full_text,
        informationType=InformationType.transcribed,
        isSimplified=False
    )

    return information_dto
