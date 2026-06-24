import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException

from src.core.services.google_cloud_vision_service import VisionService, get_text_paragraphs
from src.core.utils.text_utils import clean_punctuation
from src.model.information.information_dto import InformationDTO

router = APIRouter()

extracted_text_global = None
content_global = None

@router.get("/")
async def status():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# Testing endpoint to not call the API services everytime one needs the processed data
@router.get("/test")
async def test():
    content = get_text_paragraphs(extracted_text_global)

    global content_global
    content_global = content
    processed_result = clean_punctuation("\n".join(content))
    logging.log(logging.INFO, f"%s", processed_result)

    return {"extracted": processed_result}

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

    extracted_text = VisionService().detect_text(image_content)

    if not extracted_text.text:
        raise HTTPException(
            status_code=400,
            detail="No se ha extraido ningún texto de la imagen."
        )

    global extracted_text_global
    extracted_text_global = extracted_text

    content = get_text_paragraphs(extracted_text)

    global content_global
    content_global = content

    full_text = clean_punctuation("\n".join(content))

    # return full_text

    raise HTTPException(
        status_code=501,
        detail="Aún no se ha implementado esta funcionalidad."
    )
