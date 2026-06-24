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
    """
    filename = photo.filename

    if not filename:
        filename = uuid.uuid4().hex

    name, ext = os.path.splitext(filename)

    if not ext:
        ext = ".jpg"

    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
    if not safe_name:
        safe_name = uuid.uuid4().hex

    file_path = UPLOAD_DIR / (safe_name + ext)

    with file_path.open("wb") as buffer: # type: BinaryIO
        shutil.copyfileobj(photo.file, buffer)

    return InformationDTO(
        title=datetime.now().isoformat(),
        informationType=InformationType.transcribed,
        isSimplified=False,
        content="This is filler content"
    )
    
    ....
    def extract_words(response):
    words = []
    for page in extracted_text.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    text = "".join(s.text for s in word.symbols)
                    verts = [{"x": v.x, "y": v.y} for v in word.bounding_box.vertices]
                    words.append({"text": text, "vertices": verts})
    return words
    ....
    """
