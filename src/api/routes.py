from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException

from src.model.information.information_dto import InformationDTO

router = APIRouter()

@router.get("/")
async def status():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
@router.post("/transcribe", response_model=InformationDTO)
async def transcribe(
    photo: UploadFile = File(...),
):
    raise HTTPException(
        status_code=501,
        detail="Aún no se ha implementado esta funcionalidad."
    )

