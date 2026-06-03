from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def status():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

