from enum import Enum

from pydantic import BaseModel

class InformationType(str, Enum):
    generated = "generated"
    transcribed = "transcribed"

class InformationDTO(BaseModel):
    title: str
    informationType: InformationType
    isSimplified: bool
    content: str