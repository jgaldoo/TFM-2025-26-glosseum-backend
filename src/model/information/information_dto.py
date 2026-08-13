from enum import Enum
from typing import List

from pydantic import BaseModel

from src.model.technicism.technicism_dto import TechnicismDTO

class InformationType(str, Enum):
    generated = "generated"
    transcribed = "transcribed"

class InformationDTO(BaseModel):
    title: str
    information_type: InformationType
    is_simplified: bool
    content: str
    technicisms: List[TechnicismDTO]
