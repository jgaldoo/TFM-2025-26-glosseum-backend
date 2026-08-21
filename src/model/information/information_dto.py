from enum import Enum
from typing import List

from pydantic import BaseModel

from src.model.common.stream_dto import StreamSequence
from src.model.technicism.technicism_dto import TechnicismDTO

class InformationType(str, Enum):
    GENERATED = "generated"
    TRANSCRIBED = "transcribed"

class InformationStreamProgress(str, Enum):
    SIMPLIFYING = "simplifying"
    FINDING_TECHNICISMS = "finding_technicisms"

class InformationDTO(BaseModel):
    title: str
    information_type: InformationType
    is_simplified: bool
    content: str
    technicisms: List[TechnicismDTO]

class InformationStreamDTO(BaseModel):
    stream: StreamSequence
    stream_info: InformationStreamProgress | None = None
    is_simplified: bool | None = None
    content: str | None = None
    technicisms: List[TechnicismDTO] | None = None

class InformationSimplificationRequest(BaseModel):
    title: str
    content: str