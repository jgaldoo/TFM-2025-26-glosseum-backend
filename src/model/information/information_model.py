from enum import Enum

from pydantic import BaseModel

class InformationLabel(str, Enum):
    TITLE = "title"
    TITLE_TRANSLATION = "title_translation"
    ARTIST = "artist"
    ARTIST_DATES = "artist_dates"
    DATE = "date"
    MEDIUM = "medium"
    DATE_AND_MEDIUM = "date_and_medium"
    COLLECTION = "collection"
    INVENTORY = "inventory"
    DESCRIPTION = "description"
    DESCRIPTION_TRANSLATION = "description_translation"
    OTHER = "other"

class TitleCandidateLine(BaseModel):
    id: int
    confidence: float

class EnrichedTitleCandidateLine(BaseModel):
    text: str
    confidence: float

class InformationLine(BaseModel):
    id: int
    label: InformationLabel

class EnrichedInformationLine(BaseModel):
    text: str
    label: InformationLabel

class SimplifiedInformation(BaseModel):
    text: str
