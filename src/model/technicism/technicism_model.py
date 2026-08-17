from typing import List

from pydantic import BaseModel

class Definition(BaseModel):
    text: str
    domain: str

class TechnicismOccurrence(BaseModel):
    form: str
    position: int
    confidence: float

class Technicism(BaseModel):
    canonical_name: str
    definitions: List[Definition]
    occurrences: List[TechnicismOccurrence]

class TechnicismTextInput(BaseModel):
    title: str
    text: str