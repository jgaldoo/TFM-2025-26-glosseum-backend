from typing import List

from pydantic import BaseModel

from src.model.technicism.technicism_model import (
    Technicism,
    Definition,
    TechnicismOccurrence,
)


class DefinitionDTO(BaseModel):
    text: str
    domain: str

class TechnicismOccurrenceDTO(BaseModel):
    form: str
    position: int

class TechnicismDTO(BaseModel):
    canonical_name: str
    definitions: List[DefinitionDTO]
    occurrences: List[TechnicismOccurrenceDTO]


def definition_to_DTO(definition: Definition) -> DefinitionDTO:
    return DefinitionDTO(
        text=definition.text,
        domain=definition.domain,
    )

def technicism_occurrence_to_DTO(technicism_occurrence: TechnicismOccurrence) -> TechnicismOccurrenceDTO:
    return TechnicismOccurrenceDTO(
        form=technicism_occurrence.form,
        position=technicism_occurrence.position,
    )

def technicism_to_DTO(technicism : Technicism) -> TechnicismDTO:
    return TechnicismDTO(
        canonical_name=technicism.canonical_name,
        definitions=list(map(definition_to_DTO, technicism.definitions)),
        occurrences=list(map(technicism_occurrence_to_DTO, technicism.occurrences)),
    )