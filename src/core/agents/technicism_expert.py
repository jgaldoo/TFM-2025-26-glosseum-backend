import re
from typing import List
import numpy as np

from scipy.optimize import linear_sum_assignment

from pydantic import TypeAdapter
from pydantic_ai import Agent, NativeOutput, ModelSettings

from model.technicism.technicism_model import Technicism
from src.model.information.information_model import InformationLine
from src.model.technicism.technicism_model import (
    Technicism,
    TechnicismOccurrence,
    TechnicismTextInput,
)

# Prompt to identify technicisms and domains within a text
technicism_expert_prompt = """
Eres un sistema experto en terminología, encargado de encontrar tecnicismos en textos y dar definiciones 
a esos tecnicismos.

Un tecnicismo es:
- Un término especializado utilizado en un dominio de conocimiento concreto.
- Una palabra cuyo significado depende del conocimiento del dominio en el que se utilice.
- Una palabra compleja que pueda no conocer una persona experimentada en el dominio de conocimiento en el que se menciona.

Recibirás un título y un texto hablando de un tema en específico siguiendo este formato: 
{
    "title": str,
    "text": str,
}

Tu tarea es devolver una lista de los tecnicismos encontrados únicamente en el texto,
 junto con el nombre canónico (la forma básica escrita) del tecnicismo. Cada tecnicismo encontrado posee
una lista de definiciones asociada, con la definicion y el dominio de la definición
(arte, poesía, informática, biología, historia, etc.). Cada tecnicismo encontrado también posee una lista de sus
apariciones en el texto, que incluyen el tecnicismo, el índice de su posición de inicio en el texto, y cómo de seguro
estás de que esa palabra esté asociada al tecnicismo al que relacionas. Todo esto debe estar en el mismo idioma
en el que está el texto.


Reglas:
1. Analiza el texto primero, y luego identifica los tecnicismos.
2. Las definiciones de los tecnicismos NO deben tener otros tecnicismos.
4. Las definiciones de los tecnicismos deben poder entenderse de manera sencilla por personas de 12 años.
5. Las palabras comunes NO son tecnicismos.
6. Los plurales NO son nombres canónicos.
7. Las serie de palabras comunes NO son tecnicismos.
8. Las palabras que una persona de 12 años pueda deducir por el contexto NO son tecnicismos.
9. Si no hay definición para un posible tecnicismo, NO es un tecnicismo.
10. Todo tecnicismo encontrado debe aparecer en el texto.

Responde SOLO con una lista JSON válida:
[
  {
    "canonical_name": str,
    "definitions": [
        {
            "text": str,
            "domain": str
        },
        ...
    ],
    "occurrences": [
        {
            "form": str,
            "position": int,
            "confidence": float
        },
        ...
    ]
  },
  ...
]

No inventes ningún tecnismo.
"""

def match_occurrences(
    occurrences : list,
    positions : list,
    unresolved: list,
):
    occurrence_positions = np.array(
        [occurrence.position for _, occurrence in occurrences]
    )

    text_positions = np.array(positions)

    # Cost matrix: absolute positional distance
    costs = np.abs(
        occurrence_positions[:, None] - text_positions[None, :]
    )

    row_indices, col_indices = linear_sum_assignment(costs)

    for row, col in zip(row_indices, col_indices):
        distance = costs[row, col]

        # Omit non assigned instances
        if not np.isfinite(distance):
            continue

        _, occurrence = occurrences[row]

        occurrence.position = positions[col]

    # Remove all non-matched occurrences
    unresolved.extend([
        occurrence
        for i, (_, occurrence) in enumerate(occurrences)
        if i not in row_indices
    ])


def annotate_technicisms(text: str, technicisms) -> str:
    replacements = []

    for technicism in technicisms:
        for occurrence in technicism.occurrences:
            start = occurrence.position
            end = start + len(occurrence.form)
            value = text[start:end].replace('\n\n','\n')
            occurrence.form = value

            replacements.append((start,end,value))

    # Replace from right to left so positions remain valid.
    for start, end, value in sorted(
        replacements,
        key=lambda x: x[0],
        reverse=True,
    ):
        text = text[:start] + f"[[|{value}|]]" + text[end:]

    return text


class TechnicismExpert:
    def __init__(self, ollama_model):
        self.technicism_agent = Agent(
            ollama_model,
            output_type=NativeOutput(List[Technicism]),
            system_prompt=technicism_expert_prompt,
            model_settings=ModelSettings(temperature=0.3),
        )

    async def get_technicisms(self, technicism_input: TechnicismTextInput) -> List[Technicism]:
        result = await self.technicism_agent.run(
            technicism_input.model_dump_json()
        )

        technicisms = result.output

        # Fix broken positions by finding out the closest appearances of their forms.
        grouped = {}
        unresolved = []

        # Group by technicism forms
        for technicism in technicisms:
            for occurrence in technicism.occurrences:
                if occurrence.form not in grouped:
                    grouped[occurrence.form] = [(technicism, occurrence)]
                else:
                    grouped[occurrence.form].append((technicism, occurrence))

        # Determine if there's any occurrences for each form
        for form, occurrences in grouped.items():
            positions = [match.start() for match in re.finditer(rf"\b{r"[ \n]{1,2}".join(re.escape(x) for x in re.split(r"\s+", form))}\b", technicism_input.text, re.IGNORECASE)]

            if len(positions) != 0:
                positions.sort()
                occurrences.sort(key=lambda x: x[1].position)

                match_occurrences(occurrences, positions, unresolved)
            else:
                unresolved.extend([occurrence for _, occurrence in occurrences])

        # Remove AI hallucinations from occurrences
        for technicism in technicisms:
            technicism.occurrences = [
                occurrence
                for occurrence in technicism.occurrences
                if occurrence not in unresolved
            ]

        # Remove technicisms inside other technicisms
        all_occurrences = []

        for technicism_index, technicism in enumerate(technicisms):
            for occurrence in technicism.occurrences:
                all_occurrences.append((technicism_index, occurrence))

        for technicism_index, technicism in enumerate(technicisms):
            kept = []

            for occurrence in technicism.occurrences:
                end = occurrence.position + len(occurrence.form)

                contained = False

                for other_index, other in all_occurrences:
                    other_end = other.position + len(other.form)

                    # Never compare occurrences within the same technicism
                    if other_index != technicism_index:
                        # Check if this occurrence is inside another occurrence of another technicism
                        if other.position <= occurrence.position and end <= other_end:
                            contained = True
                            break

                if not contained:
                    kept.append(occurrence)

            technicism.occurrences = kept

        return [technicism for technicism in technicisms
                if len(technicism.occurrences) > 0 and len(technicism.definitions) > 0]