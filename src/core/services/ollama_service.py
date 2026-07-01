from typing import List

import json
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.core.config import settings
from src.model.information.information_model import InformationLine, EnrichedInformationLine, \
    InformationLabel, TitleCandidateLine

# Prompt for the overall classification of paragraphs
classif_prompt = """
Eres un sistema clasificador estricto sobre placas descriptivas de museos.

Recibirás N líneas de texto con esta estructura:
{
    "block_id": int,
    "id": int,
    "text": str,
    "geometry": {
        "x_ratio": float,
        "y_ratio": float,
        "width_ratio": float,
        "height_ratio": float
}

Tu tarea es asignar a cada línea de texto UNA categoría:

"title"
"title_translation"
"artist"
"artist_dates"
"date"
"medium"
"date_and_medium"
"collection"
"inventory"
"description"
"description_translation"
"other"

La geometría de cada línea de texto es una ayuda. No determina a ciencia cierta la categoría a aplicar.

Reglas:

1. Devuelve EXACTAMENTE N resultados, uno por cada id.
2. Analiza todas las líneas de texto como parte del mismo contexto.
3. Analiza todos los bloques como parte de un mismo sub-contexto.
4. No inventes ni omitas líneas de texto.
5. Los "title" son una única frase coherente.
6. Si una línea de texto tiene más de una oración es "description" o "description_translation".
7. Si la líneas de texto contiene únicamente nombres completos es "artist".
8. Las fechas, años o similar por si solas NO son "title".
9. Las fechas, años o similar acompañadas de lugares sin paréntesis NO son "title".
10. Los materiales o técnicas NO son "title".
11. Los números de inventario NO son "title".
12. Si dos o más líneas de texto significan lo mismo en distinto idioma → uno es de una categoría, y el resto de
    la traducción de esa categoría.
13. Si dos o más líneas de texto pueden ser "title" y no puedes saber cuál es el original → clasifica ambos como "title".
14. Si no estás seguro → usa "other".

Responde SOLO con un JSON válido:
[
  {"id": int, "label": str}
]

No inventes ningún tipo de información.


A continuación se proporciona un ejemplo de la tarea:
Input: [
    {"block_id": 0, "id": 0, "text": "María Izquierdo", "geometry": {"height_ratio": 0.0438,
     "width_ratio": 0.457, "x_ratio": 0.1066, "y_ratio": 0.0744},
    {"block_id": 0, "id": 1, "text": "1902, San Juan de los Lagos - 1955, Ciudad de México", 
    "geometry": {"height_ratio": 0.0352, "width_ratio": 0.5556, "x_ratio": 0.1087, "y_ratio": 0.1492},
    {"block_id": 1, "id": 2, "text": "El idilio", "geometry": {"height_ratio": 0.0254,
     "width_ratio": 0.3366, "x_ratio": 0.1164, "y_ratio": 0.2191},
    {"block_id": 1, "id": 3, "text": "The idyll", "geometry": {"height_ratio": 0.0254,
     "width_ratio": 0.2964, "x_ratio": 0.1185, "y_ratio": 0.2427},
    {"block_id": 1, "id": 4, "text": "1946, Óleo sobre lienzo", "geometry": {"height_ratio": 0.0214,
     "width_ratio": 0.2579, "x_ratio": 0.1393, "y_ratio": 0.4004},
    {"block_id": 2, "id": 5, "text": "Obj. 02819", "geometry": {"height_ratio": 0.0104,
     "width_ratio": 0.0805, "x_ratio": 0.7173, "y_ratio": 0.5662}
]

Output: [
    {"id": 0, "label": "artist"},
    {"id": 1, "label": "artist_dates"},
    {"id": 2, "label": "title"},
    {"id": 3, "label": "title_translation"},
    {"id": 4, "label": "date_and_medium"},
    {"id": 5, "label": "inventory"}
]
"""

# Prompt to break the tie for several titles
tiebreaking_prompt = """
Eres un agente encargado de determinar un título original entre múltiples candidatos.

Recibirás una lista de N posibles títulos con el siguiente formato:
{
    "block_id": int,
    "id": int,
    "text": str,
    "geometry": {
        "x_ratio": float,
        "y_ratio": float,
        "width_ratio": float,
        "height_ratio": float
}

Tu tarea es asignarles una confianza a estos títulos.
La confianza es un número real entre 0.0 y 1.0, ambos incluidos, siendo 0 no es un título y 1 es un título.

Reglas:

1. Devuelve EXACTAMENTE N resultados, uno por cada id.
3. No inventes ni omitas líneas de texto.
4. Un título es una única frase coherente.
5. Un título no tiene más de una oración
6. Si un candidato sólo contiene nombres completos, NO es un título.
7. Si un candidato sólo contiene fechas, años o similar NO es un título.
8. Los materiales o técnicas NO son un título.
9. Los números de inventario NO son un título.
10. Si dos o más líneas de texto significan lo mismo en distinto idioma → uno es el título de verdad, 
    el resto son traducciones.

Responde SOLO con un JSON válido:
[
  {"id": int, "confidence": float}
]

No inventes ningún tipo de información.
"""

ollama_model = OllamaModel(
    settings.ollama_local_model,
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

class OllamaService:
    def __init__(self):
        self.classification_agent = Agent(
            ollama_model,
            output_type=NativeOutput(List[InformationLine]),
            system_prompt=classif_prompt,
        )

        self.tiebreaker_agent = Agent(
            ollama_model,
            output_type=NativeOutput(List[TitleCandidateLine]),
            system_prompt=tiebreaking_prompt
        )


    async def tiebreak_titles(self, titles, lines):
        """Order title candidates according to their likelihood to be the real title.

        Run an agent to order title candidates according to their likelihood to be the real title of a text
        from a given list of title candidates, all from the same text.

            :param titles: A list of InformationLine objects that are candidates to be a title.
            :type titles: List[InformationLine]

            :param lines: A list of text lines to classify of the form
                [
                    {
                        "block_id":int
                        "id":int
                        "text":str,
                        "geometry":{
                            "x_ratio":float,
                            "y_ratio":float,
                            "width_ratio":float,
                            "height_ratio":float
                        }
                    },
                    ...
                ]
            :type lines: list[dict]

            :returns confidence_results: A list of TitleCandidateLine objects.
            :rtype: list[TitleCandidateLine]
        """
        title_candidates = [json.dumps({
            "block_id": lines[candidate.id]["block_id"],
            "id": candidate.id,
            "text": lines[candidate.id]["text"],
            "geometry": lines[candidate.id]["geometry"]
        }) for candidate in titles]

        confidence_results : List[TitleCandidateLine] = (
            (await self.tiebreaker_agent.run(title_candidates)).output)

        confidence_results.sort(key=lambda x: x.confidence, reverse=True)

        return confidence_results


    async def classify_lines(self, lines):
        """Classify text lines.

        Run an agent to classify text lines into the different categories represented by InformationLabel

            :param lines: A list of text lines to classify of the form:
                [
                    {
                        "block_id":int
                        "text":str,
                        "geometry":{
                            "x_ratio":float,
                            "y_ratio":float,
                            "width_ratio":float,
                            "height_ratio":float
                        }
                    },
                    ...
                ]
            :type lines: list[Dict]

            :returns title: The best title candidate found from the text lines given, or "Desconocido"
                classified_lines: A list of EnrichedInformationLine objects with the lines retrieved \
                after classification.
            :rtype: tuple[str, list[EnrichedInformationLine]]
        """
        indexed_lines = [json.dumps({"id": i, "text": line["text"],
                                          "geometry": line["geometry"]}) for i, line in enumerate(lines)]
        lookup = {i: p["text"] for i, p in enumerate(lines)}

        processed_lines: List[InformationLine] = (
            await self.classification_agent.run(indexed_lines)).output

        titles = [p for p in processed_lines if p.label == InformationLabel.TITLE]

        if len(titles) > 1:
            confidences = await self.tiebreak_titles(titles, lines)
            confidence_map = {c.id: c.confidence for c in confidences}

            titles.sort(key=lambda x: confidence_map[x.id]
                                if x.id and x.id in confidence_map else -1
                        , reverse=True)

        elif len(titles) == 0:
            title_translations = [p for p in processed_lines if p.label == InformationLabel.TITLE_TRANSLATION]

            if len(title_translations) == 1:
                titles = title_translations

            elif len(title_translations) > 1:
                confidences = await self.tiebreak_titles(title_translations, lines)
                confidence_map = {c.id: c.confidence for c in confidences}

                title_translations.sort(key=lambda x: confidence_map[x.id]
                if x.id and x.id in confidence_map else -1
                            , reverse=True)
                titles = title_translations

            elif len(title_translations) == 0:
                titles = [InformationLine(id=-1, label=InformationLabel.TITLE)]

        processed_lines.sort(key=lambda x: x.id)

        classified_lines : List[EnrichedInformationLine] = [
            EnrichedInformationLine(
                text=lookup[line_info.id] if line_info.id is not None and
                                             len(lookup) > line_info.id > -1
                else "Desconocido",
                label=line_info.label,
            )
            for line_info in processed_lines
        ]

        title = lookup[titles[0].id] if titles[0].id is not None and len(lookup) > titles[0].id > -1 else "Desconocido"
        return title, classified_lines