from enum import Enum
from typing import List

import json
from pydantic_ai import Agent, NativeOutput, ModelSettings
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.core.agents.title_classifier import TitleClassifier
from src.core.config import settings

ollama_model = OllamaModel(
    settings.ollama_local_model,
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

class AgentType(Enum):
    CLASSIFICATION = "classification"
    TECHNICISM = "technicism"
    SIMPLIFICATION = "simplification"

class OllamaService:
    def __init__(self):

        self.agent_fleet = {
            AgentType.CLASSIFICATION: TitleClassifier(ollama_model),
        }