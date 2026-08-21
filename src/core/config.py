from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_project_id: str

    google_cloud_vision_api_credentials: str
    google_cloud_vision_api_endpoint: str

    gemini_api_credentials: str
    gemini_api_location: str
    gemini_model: str

    ollama_local_model: str

    model_config = SettingsConfigDict(env_file=".env")

# Reads settings directly from .env file
settings = Settings()

print(Settings().model_dump())