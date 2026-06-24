from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_cloud_vision_api_credentials: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

print(Settings().model_dump())