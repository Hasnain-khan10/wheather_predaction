from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    aqicn_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    hopsworks_api_key: Optional[str] = None
    hopsworks_project_name: Optional[str] = None
    hopsworks_host: str = "eu-west.cloud.hopsworks.ai"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
