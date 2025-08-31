from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = Field(..., description="Postgres URL")
    MODEL_DIR: str = Field(default="models")
    MODEL_NAME: str = Field(default="ml2_expiry.joblib")

    class Config:
        env_file = ".env"

settings = Settings()
