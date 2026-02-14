from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = Field(..., description="Postgres URL")
    MODEL_DIR: str = Field(default="models")
    MODEL_NAME: str = Field(default="ml_expires.joblib")
    AWS_BUCKET_NAME: str = Field(default="", description="S3 bucket for model storage (empty = local filesystem)")

    class Config:
        env_file = ".env"

settings = Settings()
