from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.core.config import settings

def get_engine() -> Engine:
    return create_engine(settings.DATABASE_URL, pool_pre_ping=True)
