import os
import joblib
from src.core.config import settings

def model_path() -> str:
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    return os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)

def save_model(model) -> str:
    path = model_path()
    joblib.dump(model, path)
    return path

def load_model():
    path = model_path()
    return joblib.load(path)
