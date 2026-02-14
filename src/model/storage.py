import os
import tempfile
import joblib
from src.config import settings

_S3_CLIENT = None


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _s3_key() -> str:
    return f"models/{settings.MODEL_NAME}"


def _local_path() -> str:
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    return os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)


def _use_s3() -> bool:
    return bool(settings.AWS_BUCKET_NAME)


def save_model(model) -> str:
    if _use_s3():
        local = os.path.join(tempfile.gettempdir(), settings.MODEL_NAME)
        joblib.dump(model, local)
        s3 = _get_s3_client()
        key = _s3_key()
        s3.upload_file(local, settings.AWS_BUCKET_NAME, key)
        return f"s3://{settings.AWS_BUCKET_NAME}/{key}"

    path = _local_path()
    joblib.dump(model, path)
    return path


def load_model():
    if _use_s3():
        local = os.path.join(tempfile.gettempdir(), settings.MODEL_NAME)
        if not os.path.exists(local):
            s3 = _get_s3_client()
            s3.download_file(settings.AWS_BUCKET_NAME, _s3_key(), local)
        return joblib.load(local)

    return joblib.load(_local_path())
