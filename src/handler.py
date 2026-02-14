"""AWS Lambda handler for the Valter ML Expiry/Out-of-Stock service."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_INITIALIZED = False


def _init():
    global _INITIALIZED
    if _INITIALIZED:
        return
    from src.model.io import load_model
    try:
        load_model()
    except Exception:
        logger.warning("Model not loaded during init — will retry at predict time")
    _INITIALIZED = True


def handle_predict(event, context):
    """API Gateway HTTP API v2 — POST /predict"""
    _init()

    try:
        body = json.loads(event.get("body") or "{}")
        user_id = body.get("user_id")

        if not user_id:
            return _response(400, {"error": "user_id is required"})

        from src.pipeline.infer import predict_for_user
        df = predict_for_user(user_id)

        if df.empty:
            return _response(200, {"user_id": user_id, "items": []})

        items = [
            {
                "product_id": str(r.product_id),
                "probability_out_or_expired": round(float(r.probability), 6),
                "days_since_purchase": round(float(r.days_since_purchase), 6),
                "last_notification_at": str(r.last_notification_at) if hasattr(r, "last_notification_at") else None,
            }
            for r in df.itertuples(index=False)
        ]

        return _response(200, {"user_id": user_id, "items": items})

    except Exception as e:
        logger.exception("Error in handle_predict")
        return _response(500, {"error": str(e)})


def handle_train(event, context):
    """EventBridge CRON — weekly model retraining."""
    try:
        from src.pipeline.train import run_training
        result = run_training()
        logger.info("Training completed: %s", json.dumps(result, default=str))
        return result
    except Exception as e:
        logger.exception("Error in handle_train")
        return {"status": "error", "error": str(e)}


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
