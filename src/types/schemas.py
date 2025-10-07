from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    user_id: str

class ItemPrediction(BaseModel):
    product_id: str
    probability_out_or_expired: float
    days_since_purchase: float = None
    last_notification_at: str = None

class PredictionResponse(BaseModel):
    user_id: str
    items: List[ItemPrediction]
