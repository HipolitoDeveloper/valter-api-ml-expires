from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    user_id: str

class ItemPrediction(BaseModel):
    product_id: str
    probability_out_or_expired: float

class PredictionResponse(BaseModel):
    user_id: str
    items: List[ItemPrediction]
