from fastapi import APIRouter
from src.pipeline.train import run_training
from src.pipeline.infer import predict_for_user
from src.types.schemas import PredictionRequest, PredictionResponse, ItemPrediction

router = APIRouter()

@router.get("/healthz")
def health():
    return {"status": "ok"}

@router.post("/train", summary="Engatilha o treinamento (dados direto do DB)")
def train_model():
    result = run_training()
    return result

@router.post("/predict", response_model=PredictionResponse, summary="Prediz prob. de esgotado/vencido para um usuário")
def predict(req: PredictionRequest):
    df = predict_for_user(req.user_id)
    if df.empty:
        return PredictionResponse(user_id=req.user_id, items=[])

    items = [
        ItemPrediction(
            product_id=str(r.product_id),
            probability_out_or_expired=f"{float(r.probability):.6f}",
            last_notification_at=str(r.last_notification_at),
            days_since_purchase=f"{float(r.days_since_purchase):.6f}"
        )
        for r in df.itertuples(index=False)
    ]
    return PredictionResponse(user_id=req.user_id, items=items)


