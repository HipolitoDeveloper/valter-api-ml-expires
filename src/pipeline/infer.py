import pandas as pd
from datetime import datetime, timezone
from src.db.session import get_engine
from src.db.queries import fetch_item_transactions, fetch_pantry_validities
from src.features.cycles import build_cycles
from src.features.consumption import estimate_user_product_consumption
from src.features.dataset import build_inference_rows
from src.model.io import load_model
from src.model.estimator import predict_proba_positive as predict_proba

def predict_for_user(user_id: str) -> pd.DataFrame:
    engine = get_engine()
    items = fetch_item_transactions(engine)
    pantry = fetch_pantry_validities(engine)

    user_id_s = str(user_id).strip()
    items["user_id"] = items["user_id"].astype(str).str.strip()
    # pantry["user_id"] = pantry["user_id"].astype(str).str.strip()
    # items["product_id"] = items["product_id"].astype(str).str.strip()
    # pantry["product_id"] = pantry["product_id"].astype(str).str.strip()


    user_items = items[items["user_id"] == user_id]
    if user_items.empty:
        return pd.DataFrame(columns=["product_id","probability"])

    cycles = build_cycles(user_items, pantry[pantry["user_id"] == user_id_s])
    cons = estimate_user_product_consumption(cycles)

    now_ts = pd.to_datetime(datetime.now(timezone.utc))
    X, meta = build_inference_rows(now_ts, cycles, cons)
    if X.empty:
        return pd.DataFrame(columns=["product_id","probability"])

    model = load_model()
    proba = predict_proba(model, X)
    out = meta.copy()
    out["probability"] = proba
    return out[["product_id","probability"]]
