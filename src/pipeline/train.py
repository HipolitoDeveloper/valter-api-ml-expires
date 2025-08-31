from datetime import datetime, timezone
from src.db.session import get_engine
from src.db.queries import fetch_item_transactions, fetch_pantry_validities
from src.features.cycles import build_cycles
from src.features.consumption import estimate_user_product_consumption
from src.features.dataset import build_training_dataset
from src.model.estimator import build_classifier
from src.model.io import save_model

def run_training() -> dict:
    engine = get_engine()
    items = fetch_item_transactions(engine)
    pantry = fetch_pantry_validities(engine)

    cycles = build_cycles(items, pantry)
    cons = estimate_user_product_consumption(cycles)
    train_df = build_training_dataset(cycles, cons)

    if train_df.empty:
        return {"status": "no-data", "message": "Sem dados suficientes para treinar."}

    X = train_df[[
        "quantity_bought", "valid_for_days",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]]
    y = train_df["y"].astype(int)

    model = build_classifier()
    model.fit(X, y)

    path = save_model(model)
    return {
        "status": "ok",
        "trained_on_rows": int(len(train_df)),
        "model_path": path,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
