from datetime import datetime, timezone
from src.db.session import get_engine
from src.db.queries import fetch_item_transactions, fetch_pantry_validities
from src.features.cycles import build_cycles
from src.features.consumption import estimate_user_product_consumption
from src.features.dataset import build_training_dataset
from src.model.estimator import build_classifier, build_baseline
from src.model.io import save_model
import numpy as np


def run_training() -> dict:
    engine = get_engine()
    items = fetch_item_transactions(engine)
    pantry = fetch_pantry_validities(engine)

    cycles = build_cycles(items, pantry)
    cons = estimate_user_product_consumption(cycles)
    train_df = build_training_dataset(cycles, cons)

    if train_df.empty:
        return {
            "status": "no-data",
            "message": "Sem dados suficientes para treinar (dataset vazio após feature engineering).",
            "hints": [
                "Amplie a janela temporal das consultas (ex.: 365→720 dias).",
                "Verifique se há registros em item_transaction e pantry_items.",
            ],
        }

    feature_cols = [
        "quantity_bought", "valid_for_days", "has_validity_info",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion",
    ]
    X = train_df[feature_cols]
    y = train_df["y"].astype(int)

    if X.empty or y.empty:
        return {
            "status": "no-data",
            "message": "Sem dados de features ou labels (X ou y vazios).",
            "debug": {"rows": int(len(train_df))},
        }
    classes, counts = np.unique(y, return_counts=True)
    class_distribution = {int(c): int(n) for c, n in zip(classes, counts)}

    if len(classes) < 2:
        only_class = int(classes[0])
        model = build_baseline(constant=only_class)
        # fit para manter interface (classes_, etc.)
        model.fit(X, y)
        path = save_model(model)
        return {
            "status": "ok-baseline",
            "reason": "single-class",
            "message": (
                "O conjunto de treino contém apenas UMA classe em y "
                f"(classe {only_class}). Foi treinado um modelo baseline que "
                "prediz sempre a classe observada. Quando houver dados com as 2 classes, "
                "o próximo treino trocará automaticamente para o classificador principal."
            ),
            "class_distribution": class_distribution,
            "trained_on_rows": int(len(train_df)),
            "model_path": path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hints": [
                "Amplie o intervalo temporal na query para capturar mais ciclos.",
                "Considere abrandar o limiar do rótulo (ex.: 0.9*min(valid_for_days, expected_depletion_days)).",
                "Inclua ciclos ainda não reabastecidos usando days_since_purchase (censura à direita).",
                "Verifique se valid_for_days muito alto (fallback 999) não está mascarando rótulos.",
            ],
        }

        # 5) Treino normal (duas classes)
    model = build_classifier()
    model.fit(X, y)
    path = save_model(model)

    return {
        "status": "ok",
        "message": "Modelo treinado com sucesso.",
        "class_distribution": class_distribution,
        "trained_on_rows": int(len(train_df)),
        "model_path": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }