from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.db.session import get_engine
from src.db.queries import (
    fetch_item_transactions,
    fetch_pantry_validities,
    fetch_expiry_feedback,
)
from src.features.cycles import build_cycles
from src.features.consumption import estimate_user_product_consumption
from src.features.dataset import (
    build_training_dataset,
    build_feedback_supervised_dataset,
)
from src.model.estimator import build_classifier, build_baseline
from src.model.io import save_model

FEATURE_COLS = [
    "quantity_bought", "valid_for_days", "has_validity_info",
    "consumption_uday", "expected_depletion_days",
    "ratio_to_validity", "ratio_to_depletion",
]

def run_training() -> dict:
    engine = get_engine()
    items = fetch_item_transactions(engine)
    pantry = fetch_pantry_validities(engine)
    fb    = fetch_expiry_feedback(engine)    # ← feedback humano

    # ---------- Conjunto A: rótulo heurístico (ciclos) ----------
    cycles_all = build_cycles(items, pantry)
    cons_all   = estimate_user_product_consumption(cycles_all)
    ds_cycles  = build_training_dataset(cycles_all, cons_all)

    # ---------- Conjunto B: rótulo humano (feedback de notificações) ----------
    ds_fb = build_feedback_supervised_dataset(fb, items, pantry)

    # Se ambos vazios, não há o que treinar
    if (ds_cycles is None or ds_cycles.empty) and (ds_fb is None or ds_fb.empty):
        return {
            "status": "no-data",
            "message": "Sem dados suficientes: nem ciclos rotulados nem feedback humano.",
        }

    # Concatena datasets; feedback recebe mais peso (se existir)
    parts = []
    sample_weight = None
    if ds_cycles is not None and not ds_cycles.empty:
        tmp = ds_cycles[["y"] + FEATURE_COLS].copy()
        tmp["__w__"] = 1.0
        parts.append(tmp)
    if ds_fb is not None and not ds_fb.empty:
        ds_fb_use = ds_fb[["y"] + FEATURE_COLS].copy()
        ds_fb_use["__w__"] = 2.0  # peso maior para feedback humano
        parts.append(ds_fb_use)

    train_df = pd.concat(parts, ignore_index=True) if len(parts) > 0 else pd.DataFrame()
    if train_df.empty:
        return {
            "status": "no-data",
            "message": "Datasets construídos ficaram vazios.",
        }

    # Extrai weights se presentes
    if "__w__" in train_df.columns:
        sample_weight = train_df["__w__"].values
        train_df = train_df.drop(columns=["__w__"])

    X = train_df[FEATURE_COLS].fillna(0.0)
    y = train_df["y"].astype(int)

    # Diagnóstico de classes
    classes, counts = np.unique(y, return_counts=True)
    class_distribution = {int(c): int(n) for c, n in zip(classes, counts)}

    if len(classes) < 2:
        # Classe única → baseline
        only = int(classes[0])
        model = build_baseline(constant=only)  # DummyClassifier
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)
        path = save_model(model)
        return {
            "status": "ok-baseline",
            "reason": "single-class",
            "class_distribution": class_distribution,
            "trained_on_rows": int(len(train_df)),
            "feedback_rows": 0 if ds_fb is None else int(len(ds_fb)),
            "heuristic_rows": 0 if ds_cycles is None else int(len(ds_cycles)),
            "model_path": path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hints": [
                "Colete mais feedbacks dos usuários (is_out/is_expired).",
                "Amplie a janela de histórico e diversifique produtos.",
            ],
        }

    # Treino normal (duas classes)
    model = build_classifier()

    fit_kwargs = {}
    if sample_weight is not None:
        if hasattr(model, "named_steps") and "logreg" in getattr(model, "named_steps", {}):
            fit_kwargs["logreg__sample_weight"] = sample_weight
        else:
            fit_kwargs["sample_weight"] = sample_weight

    model.fit(X, y, **fit_kwargs)
    path = save_model(model)

    return {
        "status": "ok",
        "message": "Modelo treinado com sucesso (ciclos + feedback).",
        "class_distribution": class_distribution,
        "trained_on_rows": int(len(train_df)),
        "feedback_rows": 0 if ds_fb is None else int(len(ds_fb)),
        "heuristic_rows": 0 if ds_cycles is None else int(len(ds_cycles)),
        "model_path": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
