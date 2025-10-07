# src/pipeline/infer.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.db.session import get_engine
from src.db.queries import fetch_item_transactions, fetch_pantry_validities
from src.features.cycles import build_cycles
from src.features.consumption import estimate_user_product_consumption
from src.features.dataset import build_inference_rows
from src.model.io import load_model
from src.model.estimator import predict_proba_positive

PURCHASE_STATES = {"PURCHASED", "IN_PANTRY"}
TERMINAL_STATES = {"OUT", "EXPIRED"}  # ciclos encerrados pelo feedback

# ---------- helpers de normalização ----------

def _to_utc(ts_series: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_series, errors="coerce", utc=True)

def _normalize_items_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "id","user_id","product_id","transaction_code","state",
            "quantity","valid_for_days","created_at"
        ])
    out = df.copy()

    # ids e transaction_code como string "limpa"
    for col in ("user_id","product_id","transaction_code"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    # portion -> quantity
    if "quantity" not in out.columns and "portion" in out.columns:
        out = out.rename(columns={"portion": "quantity"})

    # tipos numéricos
    out["quantity"] = pd.to_numeric(out.get("quantity", 0.0), errors="coerce").fillna(0.0)
    if "valid_for_days" in out.columns:
        out["valid_for_days"] = pd.to_numeric(out["valid_for_days"], errors="coerce")
        out.loc[out["valid_for_days"] == 0, "valid_for_days"] = np.nan
    else:
        out["valid_for_days"] = np.nan

    # datas UTC-aware
    if "created_at" in out.columns:
        out["created_at"] = _to_utc(out["created_at"])

    return out

def _normalize_pantry_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["user_id","product_id","valid_for_days"])
    out = df.copy()
    for col in ("user_id","product_id"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    # numeric + tratar zeros como NaN
    out["valid_for_days"] = pd.to_numeric(out.get("valid_for_days", np.nan), errors="coerce")
    out.loc[out["valid_for_days"] == 0, "valid_for_days"] = np.nan
    return out

# ---------- fallback quando não há ciclos ativos ----------

def _fallback_latest_purchases_X(
    now_ts: pd.Timestamp,
    user_items: pd.DataFrame,
    pantry_user: pd.DataFrame,
    cons_df: pd.DataFrame
):
    """
    Constrói features a partir da ÚLTIMA compra (PURCHASED/IN_PANTRY) por produto,
    desde que o produto NÃO tenha sido encerrado (último estado != OUT/EXPIRED).
    """
    if user_items.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = user_items.copy()

    # Ignorar produtos cujo ÚLTIMO estado do fluxo foi terminal (OUT/EXPIRED)
    last_per_prod = (df.sort_values("created_at")
                       .groupby(["user_id","product_id"])
                       .tail(1)
                       .reset_index(drop=True))
    closed = set(
        tuple(r) for r in
        last_per_prod[last_per_prod["state"].isin(TERMINAL_STATES)][["user_id","product_id"]].itertuples(index=False, name=None)
    )
    if closed:
        mask_keep = ~df.set_index(["user_id","product_id"]).index.isin(closed)
        df = df[mask_keep].reset_index(drop=True)

    # Último evento de compra (PURCHASED/IN_PANTRY) por produto
    df_buy = df[df["state"].isin(PURCHASE_STATES)]
    if df_buy.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest = (df_buy.sort_values("created_at")
                  .groupby(["user_id","product_id"])
                  .tail(1)
                  .reset_index(drop=True))

    # juntar validade do pantry (NaN + flag)
    if pantry_user is not None and not pantry_user.empty:
        latest = latest.merge(
            pantry_user[["user_id","product_id","valid_for_days"]],
            on=["user_id","product_id"],
            how="left",
            suffixes=("", "_pantry")
        )
        # se o evento tiver valid_for_days NaN, usa o do pantry
        latest["valid_for_days"] = latest["valid_for_days"].where(
            latest["valid_for_days"].notna(), latest["valid_for_days_pantry"]
        )
        latest.drop(columns=[c for c in latest.columns if c.endswith("_pantry")], inplace=True)

    # flag de validade conhecida
    latest["has_validity_info"] = latest["valid_for_days"].notna().astype(float)

    # consumo estimado
    cons_key = cons_df.set_index(["user_id","product_id"]) if (cons_df is not None and not cons_df.empty) else None
    latest["consumption_uday"] = 1.0
    if cons_key is not None:
        idx = latest.set_index(["user_id","product_id"]).index
        match = cons_key.reindex(idx)["consumption_uday"]
        latest.loc[match.notna().values, "consumption_uday"] = match.dropna().values.astype(float)

    # dias desde a compra
    latest["purchase_time"] = latest["created_at"]
    latest["days_since_purchase"] = (
        (now_ts - latest["purchase_time"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    # proteções numéricas
    latest["quantity_bought"] = pd.to_numeric(latest["quantity"], errors="coerce").fillna(0.0).astype(float)
    latest["consumption_uday"] = pd.to_numeric(latest["consumption_uday"], errors="coerce").fillna(1.0).clip(lower=1e-3)

    latest["expected_depletion_days"] = (latest["quantity_bought"] / latest["consumption_uday"]).clip(lower=0.5)

    latest["ratio_to_validity"] = latest["days_since_purchase"] / latest["valid_for_days"].replace(0, np.nan)
    latest["ratio_to_depletion"] = latest["days_since_purchase"] / latest["expected_depletion_days"].replace(0, np.nan)

    feature_cols = [
        "quantity_bought","valid_for_days","has_validity_info",
        "consumption_uday","expected_depletion_days",
        "ratio_to_validity","ratio_to_depletion"
    ]
    X = latest[feature_cols].fillna(0.0)
    meta = latest[["user_id","product_id","days_since_purchase"]].copy()
    return X, meta

# ---------- função pública ----------

def predict_for_user(user_id: str) -> pd.DataFrame:
    engine = get_engine()
    items = fetch_item_transactions(engine)
    pantry = fetch_pantry_validities(engine)

    # normalizar
    items = _normalize_items_df(items)
    pantry = _normalize_pantry_df(pantry)

    user_id_s = str(user_id).strip()
    items["user_id"] = items["user_id"].astype(str).str.strip()
    pantry["user_id"] = pantry["user_id"].astype(str).str.strip()
    items["product_id"] = items["product_id"].astype(str).str.strip()
    pantry["product_id"] = pantry["product_id"].astype(str).str.strip()

    user_items = items[items["user_id"] == user_id_s].copy()
    pantry_user = pantry[pantry["user_id"] == user_id_s].copy()

    if user_items.empty and pantry_user.empty:
        return pd.DataFrame(columns=["product_id","probability"])

    # construir ciclos com tudo que temos do usuário
    cycles = build_cycles(user_items, pantry_user)
    cons = estimate_user_product_consumption(cycles)

    now_ts = pd.Timestamp.now(tz="UTC")

    # usar somente ciclos ATIVOS (sem terminal OUT/EXPIRED)
    cycles_active = cycles.copy()
    if "terminal_time" in cycles_active.columns:
        cycles_active = cycles_active[cycles_active["terminal_time"].isna()].reset_index(drop=True)

    if not cycles_active.empty:
        X, meta = build_inference_rows(now_ts, cycles_active, cons)
    else:
        # fallback: última compra válida (produto cujo último estado NÃO é OUT/EXPIRED)
        X, meta = _fallback_latest_purchases_X(now_ts, user_items, pantry_user, cons)

    if X is None or X.empty or meta is None or meta.empty:
        return pd.DataFrame(columns=["product_id","probability", "last_notification_at", "days_since_purchase"])

    model = load_model()
    proba = predict_proba_positive(model, X)

    out = meta[["product_id"]].copy()
    out["probability"] = proba.astype(float)
    out["last_notification_at"] = meta["last_notification_at"] if "last_notification_at" in meta.columns else pd.NaT
    out["days_since_purchase"] = meta["days_since_purchase"].astype(float) if "days_since_purchase" in meta.columns else np.nan
    return out[["product_id","probability", "last_notification_at", "days_since_purchase"]]
