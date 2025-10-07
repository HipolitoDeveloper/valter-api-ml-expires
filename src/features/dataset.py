import pandas as pd
import numpy as np

from src.features.consumption import estimate_user_product_consumption
from src.features.cycles import build_cycles


def build_inference_rows(now_ts, latest_cycles: pd.DataFrame, cons_df: pd.DataFrame):
    """
    Usa o ciclo mais recente por (user, product) para montar as features de inferência.
    Mantém:
      - valid_for_days possivelmente NaN
      - has_validity_info em {0.0, 1.0}
    Retorna:
      X: DataFrame com as features
      meta: DataFrame com colunas auxiliares (user_id, product_id, days_since_purchase opcional)
    """
    if latest_cycles.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest = (latest_cycles.sort_values("purchase_time")
              .groupby(["user_id", "product_id"])
              .tail(1)
              .reset_index(drop=True))

    # garantir UTC-aware nas datas
    if "purchase_time" in latest.columns:
        latest["purchase_time"] = pd.to_datetime(latest["purchase_time"], errors="coerce", utc=True)

    # now_ts em UTC-aware
    if not hasattr(now_ts, "tzinfo") or now_ts.tzinfo is None:
        now_ts = pd.Timestamp(now_ts, tz="UTC")
    else:
        now_ts = pd.Timestamp(now_ts).tz_convert("UTC")

    latest["days_since_purchase"] = (
        (now_ts - latest["purchase_time"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    # consumo mediano histórico por (user, product) se existir
    cons_key = cons_df.set_index(["user_id", "product_id"]) if not cons_df.empty else None
    latest["consumption_uday"] = 1.0
    if cons_key is not None:
        # join mais eficiente que iterar linha a linha
        idx = latest.set_index(["user_id", "product_id"]).index
        cons_match = cons_key.reindex(idx)["consumption_uday"]
        latest.loc[cons_match.notna().values, "consumption_uday"] = cons_match.dropna().values.astype(float)

    # proteções numéricas
    latest["quantity_bought"] = latest["quantity_bought"].astype(float).fillna(0.0)
    latest["consumption_uday"] = latest["consumption_uday"].astype(float).clip(lower=1e-3)

    latest["expected_depletion_days"] = (latest["quantity_bought"] / latest["consumption_uday"]).clip(lower=0.5)

    # flag de validade conhecida
    if "has_validity_info" not in latest.columns:
        latest["has_validity_info"] = (~latest["valid_for_days"].isna()).astype(float)
    else:
        latest["has_validity_info"] = latest["has_validity_info"].astype(float).fillna(0.0)

    # ratios (ficam NaN se divisor NaN/0; preenchidos no fillna de X)
    latest["ratio_to_validity"] = latest["days_since_purchase"] / latest["valid_for_days"].replace(0, np.nan)
    latest["ratio_to_depletion"] = latest["days_since_purchase"] / latest["expected_depletion_days"].replace(0, np.nan)

    # features alinhadas com o treino
    X = latest[[
        "quantity_bought", "valid_for_days", "has_validity_info",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)

    # meta útil para retorno/instrumentação (inclui days_since_purchase se quiser usar no Node)
    meta = latest[["user_id", "product_id", "days_since_purchase", 'last_notification_at']].copy()

    return X, meta

def _normalize_ids(df: pd.DataFrame, cols=("user_id","product_id")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out

def build_training_dataset(cycles_df: pd.DataFrame, cons_df: pd.DataFrame) -> pd.DataFrame:
    df = cycles_df.copy()
    df = df.dropna(subset=["next_in_cart_time", "days_to_restock"])
    if df.empty:
        return pd.DataFrame(columns=[
            "user_id","product_id","days_to_restock",
            "quantity_bought","valid_for_days","has_validity_info",
            "consumption_uday","expected_depletion_days",
            "ratio_to_validity","ratio_to_depletion","y"
        ])

    cons_key = cons_df.set_index(["user_id","product_id"]) if not cons_df.empty else None
    consumption = []
    for _, row in df.iterrows():
        if cons_key is not None and (row["user_id"], row["product_id"]) in cons_key.index:
            consumption.append(cons_key.loc[(row["user_id"], row["product_id"]), "consumption_uday"])
        else:
            consumption.append(1.0)
    df["consumption_uday"] = np.array(consumption, dtype=float)

    df["expected_depletion_days"] = (df["quantity_bought"] / df["consumption_uday"].clip(lower=1e-3)).clip(lower=0.5)
    df["ratio_to_validity"] = df["days_to_restock"] / df["valid_for_days"].replace(0, np.nan)
    df["ratio_to_depletion"] = df["days_to_restock"] / df["expected_depletion_days"].replace(0, np.nan)

    df["y"] = (df["days_to_restock"] >= np.minimum(df["valid_for_days"], df["expected_depletion_days"])).astype(int)

    features = df[[
        "quantity_bought", "valid_for_days", "has_validity_info",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)

    return pd.concat([df[["user_id","product_id","days_to_restock"]], features, df[["y"]]], axis=1)

def build_feedback_supervised_dataset(feedback_df: pd.DataFrame,
                                      items_df: pd.DataFrame,
                                      pantry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói dataset supervisionado a partir do feedback humano:
      - Para cada (user_id, product_id, notified_at):
        * Usa histórico ATÉ notified_at para construir ciclos e consumo
        * Usa o ciclo MAIS recente anterior a notified_at
        * Calcula features com days_since_purchase = notified_at - purchase_time
      - y_feedback = 1 se is_out OR is_expired; 0 caso contrário
    """
    if feedback_df is None or feedback_df.empty:
        return pd.DataFrame()

    fb = feedback_df.copy()
    fb = _normalize_ids(fb)
    fb["notified_at"] = pd.to_datetime(fb["notified_at"], errors="coerce", utc=True)

    items = _normalize_ids(items_df)
    items["created_at"] = pd.to_datetime(items["created_at"], errors="coerce", utc=True)
    pantry = _normalize_ids(pantry_df)

    rows = []
    # processa por usuário para eficiência
    for (u), fb_user in fb.groupby("user_id"):
        items_u = items[items["user_id"] == u]
        pantry_u = pantry[pantry["user_id"] == u]
        if items_u.empty:
            continue

        # para cada notificação
        for r in fb_user.itertuples(index=False):
            p = getattr(r, "product_id")
            t_cut = getattr(r, "notified_at")

            # histórico do produto até o corte
            hist = items_u[(items_u["product_id"] == p) & (items_u["created_at"] <= t_cut)]
            if hist.empty:
                continue

            # constroi ciclos nesse histórico
            cycles = build_cycles(hist, pantry_u[pantry_u["product_id"] == p])
            if cycles.empty:
                continue

            # ciclo mais recente anterior ao corte
            cycles = cycles.sort_values("purchase_time")
            last = cycles[cycles["purchase_time"] <= t_cut].tail(1)
            if last.empty:
                continue

            # consumo histórico no histórico (opcional: todo hist do user)
            cons = estimate_user_product_consumption(cycles)

            last = last.reset_index(drop=True)
            purchase_time = last.loc[0, "purchase_time"]
            qty = float(last.loc[0, "quantity_bought"] or 0.0)
            vfd = last.loc[0, "valid_for_days"]
            has_v = float(last.loc[0, "has_validity_info"] if "has_validity_info" in last.columns else float(~pd.isna(vfd)))
            # consumo
            cons_uday = 1.0
            if not cons.empty:
                key = cons.set_index(["user_id","product_id"])
                if (u, p) in key.index:
                    cons_uday = float(key.loc[(u,p), "consumption_uday"])

            days_since_purchase = max((t_cut - purchase_time).total_seconds()/86400.0, 0.0)
            expected_depletion_days = max(qty / max(cons_uday, 1e-3), 0.5)

            ratio_to_validity  = days_since_purchase / (vfd if (pd.notna(vfd) and vfd != 0) else np.nan)
            ratio_to_depletion = days_since_purchase / (expected_depletion_days if expected_depletion_days != 0 else np.nan)

            y_fb = 1 if (bool(getattr(r, "is_out")) or bool(getattr(r, "is_expired"))) else 0

            rows.append({
                "user_id": u, "product_id": p,
                "quantity_bought": qty,
                "valid_for_days": vfd,
                "has_validity_info": has_v,
                "consumption_uday": cons_uday,
                "expected_depletion_days": expected_depletion_days,
                "ratio_to_validity": ratio_to_validity if pd.notna(ratio_to_validity) else 0.0,
                "ratio_to_depletion": ratio_to_depletion if pd.notna(ratio_to_depletion) else 0.0,
                "y": y_fb
            })

    return pd.DataFrame(rows)

