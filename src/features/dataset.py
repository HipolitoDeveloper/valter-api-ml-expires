import pandas as pd
import numpy as np

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
            consumption.append(1.0)  # fallback consumo
    df["consumption_uday"] = np.array(consumption, dtype=float)

    df["expected_depletion_days"] = (df["quantity_bought"] / df["consumption_uday"].clip(lower=1e-3)).clip(lower=0.5)

    # Pode dar NaN quando valid_for_days é NaN → OK, será preenchido depois
    df["ratio_to_validity"] = df["days_to_restock"] / df["valid_for_days"].replace(0, np.nan)
    df["ratio_to_depletion"] = df["days_to_restock"] / df["expected_depletion_days"].replace(0, np.nan)

    # Rótulo
    df["y"] = (df["days_to_restock"] >= np.minimum(df["valid_for_days"], df["expected_depletion_days"])).astype(int)

    features = df[[
        "quantity_bought", "valid_for_days", "has_validity_info",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)

    return pd.concat([df[["user_id","product_id","days_to_restock"]], features, df[["y"]]], axis=1)


def build_inference_rows(now_ts, latest_cycles: pd.DataFrame, cons_df: pd.DataFrame):
    """
    Usa o ciclo mais recente por (user,product). Mantém:
      - valid_for_days possivelmente NaN
      - has_validity_info em {0.0, 1.0}
    """
    if latest_cycles.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest = (latest_cycles.sort_values("purchase_time")
                           .groupby(["user_id","product_id"])
                           .tail(1)
                           .reset_index(drop=True))

    # garantir UTC-aware
    if "purchase_time" in latest.columns:
        latest["purchase_time"] = pd.to_datetime(latest["purchase_time"], errors="coerce", utc=True)

    if not hasattr(now_ts, "tzinfo") or now_ts.tzinfo is None:
        now_ts = pd.Timestamp(now_ts, tz="UTC")
    else:
        now_ts = pd.Timestamp(now_ts).tz_convert("UTC")

    latest["days_since_purchase"] = ((now_ts - latest["purchase_time"]).dt.total_seconds() / 86400.0).clip(lower=0)

    cons_key = cons_df.set_index(["user_id","product_id"]) if not cons_df.empty else None
    latest["consumption_uday"] = 1.0
    if cons_key is not None:
        for i, r in latest.iterrows():
            k = (r["user_id"], r["product_id"])
            if k in cons_key.index:
                latest.at[i, "consumption_uday"] = float(cons_key.loc[k, "consumption_uday"])

    latest["expected_depletion_days"] = (latest["quantity_bought"] / latest["consumption_uday"].clip(lower=1e-3)).clip(lower=0.5)

    if "has_validity_info" not in latest.columns:
        latest["has_validity_info"] = (~latest["valid_for_days"].isna()).astype(float)

    latest["ratio_to_validity"] = latest["days_since_purchase"] / latest["valid_for_days"].replace(0, np.nan)
    latest["ratio_to_depletion"] = latest["days_since_purchase"] / latest["expected_depletion_days"].replace(0, np.nan)

    X = latest[[
        "quantity_bought", "valid_for_days", "has_validity_info",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)
    meta = latest[["user_id","product_id"]].copy()
    return X, meta
