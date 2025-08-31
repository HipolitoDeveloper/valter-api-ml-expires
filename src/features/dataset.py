import pandas as pd
import numpy as np

def build_training_dataset(cycles_df: pd.DataFrame, cons_df: pd.DataFrame) -> pd.DataFrame:
    df = cycles_df.copy()
    df = df.dropna(subset=["next_in_cart_time", "days_to_restock"])
    if df.empty:
        return pd.DataFrame(columns=["user_id","product_id","quantity_bought",
                                     "valid_for_days","days_to_restock","consumption_uday",
                                     "expected_depletion_days","ratio_to_validity","ratio_to_depletion","y"])

    cons_key = cons_df.set_index(["user_id","product_id"]) if not cons_df.empty else None
    consumption = []
    for i, row in df.iterrows():
        if cons_key is not None and (row["user_id"], row["product_id"]) in cons_key.index:
            consumption.append(cons_key.loc[(row["user_id"], row["product_id"]), "consumption_uday"])
        else:
            consumption.append(1.0)
    df["consumption_uday"] = np.array(consumption).astype(float)

    df["expected_depletion_days"] = (df["quantity_bought"] / df["consumption_uday"].clip(lower=1e-3)).clip(lower=0.5)
    df["ratio_to_validity"] = df["days_to_restock"] / df["valid_for_days"].replace(0, np.nan)
    df["ratio_to_depletion"] = df["days_to_restock"] / df["expected_depletion_days"].replace(0, np.nan)

    df["y"] = (df["days_to_restock"] >= np.minimum(df["valid_for_days"], df["expected_depletion_days"])).astype(int)

    features = df[[
        "quantity_bought", "valid_for_days",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)

    return pd.concat([df[["user_id","product_id","days_to_restock"]], features, df[["y"]]], axis=1)


def build_inference_rows(now_ts, latest_cycles: pd.DataFrame, cons_df: pd.DataFrame):
    if latest_cycles.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest = (latest_cycles.sort_values("purchase_time")
                           .groupby(["user_id","product_id"]).tail(1)
                           .reset_index(drop=True))
    latest["days_since_purchase"] = ((now_ts - latest["purchase_time"]).dt.total_seconds() / 86400.0).clip(lower=0)

    cons_key = cons_df.set_index(["user_id","product_id"]) if not cons_df.empty else None
    latest["consumption_uday"] = 1.0
    if cons_key is not None:
        idx = latest.set_index(["user_id","product_id"]).index
        inter = idx.intersection(cons_key.index)
        if len(inter) > 0:
            for i, r in latest.iterrows():
                key = (r["user_id"], r["product_id"])
                if key in cons_key.index:
                    latest.at[i, "consumption_uday"] = cons_key.loc[key, "consumption_uday"]

    latest["expected_depletion_days"] = (latest["quantity_bought"] / latest["consumption_uday"].clip(lower=1e-3)).clip(lower=0.5)
    latest["ratio_to_validity"] = latest["days_since_purchase"] / latest["valid_for_days"].replace(0, np.nan)
    latest["ratio_to_depletion"] = latest["days_since_purchase"] / latest["expected_depletion_days"].replace(0, np.nan)

    X = latest[[
        "quantity_bought", "valid_for_days",
        "consumption_uday", "expected_depletion_days",
        "ratio_to_validity", "ratio_to_depletion"
    ]].fillna(0.0)
    meta = latest[["user_id","product_id"]].copy()
    return X, meta
