import pandas as pd

def estimate_user_product_consumption(cycles_df: pd.DataFrame) -> pd.DataFrame:
    df = cycles_df.dropna(subset=["days_to_restock"]).copy()
    df = df[(df["days_to_restock"] > 0) & (df["had_removed_between"] == False)]
    if df.empty:
        return pd.DataFrame(columns=["user_id","product_id","consumption_uday"])

    df["consumption_uday"] = df["quantity_bought"] / df["days_to_restock"].clip(lower=1)
    agg = (df.groupby(["user_id","product_id"])['consumption_uday']
             .median()
             .reset_index())
    return agg
