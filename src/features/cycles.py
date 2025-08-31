import pandas as pd
from typing import Tuple
from src.types.enums import TxState

def _mark_purchase_events(df: pd.DataFrame) -> pd.Series:
    return df["state"].isin([TxState.PURCHASED, TxState.IN_PANTRY])

def build_cycles(items_df: pd.DataFrame, pantry_df: pd.DataFrame) -> pd.DataFrame:
    if items_df.empty:
        return pd.DataFrame(columns=[
            "user_id","product_id","purchase_time","next_in_cart_time",
            "quantity_bought","had_removed_between","valid_for_days","days_to_restock"
        ])

    items_df = items_df.copy()
    items_df["created_at"] = pd.to_datetime(items_df["created_at"])
    items_df.sort_values(["user_id","product_id","created_at"], inplace=True)

    pantry_key = pantry_df.set_index(["user_id","product_id"]) if not pantry_df.empty else None

    rows = []
    for (u, p), grp in items_df.groupby(["user_id","product_id"]):
        grp = grp.reset_index(drop=True)
        is_purchase = _mark_purchase_events(grp)
        idx_purchase = grp[is_purchase].index.tolist()

        for start_idx in idx_purchase:
            purchase_time = grp.loc[start_idx, "created_at"]
            quantity_bought = grp.loc[start_idx, "portion"] or 0
            had_removed_between = False
            next_in_cart_time = None

            j = start_idx + 1
            while j < len(grp):
                state_j = grp.loc[j, "state"]
                t_j = grp.loc[j, "created_at"]

                if state_j == TxState.IN_CART:
                    next_in_cart_time = t_j
                    break
                if state_j in [TxState.PURCHASED, TxState.IN_PANTRY]:
                    break
                if state_j == TxState.REMOVED:
                    had_removed_between = True
                j += 1

            for k in range(start_idx+1, min(j, len(grp))):
                if grp.loc[k,"state"] in [TxState.PURCHASED, TxState.IN_PANTRY]:
                    quantity_bought += (grp.loc[k,"portion"] or 0)

            valid_for_days = 999
            if pantry_key is not None and (u,p) in pantry_key.index:
                v = pantry_key.loc[(u,p),"valid_for_days"]
                try:
                    valid_for_days = float(v)
                except Exception:
                    pass

            days_to_restock = None
            if next_in_cart_time is not None:
                days_to_restock = max((next_in_cart_time - purchase_time).days, 0)

            rows.append({
                "user_id": u,
                "product_id": p,
                "purchase_time": purchase_time,
                "next_in_cart_time": next_in_cart_time,
                "quantity_bought": float(quantity_bought or 0),
                "had_removed_between": had_removed_between,
                "valid_for_days": float(valid_for_days),
                "days_to_restock": days_to_restock
            })

    return pd.DataFrame(rows)
