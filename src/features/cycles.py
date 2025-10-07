import pandas as pd
import numpy as np
from src.types.enums import TxState

PURCHASE_STATES = {TxState.PURCHASED, TxState.IN_PANTRY}
ANCHOR_STATES   = {TxState.IN_CART, TxState.PURCHASED, TxState.IN_PANTRY}
TERMINAL_STATES = {TxState.OUT, TxState.EXPIRED}  # ✅

def build_cycles(items_df: pd.DataFrame, pantry_df: pd.DataFrame) -> pd.DataFrame:
    if items_df is None or items_df.empty:
        return pd.DataFrame(columns=[
            "user_id","product_id","purchase_time","next_in_cart_time",
            "quantity_bought","had_removed_between",
            "valid_for_days","has_validity_info","days_to_restock",
            "terminal_time","terminal_state"  # ✅ novos
        ])

    items = items_df.copy()

    # normalizações
    for col in ("user_id","product_id","transaction_code"):
        if col in items.columns:
            items[col] = items[col].astype(str).str.strip()

    if "quantity" not in items.columns and "portion" in items.columns:
        items = items.rename(columns={"portion": "quantity"})

    items["created_at"] = pd.to_datetime(items["created_at"], errors="coerce", utc=True)
    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(0.0)

    if "valid_for_days" in items.columns:
        items["valid_for_days"] = pd.to_numeric(items["valid_for_days"], errors="coerce")
        items.loc[items["valid_for_days"] == 0, "valid_for_days"] = np.nan
    else:
        items["valid_for_days"] = np.nan

    pantry = pantry_df.copy() if pantry_df is not None else pd.DataFrame()
    if not pantry.empty:
        for col in ("user_id","product_id"):
            pantry[col] = pantry[col].astype(str).str.strip()
        pantry_key = pantry.set_index(["user_id","product_id"])
    else:
        pantry_key = None

    items.sort_values(["user_id","product_id","transaction_code","created_at"], inplace=True)

    rows = []
    for (u, p), grp_up in items.groupby(["user_id","product_id"], sort=False):
        # ordenar códigos por menor created_at
        codes = (grp_up.groupby("transaction_code")["created_at"]
                        .min().sort_values().index.tolist())
        grouped = {
            code: grp_up[grp_up["transaction_code"] == code].sort_values("created_at").reset_index(drop=True)
            for code in codes
        }

        for idx, code in enumerate(codes):
            g = grouped[code].copy()

            last_notification_at = g.loc[0, "last_notification_at"]

            # ✅ aplicar UPDATE no último âncora
            last_anchor_idx = None
            for i, row in g.iterrows():
                st = row["state"]
                if st in ANCHOR_STATES:
                    last_anchor_idx = i
                elif st == TxState.UPDATE and last_anchor_idx is not None:
                    if pd.notna(row.get("quantity", np.nan)):
                        g.at[last_anchor_idx, "quantity"] = float(row["quantity"])
                    if pd.notna(row.get("valid_for_days", np.nan)):
                        g.at[last_anchor_idx, "valid_for_days"] = float(row["valid_for_days"])

            # primeiro abastecimento do ciclo
            mask_purchase = g["state"].isin(PURCHASE_STATES)
            if mask_purchase.any():
                first_purchase_idx = g[mask_purchase].index[0]
                purchase_time = g.loc[first_purchase_idx, "created_at"]
            else:
                purchase_time = g.loc[0, "created_at"]

            # soma de compras (após UPDATEs)
            quantity_bought = g.loc[mask_purchase, "quantity"].sum()

            # removed depois do primeiro purchase
            had_removed_between = False
            if mask_purchase.any():
                after = g.loc[first_purchase_idx+1:]
                if not after.empty and (after["state"] == TxState.REMOVED).any():
                    had_removed_between = True

            # validade do ciclo (preferir valores do ciclo; fallback pantry)
            vfd_cycle = g["valid_for_days"].astype(float).replace({0.0: np.nan})
            if vfd_cycle.notna().any():
                valid_for_days = float(vfd_cycle.dropna().median())
                has_validity_info = 1.0
            else:
                valid_for_days = np.nan
                has_validity_info = 0.0
                if pantry_key is not None and (u, p) in pantry_key.index:
                    v = pantry_key.loc[(u, p), "valid_for_days"]
                    try:
                        vf = float(v)
                        if np.isfinite(vf) and vf > 0:
                            valid_for_days = vf
                            has_validity_info = 1.0
                    except Exception:
                        pass

            # início do próximo ciclo
            next_in_cart_time = None
            if idx + 1 < len(codes):
                g_next = grouped[codes[idx + 1]].sort_values("created_at")
                next_in_cart_time = g_next["created_at"].iloc[0]

            # ✅ terminal dentro do ciclo (primeiro OUT/EXPIRED após purchase)
            terminal_time = None
            terminal_state = None
            if mask_purchase.any():
                seq_after = g.loc[first_purchase_idx+1:]
                term_mask = seq_after["state"].isin(TERMINAL_STATES)
                if term_mask.any():
                    first_term = seq_after[term_mask].iloc[0]
                    terminal_time = first_term["created_at"]
                    terminal_state = str(first_term["state"])

            # fim do ciclo: o que vier primeiro (terminal OU próximo ciclo)
            end_candidates = [t for t in [terminal_time, next_in_cart_time] if t is not None]
            end_time = min(end_candidates) if end_candidates else None

            days_to_restock = None
            if end_time is not None and purchase_time is not None:
                days_to_restock = max((end_time - purchase_time).days, 0)

            rows.append({
                "user_id": u,
                "product_id": p,
                "purchase_time": purchase_time,
                "next_in_cart_time": next_in_cart_time,
                "quantity_bought": float(quantity_bought or 0.0),
                "had_removed_between": had_removed_between,
                "valid_for_days": valid_for_days,
                "has_validity_info": has_validity_info,
                "days_to_restock": days_to_restock,
                "terminal_time": terminal_time,
                "terminal_state": terminal_state,
                "last_notification_at": last_notification_at
            })

    return pd.DataFrame(rows)
