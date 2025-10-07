from sqlalchemy import text
import pandas as pd
from sqlalchemy.engine import Engine
import os

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "720"))
FEEDBACK_DAYS = int(os.getenv("FEEDBACK_DAYS", "365"))

ITEM_TX_SQL =  f"""
SELECT it.id,
       it.user_id::text    AS user_id,
       it.product_id::text AS product_id,
       it.state,
       it.portion,
       it.created_at,
       it.transaction_code,
       ln.last_notification_at

FROM item_transaction it


         LEFT JOIN LATERAL (
    SELECT ne.created_at AS last_notification_at
    FROM notification_expires ne
    WHERE ne.product_id = it.product_id
    ORDER BY ne.created_at DESC
    LIMIT 1
    ) ln ON TRUE

WHERE it.created_at >= NOW() - INTERVAL '{HISTORY_DAYS} days'
ORDER BY it.user_id, it.product_id, it.created_at;

"""

PANTRY_SQL = """
SELECT
    u.id AS user_id,
    product_id,
    valid_for_days
FROM pantry_item
JOIN pantry p on pantry_item.pantry_id = p.id
JOIN public."user" u on p.id = u.pantry_id
"""

FEEDBACK_SQL = f"""
SELECT
  n.user_id::text           AS user_id,
  ne.product_id::text       AS product_id,
  n.created_at              AS notified_at,    -- instante da notificação (corte temporal de features)
  ne.is_out,
  ne.is_expired,
  ne.predicted_probability,
  ne.days_since_last_purchase
FROM notification_expires ne
JOIN notification n ON n.id = ne.notification_id
WHERE n.type = 'PRODUCT_EXPIRES'
  AND n.created_at >= NOW() - INTERVAL '{FEEDBACK_DAYS} days'
"""

def fetch_item_transactions(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(ITEM_TX_SQL), conn)

def fetch_pantry_validities(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(PANTRY_SQL), conn)

def fetch_expiry_feedback(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(FEEDBACK_SQL), conn)
