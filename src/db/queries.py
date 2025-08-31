from sqlalchemy import text
import pandas as pd
from sqlalchemy.engine import Engine

ITEM_TX_SQL = """
SELECT
  id,
  user_id,
  product_id,
  state,
  portion,
  created_at
FROM item_transaction
WHERE created_at >= NOW() - INTERVAL '365 days'
ORDER BY user_id, product_id, created_at
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

def fetch_item_transactions(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(ITEM_TX_SQL), conn)
    return df

def fetch_pantry_validities(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(PANTRY_SQL), conn)
    return df
