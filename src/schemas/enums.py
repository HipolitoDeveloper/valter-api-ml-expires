from enum import Enum

class TxState(str, Enum):
    IN_CART = "IN_CART"
    PURCHASED = "PURCHASED"
    IN_PANTRY = "IN_PANTRY"
    REMOVED = "REMOVED"
    UPDATE     = "UPDATE"
    EXPIRED = "EXPIRED"
    OUT = "OUT"
