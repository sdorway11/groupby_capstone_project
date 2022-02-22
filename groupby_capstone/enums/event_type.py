from enum import Enum


class EventType(Enum):
    VIEW = "view"
    CART = "cart"
    REMOVE = "remove_from_cart"
    PURCHASE = "purchase"
