from datetime import datetime, timezone

from groupby_capstone.enums.event_type import EventType


class SessionEvent:
    def __init__(
            self,
            event_type: EventType,
            product_id: str,
            category_id: str,
            category_code: str,
            brand: str,
            price: float,
            user_id: str,
            user_session: str
    ):
        self.event_time = datetime.now(timezone.utc)
        self.event_type = event_type.value
        self.product_id = product_id
        self.category_id = category_id
        self.category_code = category_code
        self.brand = brand
        self.price = price
        self.user_id = user_id
        self.user_session = user_session
        self.year = self.event_time.year
        self.month = self.event_time.month
        self.weekday = self.event_time.weekday()
        self.hour = self.event_time.hour
