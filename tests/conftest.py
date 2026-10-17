import pandas as pd
import pytest

COLUMNS = ["event_time", "event_type", "product_id", "price", "user_id", "user_session"]


@pytest.fixture
def make_events():
    """Build a tiny events DataFrame from (time, type, product, price, user, session) tuples."""

    def _make(rows):
        events = pd.DataFrame(rows, columns=COLUMNS)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        return events

    return _make