from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"
needs_data = pytest.mark.skipif(
    not (PROCESSED / "sessions.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@needs_data
def test_one_row_per_session():
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    assert not sessions.duplicated(["user_id", "user_session"]).any()

def test_events_count():
    events = pd.read_parquet(PROCESSED / "events.parquet")
    assert len(events) == 20692840 #total number of raw events

def test_revenue_aggregation():
    revenue_agg = pd.read_parquet(PROCESSED / "sessions.parquet")["revenue"].sum()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    revenue_raw = events["price"].where(events["event_type"]=="purchase").sum()
    assert round(revenue_raw, 2) == round(revenue_agg, 2)
