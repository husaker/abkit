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
    assert sessions["user_session"].is_unique