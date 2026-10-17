"""Unit tests for the two rules that define chapter 0's metrics: revenue and conversion."""

import pytest

from abkit.datasets.rees46 import build_sessions, build_user_months


def test_revenue_counts_only_purchases(make_events):
    # Arrange: one session with every event type. Only the purchase may count.
    events = make_events([
        ("2019-10-05 10:00:00", "view", 1, 12.0, 101, "s1"),
        ("2019-10-05 10:01:00", "cart", 1, 12.0, 101, "s1"),
        ("2019-10-05 10:02:00", "remove_from_cart", 1, 12.0, 101, "s1"),
        ("2019-10-05 10:03:00", "cart", 2, 4.5, 101, "s1"),
        ("2019-10-05 10:05:00", "purchase", 2, 4.5, 101, "s1"),
    ])

    # Act
    sessions = build_sessions(events)
    user_months = build_user_months(events)

    # Assert: 4.5 from the purchase. The 12.0 that was viewed, carted and removed adds nothing.
    assert sessions["revenue"].tolist() == pytest.approx([4.5])
    assert user_months["revenue"].tolist() == pytest.approx([4.5])


@pytest.mark.parametrize(
    "timestamp, expected_month",
    [
        ("2019-10-31 23:59:59", "2019-10"),
        ("2019-11-01 00:00:00", "2019-11"),
    ],
)
def test_user_month_without_purchase_is_not_converted(make_events, timestamp, expected_month):
    # Arrange: a user who only adds to cart, right at a month boundary
    events = make_events([(timestamp, "cart", 7, 3.0, 202, "s2")])

    # Act
    row = build_user_months(events).iloc[0]

    # Assert: the right UTC month, not converted, no revenue
    assert row["month"] == expected_month
    assert not row["converted"]
    assert row["revenue"] == 0
