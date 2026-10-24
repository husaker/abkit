"""Acceptance tests for chapter 2: capping a heavy-tailed metric (abkit.capping)."""

import numpy as np
import pytest
from scipy import stats

from abkit.capping import cap_upper


@pytest.fixture
def groups():
    """Control holds 1..50 and treatment 51..100, so together they are 1..100."""
    return np.arange(1.0, 51.0), np.arange(51.0, 101.0)


def test_threshold_is_the_quantile_of_both_groups_together(groups):
    control, treatment = groups

    _, _, threshold = cap_upper(control, treatment, quantile=0.9)

    assert threshold == pytest.approx(np.quantile(np.arange(1.0, 101.0), 0.9))   # 90.1


def test_values_above_the_threshold_are_set_to_it(groups):
    control, treatment = groups

    capped_c, capped_t, threshold = cap_upper(control, treatment, quantile=0.9)

    assert capped_t.max() == pytest.approx(threshold)
    assert np.sum(capped_t == threshold) == 10          # 91..100 became 90.1
    np.testing.assert_array_equal(capped_t[:40], treatment[:40])   # 51..90 untouched


def test_both_groups_share_one_threshold(groups):
    # Capping each group at its own 90th percentile would cap control at 45.1. One shared
    # threshold keeps the metric's definition identical in both groups.
    control, treatment = groups

    capped_c, _, _ = cap_upper(control, treatment, quantile=0.9)

    np.testing.assert_array_equal(capped_c, control)


def test_inputs_are_not_modified(groups):
    control, treatment = groups
    before_c, before_t = control.copy(), treatment.copy()

    cap_upper(control, treatment, quantile=0.5)

    np.testing.assert_array_equal(control, before_c)
    np.testing.assert_array_equal(treatment, before_t)


def test_quantile_one_changes_nothing(groups):
    control, treatment = groups

    capped_c, capped_t, threshold = cap_upper(control, treatment, quantile=1.0)

    assert threshold == 100.0
    np.testing.assert_array_equal(capped_c, control)
    np.testing.assert_array_equal(capped_t, treatment)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_rejects_quantiles_outside_zero_to_one(groups, bad):
    with pytest.raises(ValueError):
        cap_upper(*groups, quantile=bad)


def test_capping_reduces_skewness():
    rng = np.random.default_rng(10)
    revenue_like = (rng.random(20_000) < 0.07) * rng.lognormal(3.4, 1.2, 20_000)

    capped_c, capped_t, _ = cap_upper(revenue_like[:10_000], revenue_like[10_000:], quantile=0.99)

    assert stats.skew(np.r_[capped_c, capped_t]) < stats.skew(revenue_like) / 2
