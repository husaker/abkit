"""Acceptance tests for chapter 2: the HC2 standard error, and why it is Welch's."""

import numpy as np
import pytest
import statsmodels.api as sm

from abkit.regression import ols
from abkit.ttest import welch_t_test


def two_groups(control, treatment):
    y = np.r_[control, treatment]
    treated = np.r_[np.zeros(len(control)), np.ones(len(treatment))]
    return y, np.column_stack([np.ones_like(y), treated])


def test_hc2_matches_statsmodels():
    rng = np.random.default_rng(11)
    x1, x2 = rng.normal(size=200), (rng.random(200) < 0.3).astype(float)
    y = 2.0 + x1 + 0.5 * x2 + rng.normal(size=200) * (0.5 + np.abs(x1))
    X = sm.add_constant(np.column_stack([x1, x2]))

    result = ols(y, X)

    np.testing.assert_allclose(result.hc2_se, sm.OLS(y, X).fit(cov_type="HC2").bse, rtol=1e-9)


def test_hc2_is_never_smaller_than_hc0():
    # HC2 divides each squared residual by (1 − leverage), a number below 1.
    rng = np.random.default_rng(12)
    X = sm.add_constant(rng.normal(size=(50, 2)))
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=50)

    result = ols(y, X)

    assert np.all(result.hc2_se >= result.robust_se)


def test_hc2_slope_is_the_welch_standard_error():
    rng = np.random.default_rng(13)
    control, treatment = rng.exponential(3.0, 900), rng.exponential(8.0, 100)

    slope_se = ols(*two_groups(control, treatment)).hc2_se[1]

    assert slope_se == pytest.approx(welch_t_test(control, treatment).standard_error, rel=1e-9)
