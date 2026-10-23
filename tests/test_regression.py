"""Acceptance tests for chapter 1: least squares with classical and robust (HC0) standard errors.

The bridge: regressing a 0/1 metric on a treatment indicator gives back the z-test.
"""

import numpy as np
import pytest
import statsmodels.api as sm

from abkit.regression import OLSResult, ols
from abkit.ztest import z_test_proportions


def two_groups(control, treatment):
    """Stack two groups into y and a design matrix X = [intercept, treated]."""
    y = np.r_[control, treatment]
    treated = np.r_[np.zeros(len(control)), np.ones(len(treatment))]
    return y, np.column_stack([np.ones_like(y), treated])


@pytest.fixture
def unequal_groups():
    """A 90/10 split where treatment converts at 30% and control at 10%."""
    rng = np.random.default_rng(7)
    control = (rng.random(900) < 0.10).astype(float)
    treatment = (rng.random(100) < 0.30).astype(float)
    return control, treatment


def test_ols_matches_statsmodels():
    # Any design, not just two groups: three predictors and noise whose spread grows with x1.
    rng = np.random.default_rng(3)
    x1, x2 = rng.normal(size=300), (rng.random(300) < 0.4).astype(float)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=300) * (1 + np.abs(x1))
    X = sm.add_constant(np.column_stack([x1, x2]))

    result = ols(y, X)

    classical = sm.OLS(y, X).fit()
    robust = sm.OLS(y, X).fit(cov_type="HC0")
    assert isinstance(result, OLSResult)
    np.testing.assert_allclose(result.coef, classical.params, rtol=1e-9)
    np.testing.assert_allclose(result.classical_se, classical.bse, rtol=1e-9)
    np.testing.assert_allclose(result.robust_se, robust.bse, rtol=1e-9)


def test_intercept_is_the_control_mean_and_slope_is_the_difference(unequal_groups):
    control, treatment = unequal_groups

    result = ols(*two_groups(control, treatment))

    assert result.coef[0] == pytest.approx(control.mean())
    assert result.coef[1] == pytest.approx(treatment.mean() - control.mean())


def test_robust_standard_error_is_the_unpooled_z_test(unequal_groups):
    control, treatment = unequal_groups

    slope_se = ols(*two_groups(control, treatment)).robust_se[1]

    z = z_test_proportions(control, treatment)
    assert slope_se == pytest.approx(z.standard_error, rel=1e-9)


def test_classical_standard_error_is_the_pooled_variance_formula(unequal_groups):
    control, treatment = unequal_groups
    n_c, n_t = len(control), len(treatment)

    slope_se = ols(*two_groups(control, treatment)).classical_se[1]

    pooled_variance = ((n_c - 1) * control.var(ddof=1) + (n_t - 1) * treatment.var(ddof=1)) / (n_c + n_t - 2)
    assert slope_se == pytest.approx(np.sqrt(pooled_variance * (1 / n_c + 1 / n_t)), rel=1e-9)


def test_equal_groups_make_classical_and_robust_agree():
    rng = np.random.default_rng(8)
    control = (rng.random(2000) < 0.10).astype(float)
    treatment = (rng.random(2000) < 0.30).astype(float)

    result = ols(*two_groups(control, treatment))

    # They differ only by n − 1 vs n inside the variances: a factor of about 1.00025 here.
    assert result.classical_se[1] == pytest.approx(result.robust_se[1], rel=1e-3)


def test_unequal_groups_with_unequal_variances_make_them_disagree(unequal_groups):
    result = ols(*two_groups(*unequal_groups))

    # The classical formula averages the two variances, weighted towards the big, quiet
    # control group, so it understates the noise from the small treatment group.
    assert result.classical_se[1] < 0.8 * result.robust_se[1]


def test_rejects_mismatched_shapes():
    X = np.column_stack([np.ones(9), np.arange(9.0)])

    with pytest.raises(ValueError, match="row"):
        ols(np.zeros(10), X)
