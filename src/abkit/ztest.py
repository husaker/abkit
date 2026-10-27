import math
import numpy as np
from statistics import NormalDist
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult():
    """Result of comparing a treatment group with a control group.

    Attributes
    ----------
    estimate : float
        Estimated difference, treatment minus control.
    standard_error : float
        Unpooled standard error of the estimate; the confidence interval uses it.
    statistic : float
        Test statistic, z or t. A pooled z-test computes it with the pooled standard
        error, so it can differ from estimate / standard_error.
    p_value : float
        Two-sided p-value.
    ci_low, ci_high : float
        Bounds of the two-sided (1 - alpha) confidence interval for the difference.
    alpha : float
        Significance level of the interval.
    df : float, default math.inf
        Degrees of freedom of the t reference distribution; math.inf means the
        standard normal, as in a z-test.
    """

    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    df: float = math.inf



def z_test_proportions(control:np.ndarray, treatment:np.ndarray, alpha: float=0.05, pooled: bool=False) -> ComparisonResult:
    """Compare two conversion rates with a two-sided z-test.

    Parameters
    ----------
    control, treatment : np.ndarray
        One 0/1 value per user in each group.
    alpha : float, default 0.05
        Significance level for the confidence interval.
    pooled : bool, default False
        If True, the test statistic uses the pooled standard error, which assumes H0
        (equal rates). The confidence interval always uses the unpooled one.

    Returns
    -------
    ComparisonResult
        Difference in rates (treatment minus control), its unpooled standard error,
        z, the p-value and the confidence interval.

    Raises
    ------
    ValueError
        If a group contains values other than 0 and 1, a group is empty, or the
        standard error is 0 (each group is all 0s or all 1s).
    """

    if not np.isin(control, [0, 1]).all():
        raise ValueError(f"control values must be only 0s and 1s")
    elif not np.isin(treatment, [0, 1]).all():
        raise ValueError(f"treatment values must be only 0s and 1s")
    elif len(control) == 0 or len(treatment) == 0:
        raise ValueError(f"control and treatment values must not be empty")

    n_c = control.shape[0]
    n_t = treatment.shape[0]

    p_c = control.mean()
    p_t = treatment.mean()
    d = p_t - p_c # difference between groups
    se = math.sqrt((p_c*(1-p_c) / n_c) + (p_t*(1-p_t) / n_t))

    if se == 0:
        raise ValueError("SE cannot be zero")

    if pooled: # calculating pooled variance
        p_hat = (control.sum() + treatment.sum()) / (n_c+n_t)
        se_test = math.sqrt(p_hat*(1-p_hat) * (1 / n_c + 1 / n_t))
    else:
        se_test = se

    z_stat = (d-0) / se_test

    ci_low = d - se * NormalDist().inv_cdf(1 - alpha / 2)
    ci_high = d + se * NormalDist().inv_cdf(1 - alpha / 2)
    p_value = math.erfc(abs(z_stat) / np.sqrt(2))

    return ComparisonResult(estimate=d,
                            standard_error=se,
                            statistic=z_stat,
                            p_value=p_value,
                            ci_low=ci_low,
                            ci_high=ci_high,
                            alpha=alpha)



