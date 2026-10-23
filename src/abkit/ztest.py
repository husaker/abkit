import math
import numpy as np
from statistics import NormalDist
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult():
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float



def z_test_proportions(control:np.ndarray, treatment:np.ndarray, alpha=0.05, pooled=False):
    """Statistical comparison between treatment and control for bernouli random variables."""

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



