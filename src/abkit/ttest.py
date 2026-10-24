import numpy as np
import math
from scipy.stats import t
from abkit.ztest import ComparisonResult


def welch_t_test(control:np.ndarray, treatment:np.ndarray, alpha=0.05, pooled=False) -> ComparisonResult:
    """Statistical comparison between treatment and control for means of continuous random variables."""

    control_copy = np.asarray(control, dtype=float)
    treatment_copy = np.asarray(treatment, dtype=float)

    n_c = control_copy.size
    n_t = treatment_copy.size

    if n_c < 2 or n_t < 2:
        raise ValueError("group cannot be less than 2 values")

    std_c_hat = control_copy.std(ddof=1) # estimate of standard deviation in control
    std_t_hat = treatment_copy.std(ddof=1)
    v_c = control.var(ddof=1) / n_c # estimate of variance
    v_t = treatment.var(ddof=1) / n_t # estimate of variance

    mean_c = control_copy.mean() # mean in control group
    mean_t = treatment_copy.mean()

    d_hat = mean_t - mean_c

    se = math.sqrt(std_c_hat**2 / n_c + std_t_hat**2 / n_t)

    if se == 0:
        raise ValueError("SE cannot be 0")

    df = (v_c  + v_t)**2 / (v_c**2 / (n_c - 1) + v_t**2 / (n_t - 1))

    t_stat = d_hat / se

    p_value = 2*(t.sf(abs(t_stat), df))

    ci_low = d_hat - se * t.ppf(1 - alpha / 2, df)
    ci_high = d_hat + se * t.ppf(1 - alpha / 2, df)



    return ComparisonResult(estimate=d_hat, standard_error=se, statistic=t_stat, p_value=p_value,
                            ci_low=ci_low, ci_high=ci_high, alpha=alpha, df=df)




