from dataclasses import dataclass
import numpy as np
import math
from abkit.ztest import ComparisonResult
from statistics import NormalDist
import pytest

@dataclass(frozen=True)
class RatioMetric:

    """Metric used to calculate ratio between two groups"""

    numerator: np.ndarray
    denominator: np.ndarray

    def __post_init__(self) -> None:
        numerator = np.array(self.numerator, dtype=float, copy=True)
        denominator = np.array(self.denominator, dtype=float, copy=True)

        numerator.setflags(write=False)
        denominator.setflags(write=False)

        if len(np.array(numerator).shape) > 1 or len(np.array(denominator).shape) > 1:
            raise ValueError("numerator and denominator must be 1D")
        if denominator.sum() == 0:
            raise ValueError("denominator cannot be zero")
        if numerator.size != denominator.size:
            raise ValueError("numerator and denominator must have the same length")
        if numerator.size <= 1 or denominator.size <= 1:
            raise ValueError("there must be at least two users")
        if any(np.isnan(numerator)) or any(np.isnan(denominator)):
            raise ValueError("values can't be NaNs")
        if np.any(numerator < 0) or np.any(denominator < 0):
            raise ValueError("values can't be negative")

        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    @property
    def n_users(self) -> int:
        return self.numerator.size

    @property
    def ratio(self) ->float:
        return np.sum(self.numerator) / np.sum(self.denominator)

    @property
    def standard_error(self) -> float:
        x_dash = self.numerator.mean()
        y_dash = self.denominator.mean()
        n_x = self.numerator.size
        n_y = self.denominator.size

        s_x = math.sqrt(np.sum((self.numerator - x_dash)**2) / (n_x - 1))
        s_y = math.sqrt(np.sum((self.denominator - y_dash)**2) / (n_y - 1))
        cov_xy = np.sum((self.numerator - x_dash)*(self.denominator - y_dash)) / (n_x - 1)

        var = (s_x**2 / y_dash**2 + (x_dash**2 * s_y**2) / y_dash**4 - (2*x_dash*cov_xy) / y_dash**3) / n_x


        return math.sqrt(var)

    def linearize(self, ratio: float| None = None) -> np.ndarray:
        k = self.ratio if ratio is None else ratio
        return self.numerator - k * self.denominator


def delta_ratio_test(control: RatioMetric, treatment: RatioMetric, *, alpha:float=0.05) -> ComparisonResult:
    """Comparison between two groups."""

    if alpha < 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    d_hat = treatment.ratio - control.ratio

    se_d = math.sqrt(control.standard_error**2 + treatment.standard_error**2)

    if se_d == 0:
        raise ValueError("zero variance")

    z_stat = (d_hat) / se_d

    ci_low = d_hat - se_d * NormalDist().inv_cdf(1 - alpha / 2)
    ci_high = d_hat + se_d * NormalDist().inv_cdf(1 - alpha / 2)

    p_value = math.erfc(abs(z_stat) / np.sqrt(2))

    return ComparisonResult(estimate=d_hat,
                            standard_error=se_d,
                            statistic=z_stat,
                            p_value=p_value,
                            ci_low=ci_low,
                            ci_high=ci_high,
                            alpha=alpha)


def relative_lift_test(control: np.ndarray, treatment: np.ndarray, *, alpha:float=0.05) -> ComparisonResult:
    """Comparison between two groups."""

    if alpha < 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")

    control_copy = np.asarray(control, dtype=float)
    treatment_copy = np.asarray(treatment, dtype=float)

    if control_copy.sum() == 0:
        raise ValueError("zero variance")
    if control_copy.size <= 1 or treatment_copy.size <= 1:
        raise ValueError("treatment and control must have at least 2 observations")
    if any(control_copy == np.inf) or any(treatment_copy == np.inf):
        raise ValueError("treatment and control have inf number")

    v_c = control_copy.var(ddof=1) / control_copy.size # estimate of variance
    v_t = treatment_copy.var(ddof=1) / treatment_copy.size # estimate of variance

    c_mean = np.mean(control_copy)
    t_mean = np.mean(treatment_copy)

    d_hat_rel = t_mean / c_mean - 1

    se_d = math.sqrt(v_t / c_mean**2 + t_mean**2 * v_c / c_mean**4)

    if se_d == 0:
        raise ValueError("zero variance")

    z_stat = d_hat_rel / se_d

    ci_low = d_hat_rel - se_d * NormalDist().inv_cdf(1 - alpha / 2)
    ci_high = d_hat_rel + se_d * NormalDist().inv_cdf(1 - alpha / 2)

    p_value = math.erfc(abs(z_stat) / np.sqrt(2))

    return ComparisonResult(estimate=d_hat_rel,
                            standard_error=se_d,
                            statistic=z_stat,
                            p_value=p_value,
                            ci_low=ci_low,
                            ci_high=ci_high,
                            alpha=alpha)


SESSIONS = [1, 2, 3, 4, 10]
SHOP_A = ([0, 1, 1, 1, 1], SESSIONS)
SHOP_B = ([1, 1, 1, 1, 0], SESSIONS)
SHOP_C = ([1, 1, 1, 1, 1], SESSIONS)

# The chapter's five-user revenue groups from part 0 of chapter 4.
CONTROL = np.array([2.0, 4.0, 4.0, 6.0, 9.0])
TREATMENT = np.array([3.0, 5.0, 7.0, 8.0, 12.0])

result = np.array([1, float("inf")])
print(any(result == np.inf))
