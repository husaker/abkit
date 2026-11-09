"""CUPED: variance reduction with a pre-experiment covariate (Deng, Xu, Kohavi and Walker, 2013).

Each user's metric Y is replaced by Y - theta * (X - mean(X)), where X is a covariate measured
before the experiment, such as the same metric over the pre-experiment period, and
theta = cov(Y, X) / var(X). The adjusted metric has the same mean and a variance smaller by the
factor 1 - rho**2, where rho is the correlation of Y and X, so the same experiment can detect
smaller effects.
"""

from dataclasses import dataclass
import numpy as np
from abkit.ztest import ComparisonResult
from abkit.ttest import welch_t_test


def theta(values: np.ndarray, covariate: np.ndarray) -> float:
    """Slope of the metric on the covariate, cov(Y, X) / var(X).

    This is the theta of Deng, Xu, Kohavi and Walker (2013), equation (4): the value that
    makes the variance of Y - theta * X as small as possible. It equals the least-squares
    slope of a regression of the metric on the covariate, and it is the same whether the
    variance and the covariance divide by n or by n - 1.

    Parameters
    ----------
    values : np.ndarray
        The metric Y, one value per user.
    covariate : np.ndarray
        The covariate X, one value per user, in the same user order as values.

    Returns
    -------
    float
        cov(Y, X) / var(X).

    Raises
    ------
    ValueError
        If either array is not one-dimensional, contains nan or infinity, if the two differ
        in length or hold fewer than two users, or if the covariate is constant, so that
        var(X) is zero and theta is undefined.
    """

    if len(np.array(values).shape) > 1 or len(np.array(covariate).shape) > 1:
        raise ValueError("numerator and denominator must be 1D")
    if not (np.isfinite(values).all() and np.isfinite(covariate).all()):
        raise ValueError("values and covariate must be finite")
    if covariate.sum() == 0:
        raise ValueError("denominator cannot be zero")
    if covariate.var() == 0:
        raise ValueError("covariate cannot be constant")
    if values.size != covariate.size:
        raise ValueError("numerator and denominator must have the same length")
    if values.size <= 1 or covariate.size <= 1:
        raise ValueError("there must be at least two users")
    if any(np.isnan(values)) or any(np.isnan(covariate)):
        raise ValueError("values can't be NaNs")

    var_x = np.var(covariate, ddof=1)
    cov = np.cov(values, covariate, ddof=1)[0, 1]

    return cov / var_x

@dataclass(frozen=True)
class UserMetric:
    """A per-user metric together with a pre-experiment covariate, for CUPED.

    Holds the metric Y and the covariate X of one group of users, one value of each per
    user, in the same user order. The covariate is a quantity measured before the
    experiment, such as the same metric over the pre-experiment period, that is correlated
    with Y but cannot be affected by the treatment (Deng et al. 2013, section 3.2.2). The
    object is frozen: both inputs are copied to read-only float arrays when it is created,
    so changing them afterwards does not change the metric.

    A constant covariate is accepted when the object is created, because the group can
    still be adjusted with a theta computed on a larger sample, as cuped_test does. Only
    the group's own theta and correlation are undefined then, and those two raise.

    Attributes
    ----------
    values : np.ndarray
        The metric Y, a read-only float array with one value per user.
    covariate : np.ndarray
        The covariate X, a read-only float array with one value per user.
    n_users : int
        Number of users.
    theta : float
        The group's own slope cov(Y, X) / var(X); see theta().
    correlation : float
        Pearson correlation between Y and X, the rho in the variance factor 1 - rho**2.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, if the two differ in length or hold fewer
        than two users, or if either contains nan or infinity.
    """

    values: np.ndarray
    covariate: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        covariate = np.array(self.covariate, dtype=float, copy=True)

        values.setflags(write=False)
        covariate.setflags(write=False)

        if len(np.array(values).shape) > 1 or len(np.array(covariate).shape) > 1:
            raise ValueError("values and covariate must be 1D")
        if covariate.sum() == 0:
            raise ValueError("covariate cannot be zero")
        if values.size != covariate.size:
            raise ValueError("values and covariate must have the same length")
        if values.size <= 1 or covariate.size <= 1:
            raise ValueError("there must be at least two users")
        if any(np.isnan(values)) or any(np.isnan(covariate)):
            raise ValueError("values can't be NaNs")
        if any(values == np.inf) or any(covariate == np.inf):
            raise ValueError("values can't be infinity")


        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariate", covariate)

    @property
    def n_users(self) -> int:
        """Number of users."""
        return self.values.size

    @property
    def theta(self) -> float:
        """The group's own slope cov(Y, X) / var(X); see theta()."""
        return theta(self.values, self.covariate)

    @property
    def correlation(self) -> float:
        """Pearson correlation between the metric and the covariate.

        Raises
        ------
        ValueError
            If either array is constant, so that one standard deviation is zero and the
            correlation is undefined.
        """
        if self.values.std() == 0 or self.covariate.std() == 0:
            raise ValueError("the correlation is undefined when either array is constant")
        return float(np.corrcoef(self.values, self.covariate)[0, 1])


    def adjust(self, theta:float| None=None, covariate_mean:float | None=None) -> np.ndarray:
        """CUPED-adjusted values, Y - theta * (X - covariate_mean), as a new array.

        Subtracting theta * (X - covariate_mean) removes the part of each user's metric that
        the covariate predicts. When covariate_mean is the mean of X the adjusted values keep
        the mean of Y, and with the group's own theta their variance is var(Y) * (1 - rho**2)
        (Deng et al. 2013, section 3.2). cuped_test passes the same theta and covariate_mean
        to both groups, computed on the pooled sample, so that the two groups are adjusted in
        the same way.

        Parameters
        ----------
        theta : float, optional
            Slope to use. By default the group's own theta, which is undefined when the
            covariate is constant.
        covariate_mean : float, optional
            Value the covariate is centred on. By default the mean of this group's covariate.

        Returns
        -------
        np.ndarray
            A new float array with one adjusted value per user; the stored arrays are not
            modified.
        """
        if theta is None:
            theta_adj = self.theta
        else:
            theta_adj = theta

        if covariate_mean is None:
            mean = float(self.covariate.mean())
        else:
            mean = covariate_mean

        return self.values - theta_adj * (self.covariate - mean)


def cuped_test(control: UserMetric, treatment: UserMetric, *, alpha:float=0.05) -> ComparisonResult:
    """Compare two means after a CUPED adjustment with a pre-experiment covariate.

    Implements the control-variates estimator of Deng, Xu, Kohavi and Walker (2013): one
    theta and one covariate mean are computed on the two groups pooled together, every
    user's value is adjusted to Y - theta * (X - mean), and the adjusted groups are
    compared with Welch's t-test. The difference in adjusted means is the in-experiment
    delta corrected by the delta in the covariate,
    (mean(Y_T) - mean(Y_C)) - theta * (mean(X_T) - mean(X_C)). Because the covariate was
    measured before the experiment, randomisation makes the expected delta in X zero, so
    the estimate is unbiased, while its standard error shrinks by about the factor
    sqrt(1 - rho**2).

    Parameters
    ----------
    control, treatment : UserMetric
        The metric and the covariate of each group.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    ComparisonResult
        Difference in adjusted means (treatment minus control), its standard error, t, the
        p-value, the confidence interval and the Welch degrees of freedom, as returned by
        welch_t_test.

    Raises
    ------
    ValueError
        If alpha is not strictly between 0 and 1, or if the covariate is constant in either
        group, so that theta is undefined.

    Notes
    -----
    The covariate must not be affected by the treatment. A covariate measured during the
    experiment absorbs part of the effect and biases the estimate (Deng et al. 2013,
    sections 4.3 and 5.3).
    """

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in [0, 1]")

    pooled_val = np.concatenate((control.values, treatment.values))
    pooled_cov = np.concatenate((control.covariate, treatment.covariate))

    if np.isclose(control.covariate.var(), 0) or np.isclose(treatment.covariate.var(), 0):
        raise ValueError("covariate cannot be constant")

    theta_pooled = theta(pooled_val, pooled_cov)
    cov_mean = pooled_cov.mean()


    control_adj = control.adjust(theta_pooled, cov_mean)
    treat_adj = treatment.adjust(theta_pooled, cov_mean)

    res = welch_t_test(control_adj, treat_adj, alpha)

    return res

