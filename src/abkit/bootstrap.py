from dataclasses import dataclass, field
from statistics import NormalDist
import numpy as np
from typing import Any, Callable


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap comparison between two groups.

    Attributes
    ----------
    estimate : float
        The statistic computed on the original data: the treatment value minus
        the control value, or the relative lift ``treatment / control - 1``
        when ``relative=True``.
    standard_error : float
        Standard deviation of ``distribution`` (``ddof=1``), the bootstrap
        estimate of the standard error of ``estimate``.
    ci_low, ci_high : float
        Endpoints of the two-sided ``1 - alpha`` confidence interval, computed
        by ``method``.
    alpha : float
        Significance level the interval was built for.
    method : str
        ``"percentile"`` (quantiles of ``distribution``) or ``"normal"``
        (``estimate`` plus or minus a normal quantile times ``standard_error``).
    distribution : np.ndarray
        The ``n_resamples`` resampled values of the statistic. The array is
        read-only so that a result cannot be changed after it is returned.
    """

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str
    distribution: np.ndarray = field()

    def __post_init__(self) -> None:
        arr = np.array(self.distribution, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "distribution", arr)


def bootstrap(control: np.ndarray, treatment: np.ndarray, statistic: Callable[..., Any]=np.mean,
              n_resamples:int=2000, alpha:float=0.05, method:str="percentile",
              relative:bool=False, seed:int =0, batch_size: int | None=None) -> BootstrapResult:

    """Bootstrap the difference (or relative lift) of a statistic between two groups.

    Each group is resampled with replacement at its own size, separately from
    the other group, ``n_resamples`` times. ``statistic`` is evaluated on every
    resample, the treatment value minus the control value (or the relative
    lift) is recorded, and the spread of those recorded values estimates the
    uncertainty of the same quantity computed on the original data.

    Parameters
    ----------
    control, treatment : np.ndarray
        One-dimensional arrays of per-user values, one per group. Each must
        contain at least two values.
    statistic : callable, default np.mean
        Function of the data, called as ``statistic(x)`` on the original
        one-dimensional groups and as ``statistic(x, axis=-1)`` on
        two-dimensional batches of resamples, one resample per row.
    n_resamples : int, default 2000
        Number of bootstrap resaamples. Must be at least 2.
    alpha : float, default 0.05
        Significance level; the interval covers ``1 - alpha``. Must lie
        strictly between 0 and 1.
    method : {"percentile", "normal"}, default "percentile"
        How the interval is built. ``"percentile"`` takes the ``alpha / 2`` and
        ``1 - alpha / 2`` quantiles of the bootstrap distribution
        (``np.quantile`` with its default linear interpolation). ``"normal"``
        uses ``estimate`` plus or minus the normal quantile ``z_{1 - alpha/2}``
        times the bootstrap standard error.
    relative : bool, default False
        If True, record ``statistic(treatment) / statistic(control) - 1``
        instead of the difference, on the original data and on every resample.
    seed : int, default 0
        Seed for ``np.random.default_rng``. The same seed gives the same result.
    batch_size : int or None, default None
        Number of resamples drawn per batch, to bound memory. If None, a size
        is chosen so that one batch of indices takes about 250 MB. Must be at
        least 1 when given.

    Returns
    -------
    BootstrapResult
        The estimate on the original data, its bootstrap standard error, the
        confidence interval, and the full bootstrap distribution.

    Raises
    ------
    ValueError
        If either group has fewer than two values, ``alpha`` is not in (0, 1),
        ``n_resamples`` is below 2, ``method`` is unknown, ``batch_size`` is
        below 1, or ``relative=True`` and the control statistic is zero on the
        original data or on any resample.

    Notes
    -----
    Because the groups are resampled independently, the bootstrap standard
    error of a difference in means is ``sqrt((n - 1) / n)`` times
    ``s / sqrt(n)`` within each group, which is noticeably smaller only for
    very small groups.
    """

    n_c = control.size
    n_t = treatment.size

    if n_c <= 1 or n_t <= 1:
        raise ValueError('in control and treatment groups there should be at least two observations')
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be between 0 and 1')
    if method not in ["percentile", "normal"]:
        raise ValueError('method must be "percentile" or "normal"')
    if n_resamples <= 1:
        raise ValueError('n_resamples must be greater than 1')

    rng = np.random.default_rng(seed)

    if batch_size is None:
        b = max(1, int((250*10**6) / ((n_c+n_t)*64)))
    else:
        b = batch_size

    distribution = np.empty(n_resamples)
    for start in range(0, n_resamples, b):
        stop = min(start + b, n_resamples)

        idx_c = rng.integers(0, n_c, size=(stop - start, n_c))
        idx_t = rng.integers(0, n_t, size=(stop - start, n_t))

        samples_c = statistic(control[idx_c], axis=1)
        samples_t = statistic(treatment[idx_t], axis=1)

        if relative:
            if np.any(samples_c == 0):
                raise ValueError('relative to zero is not possible')
            distribution[start:stop] = samples_t / samples_c - 1
        else:
            distribution[start:stop] = samples_t - samples_c

    standard_error = distribution.std(ddof=1)

    if relative:
        estimate = statistic(treatment) / statistic(control) - 1
    else:
        estimate = statistic(treatment) - statistic(control)

    if method == "percentile":
        ci_low = np.quantile(distribution, alpha/2)
        ci_high = np.quantile(distribution, 1-alpha/2)
    if method == "normal":
        z_stat = NormalDist().inv_cdf(1 - alpha / 2)
        ci_low = estimate - z_stat * standard_error
        ci_high = estimate + z_stat * standard_error

    return BootstrapResult(estimate=estimate, standard_error=standard_error,
                           ci_low=ci_low, ci_high=ci_high, alpha=alpha, method=method,
                           distribution=distribution)

def bootstrap_ratio(control_numerator:np.ndarray, control_denominator:np.ndarray,
                    treatment_numerator:np.ndarray, treatment_denominator:np.ndarray,
                    n_resamples:int=2000, alpha:float=0.05,
                    method:str="percentile", seed:int= 0,
                    batch_size: int | None=None) -> BootstrapResult:
    """Bootstrap the difference of two ratio metrics, resampling whole users.

    A ratio metric such as conversions per session is ``sum(numerator) /
    sum(denominator)`` over users. Sessions of one user are not independent, so
    the unit of resampling is the user: the same resampled indices are applied
    to the numerator and the denominator of a group, and the ratio is
    recomputed from the resampled sums. The recorded quantity is the treatment
    ratio minus the control ratio.

    Parameters
    ----------
    control_numerator, control_denominator : np.ndarray
        Per-user numerator and denominator for the control group, aligned by
        position. Must have the same length, at least 2.
    treatment_numerator, treatment_denominator : np.ndarray
        The same for the treatment group.
    n_resamples : int, default 2000
        Number of bootstrap resamples. Must be at least 2.
    alpha : float, default 0.05
        Significance level; the interval covers ``1 - alpha``. Must lie
        strictly between 0 and 1.
    method : {"percentile", "normal"}, default "percentile"
        How the interval is built; see :func:`bootstrap`.
    seed : int, default 0
        Seed for ``np.random.default_rng``.
    batch_size : int or None, default None
        Number of resamples drawn per batch; see :func:`bootstrap`.

    Returns
    -------
    BootstrapResult
        The difference of the two ratios on the original data, its bootstrap
        standard error, the confidence interval, and the bootstrap
        distribution.

    Raises
    ------
    ValueError
        If a group has fewer than two users, a numerator and its denominator
        differ in length, ``alpha`` is not in (0, 1), ``n_resamples`` is below
        2, ``method`` is unknown, ``batch_size`` is below 1, or a denominator
        sums to zero on the original data or on any resample.
    """
    n_c = control_numerator.size
    n_t = treatment_numerator.size

    if n_c <= 1 or n_t <= 1:
        raise ValueError('in control and treatment groups there should be at least two observations')
    if n_c != control_denominator.size or n_t != treatment_denominator.size:
        raise ValueError('Control and treatment numerator should be the same size as denominator')
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be between 0 and 1')
    if control_denominator.sum() == 0 or treatment_denominator.sum() == 0:
        raise ValueError('control and treatment denominator should not be zero')
    if method not in ["percentile", "normal"]:
        raise ValueError('method must be "percentile" or "normal"')
    if n_resamples <= 1:
        raise ValueError('n_resamples must be greater than 1')

    rng = np.random.default_rng(seed)

    if batch_size is None:
        b = max(1, int((250*10**6) / ((n_c+n_t)*64)))
    else:
        b = batch_size

    distribution = np.empty(n_resamples)
    for start in range(0, n_resamples, b):
        stop = min(start + b, n_resamples)

        idx_c = rng.integers(0, n_c, size=(stop - start, n_c))
        idx_t = rng.integers(0, n_t, size=(stop - start, n_t))

        if np.any(control_denominator[idx_c].sum(axis=1) == 0) or np.any(treatment_denominator[idx_t].sum(axis=1) == 0):
            raise ValueError('control and treatment denominator should not be zero')

        samples_c = control_numerator[idx_c].sum(axis=1) / control_denominator[idx_c].sum(axis=1)
        samples_t = treatment_numerator[idx_t].sum(axis=1) / treatment_denominator[idx_t].sum(axis=1)

        distribution[start:stop] = samples_t - samples_c

    standard_error = distribution.std(ddof=1)
    metric_c = control_numerator.sum() / control_denominator.sum()
    metric_t = treatment_numerator.sum() / treatment_denominator.sum()

    estimate = metric_t - metric_c

    if method == "percentile":
        ci_low = np.quantile(distribution, alpha/2)
        ci_high = np.quantile(distribution, 1-alpha/2)
    if method == "normal":
        z_stat = NormalDist().inv_cdf(1 - alpha / 2)
        ci_low = estimate - z_stat * standard_error
        ci_high = estimate + z_stat * standard_error

    return BootstrapResult(estimate=estimate, standard_error=standard_error,
                           ci_low=ci_low, ci_high=ci_high, alpha=alpha, method=method,
                           distribution=distribution)


