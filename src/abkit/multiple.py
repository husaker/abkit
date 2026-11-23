"""Corrections for testing many hypotheses at once (ISLP, chapter 13).

Bonferroni and Holm control the family-wise error rate, the probability of at least one false
rejection; Benjamini-Hochberg controls the false discovery rate, the expected share of false
rejections among all rejections. Each function turns a family of p-values into adjusted p-values,
one per hypothesis, and rejects every hypothesis whose adjusted p-value is at most the level
(ISLP, lab 13.6.2, p. 586).
"""

from dataclasses import dataclass

import numpy as np
import numbers

@dataclass(frozen=True)
class CorrectionResult:
    """Adjusted p-values and decisions for a family of hypotheses, in the order they were given.

    bonferroni, holm and benjamini_hochberg build it. Every field holds plain Python values, in
    tuples where there is one value per hypothesis, so two results compare with ==.

    Attributes
    ----------
    method : str
        The correction that produced it: "bonferroni", "holm" or "benjamini_hochberg".
    level : float
        The level being controlled: the family-wise error rate alpha for Bonferroni and Holm,
        the false discovery rate q for Benjamini-Hochberg.
    p_values : tuple of float
        The p-values as given.
    adjusted : tuple of float
        One adjusted p-value per hypothesis: the smallest level at which the correction rejects
        it, capped at 1. It does not depend on the level used, so it can be compared with any
        other level.
    reject : tuple of bool
        True where the adjusted p-value is at most the level.
    """

    method: str
    level: float
    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]


def bonferroni(p_values: tuple[float, ...], *, alpha: float=0.05) -> CorrectionResult:
    """Adjust p-values with the Bonferroni correction, which controls the family-wise error rate.

    Each p-value is compared with alpha / m, where m is the number of p-values, so the probability
    of at least one false rejection is at most alpha whether or not the tests are independent
    (ISLP, (13.6), pp. 567-568). The adjusted p-value is m * p, capped at 1 (lab, p. 586). The
    correction is conservative: holm rejects at least as many hypotheses at the same level.

    Parameters
    ----------
    p_values : sequence of float or one-dimensional np.ndarray
        One p-value per hypothesis, at least one, each between 0 and 1 inclusive.
    alpha : float, default 0.05
        The family-wise error rate to control, strictly between 0 and 1. Keyword-only.

    Returns
    -------
    CorrectionResult
        Method "bonferroni", the level, the p-values, the adjusted p-values and the decisions,
        in the order the p-values were given.

    Raises
    ------
    ValueError
        If alpha is not a real number strictly between 0 and 1, or if p_values is empty, is not
        one-dimensional, or holds nan or a value outside [0, 1].
    """

    if not isinstance(alpha, numbers.Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number between 0 and 1")

    p_vals = np.asarray(p_values, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        raise ValueError("p_values must be a 1D array")
    if not np.all((p_vals >= 0) & (p_vals <= 1)):
        raise ValueError("p_values must be real numbers without nans")

    num_p_vals = len(p_vals)

    adjusted_p_vals = p_vals * num_p_vals
    capped = np.minimum(adjusted_p_vals, 1.0)

    reject = adjusted_p_vals <= alpha

    return CorrectionResult(method="bonferroni", level=float(alpha), p_values=tuple(p_vals.tolist()),
                            adjusted=tuple(capped.tolist()), reject=tuple(reject.tolist()))


def holm(p_values: tuple[float, ...], *, alpha: float=0.05) -> CorrectionResult:
    """Adjust p-values with Holm's step-down procedure, which controls the family-wise error rate.

    The p-values are sorted, and the one at rank j (1 for the smallest) is compared with
    alpha / (m + 1 - j); every hypothesis before the first p-value that fails is rejected (ISLP,
    Algorithm 13.1, p. 569). Like Bonferroni it assumes nothing about independence, and it rejects
    at least as many hypotheses. The adjusted p-value at rank j is the largest of
    (m + 1 - i) * p_(i) over the ranks i up to j, capped at 1: the smallest alpha at which the walk
    gets past rank j. Equal p-values get equal adjusted p-values.

    Parameters
    ----------
    p_values : sequence of float or one-dimensional np.ndarray
        One p-value per hypothesis, at least one, each between 0 and 1 inclusive.
    alpha : float, default 0.05
        The family-wise error rate to control, strictly between 0 and 1. Keyword-only.

    Returns
    -------
    CorrectionResult
        Method "holm", the level, the p-values, the adjusted p-values and the decisions, in the
        order the p-values were given.

    Raises
    ------
    ValueError
        If alpha is not a real number strictly between 0 and 1, or if p_values is empty, is not
        one-dimensional, or holds nan or a value outside [0, 1].
    """

    if not isinstance(alpha, numbers.Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number between 0 and 1")

    p_vals = np.asarray(p_values, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        raise ValueError("p_values must be a 1D array")
    if not np.all((p_vals >= 0) & (p_vals <= 1)):
        raise ValueError("p_values must be real numbers without nans")

    num_p_vals = len(p_vals)

    order = np.argsort(p_vals)
    sorted_p_vals = p_vals[order]

    running_max = 0.0
    by_rank = []

    for j, p in enumerate(sorted_p_vals, start=1):
        running_max = max(running_max, (num_p_vals + 1 - j) * p)
        by_rank.append(min(running_max, 1.0))

    adjusted = np.empty(num_p_vals)

    adjusted[order] = by_rank
    reject = adjusted <= alpha

    return CorrectionResult(method="holm", level=float(alpha), p_values=tuple(p_vals.tolist()),
                            adjusted=tuple(adjusted.tolist()), reject=tuple(reject.tolist()))


def benjamini_hochberg(p_values: tuple[float, ...], *, q: float=0.05) -> CorrectionResult:
    """Adjust p-values with the Benjamini-Hochberg procedure, which controls the false discovery rate.

    The p-values are sorted, the procedure finds the largest rank j at which p_(j) <= q * j / m,
    and every hypothesis with a p-value up to p_(j) is rejected (ISLP, Algorithm 13.2, p. 575), so
    a p-value above its own line is still rejected when a larger one passes. When the p-values are
    independent or only mildly dependent, the expected share of false rejections among all
    rejections is at most q (p. 575). The adjusted p-value at rank j, which ISLP's lab calls a
    q-value (p. 589), is the smallest of m * p_(k) / k over the ranks k from j to m: the smallest q
    at which the hypothesis is rejected. It never exceeds 1.

    Parameters
    ----------
    p_values : sequence of float or one-dimensional np.ndarray
        One p-value per hypothesis, at least one, each between 0 and 1 inclusive.
    q : float, default 0.05
        The false discovery rate to control, strictly between 0 and 1. Keyword-only. There is no
        standard choice: a lower q suits a follow-up that is expensive (ISLP, pp. 574-575).

    Returns
    -------
    CorrectionResult
        Method "benjamini_hochberg", the level, the p-values, the adjusted p-values and the
        decisions, in the order the p-values were given.

    Raises
    ------
    ValueError
        If q is not a real number strictly between 0 and 1, or if p_values is empty, is not
        one-dimensional, or holds nan or a value outside [0, 1].
    """

    if not isinstance(q, numbers.Real) or not 0 < q < 1:
        raise ValueError("q must be a number between 0 and 1")

    p_vals = np.asarray(p_values, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        raise ValueError("p_values must be a 1D array")
    if not np.all((p_vals >= 0) & (p_vals <= 1)):
        raise ValueError("p_values must be real numbers without nans")

    num_p_vals = len(p_vals)

    order = np.argsort(p_vals)
    sorted_p_vals = p_vals[order]

    running_min = 1.0
    by_rank = [0.0] * num_p_vals

    for j in range(num_p_vals, 0, -1):
        running_min = min(running_min, num_p_vals * sorted_p_vals[j - 1] / j)
        by_rank[j - 1] = running_min

    adjusted = np.empty(num_p_vals)

    adjusted[order] = by_rank
    reject = adjusted <= q

    return CorrectionResult(method="benjamini_hochberg", level=float(q), p_values=tuple(p_vals.tolist()),
                            adjusted=tuple(adjusted.tolist()), reject=tuple(reject.tolist()))
