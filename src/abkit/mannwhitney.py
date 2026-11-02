from dataclasses import dataclass
from scipy.stats import rankdata, norm
import numpy as np


@dataclass(frozen=True)
class MannWhitneyResult:
    """Result of a Mann-Whitney U test between two groups.

    Attributes
    ----------
    u : float
        The U statistic for the treatment group: the number of
        (treatment, control) pairs in which the treatment value is larger,
        counting ties as one half. Ranges from 0 to ``n_control * n_treatment``.
    prob_superiority : float
        ``u / (n_control * n_treatment)``, the estimated probability that a
        randomly chosen treatment user has a larger value than a randomly
        chosen control user, ties counting as one half. Equals 0.5 when the
        two groups have the same distribution.
    statistic : float
        Signed, continuity-corrected z statistic: ``u`` minus its mean under
        the null, moved half a step toward that mean, divided by the
        tie-corrected standard deviation. Positive when treatment values tend
        to be larger; swapping the groups changes its sign.
    p_value : float
        Two-sided p-value from the normal approximation,
        ``2 * (1 - Phi(|statistic|))``.
    """

    u: int
    prob_superiority: float
    statistic: float
    p_value: float


def mann_whitney(control:np.ndarray, treatment:np.ndarray) -> MannWhitneyResult:
    """Mann-Whitney U test of whether two groups come from the same distribution.

    Both groups are ranked together, with tied values sharing their average
    rank. ``U`` is the treatment rank sum minus ``n_treatment * (n_treatment +
    1) / 2``, which equals the number of treatment-control pairs the treatment
    value wins, ties counting as one half. The p-value uses the normal
    approximation with the tie-corrected variance and a continuity correction
    of one half, as SciPy's ``method="asymptotic"`` does.

    Parameters
    ----------
    control, treatment : np.ndarray
        One-dimensional arrays of per-user values, one per group. Each must
        contain at least one value.

    Returns
    -------
    MannWhitneyResult
        ``u``, the probability of superiority ``u / (n_control *
        n_treatment)``, the signed z statistic and the two-sided p-value.

    Raises
    ------
    ValueError
        If either group is empty, or all values in both groups are identical,
        so that the variance of ``U`` is zero and no test is possible.

    Notes
    -----
    The null hypothesis is that the two groups have the same distribution. The
    test is then valid for any shape of distribution and any split between the
    groups. It is not, in general, a valid test of ``prob_superiority = 0.5``
    when the two groups differ in spread; use ``scipy.stats.brunnermunzel``
    for that question.
    """

    n_control = control.size
    n_treatment = treatment.size
    n = n_control + n_treatment

    if n_control <= 0 or n_treatment <= 0:
        raise ValueError('in control and treatment groups there should be at least one observations')

    pooled = np.concatenate((control, treatment))
    R_t = rankdata(pooled)[n_control:].sum()
    u = R_t - n_treatment * (n_treatment + 1) / 2
    prob_superiority = u / (n_control * n_treatment)

    counts = np.unique(pooled, return_counts=True)[1].astype(float)
    tie_term = (counts ** 3 - counts).sum()

    var_u = (n_control * n_treatment / 12) * ((n + 1) - tie_term / (n * (n - 1)))
    sd = np.sqrt(var_u)

    if sd == 0:
        raise ValueError('standard deviation cannot be zero')
    diff = u - n_control * n_treatment / 2  # positive when treatment tends to be higher
    corrected = np.sign(diff) * max(abs(diff) - 0.5, 0)  # step 0.5 toward the mean,

    z_stat = corrected / sd
    p_value = 2 * norm.sf(abs(z_stat))

    return MannWhitneyResult(u=u, prob_superiority=prob_superiority, statistic=z_stat, p_value=p_value)
