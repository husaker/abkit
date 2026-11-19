"""The sample ratio check: a chi-square test of group sizes, and a scorecard that runs it first."""

from dataclasses import dataclass
import scipy.stats
import numpy as np
from abkit.errors import SmallExpectedCountWarning, SampleRatioMismatchError, SampleRatioMismatchWarning
import warnings
from abkit.protocols import SupportsPValue, TwoSampleTest
from typing import Mapping
import numbers


@dataclass(frozen=True)
class SRMResult:
    """Result of a chi-square test of observed group sizes against the designed shares.

    srm_test builds it and check_srm returns it. Every field holds plain Python numbers, in
    tuples where there is one value per group, so two results compare with ==. Because it has a
    read-only p_value, it satisfies the SupportsPValue protocol without declaring it.

    Attributes
    ----------
    counts : tuple of int
        Observed number of users in each group, in the order given.
    expected : tuple of float
        Number of users each group should have under the design: the total times its share.
    statistic : float
        Pearson's chi-square statistic, the sum over the groups of
        (observed - expected) ** 2 / expected.
    df : int
        Degrees of freedom: the number of groups minus one.
    p_value : float
        Upper-tail probability of the chi-square distribution with df degrees of freedom: how
        often an assignment that works as designed gives a statistic at least this large.
    """

    counts: tuple[int, ...]
    expected: tuple[float, ...]
    statistic: float
    df: int
    p_value: float


def srm_test(counts: tuple[int, ...], shares:tuple[float, ...]| None=None) -> SRMResult:
    """Test whether group sizes match the designed shares with a chi-square goodness-of-fit test.

    Each group's expected count is the total number of users times the group's share. The
    statistic is OpenIntro's X² (section 6.3), referred to the chi-square distribution with one
    degree of freedom fewer than the number of groups. For two groups it is the square of the
    z-statistic of chapter 1's test of a share, and the p-values agree.

    Parameters
    ----------
    counts : sequence of int
        Number of users in each group; at least two groups. Python or NumPy integers only:
        floats, even whole ones, and booleans are rejected, so that shares passed by mistake
        cannot pass as counts. A group may have zero users, but not all of them.
    shares : sequence of float, optional
        Designed share of each group, one per group, each positive and finite, summing to 1.
        None, the default, means equal shares.

    Returns
    -------
    SRMResult
        The counts, the expected counts, the statistic, its degrees of freedom and the p-value.

    Raises
    ------
    ValueError
        If counts is not one-dimensional, has fewer than two groups, holds anything but
        non-negative integers, or totals zero; or if shares has the wrong length, holds a share
        that is not positive and finite, or does not sum to 1.

    Warns
    -----
    SmallExpectedCountWarning
        If any expected count is below 5, where the chi-square approximation is unreliable
        (OpenIntro, p. 235). The result is still returned.
    """

    observed = np.array(counts)
    num_groups = len(observed)
    total_count = sum(observed)

    if shares is None:
        probabilities = np.full(num_groups, 1 / num_groups)
    else:
        probabilities = np.asarray(shares, dtype=float)

    if observed.ndim != 1 or len(observed) < 2:
        raise ValueError("counts must hold one number of users for each of at least two groups")
    if len(observed) == 1:
        raise ValueError("there should be at least two groups")
    if len(observed) != len(probabilities):
        raise ValueError("wrong length of shares")
    if any(isinstance(c, float) for c in observed) or any(isinstance(c, bool) for c in observed):
        raise ValueError("counts should be type of integer")
    if any(c < 0 for c in observed):
        raise ValueError("each of counts should be positive integer")
    if sum(observed) <= 0:
        raise ValueError("total_count should be above 0")
    if ~np.isclose(sum(probabilities), 1.0):
        raise ValueError("shares should sum to 1")
    if any(c < 0 for c in probabilities):
        raise ValueError("shares should be positive")
    if any(c >= 1 for c in probabilities) or any(c <= 0 for c in observed):
        raise ValueError("shares should be between 0 and 1")
    if any(c == float("nan") for c in probabilities):
        raise ValueError("shares can't be nans")

    expected_counts = tuple(float(s * total_count) for s in probabilities)


    if any(c < 5 for c in expected_counts):
        warnings.warn(
            "one of the expected counts is less than 5",
            SmallExpectedCountWarning,
            stacklevel=2
        )

    df = num_groups - 1
    chi_square_statistic = 0.0
    for count, expected_count in zip(observed, expected_counts):
        chi_square_statistic += (count - expected_count)**2 / expected_count

    p_value = scipy.stats.chi2.sf(chi_square_statistic, df)

    return SRMResult(counts=tuple(int(c) for c in observed), expected=tuple(float(e) for e in expected_counts),
                     statistic=float(chi_square_statistic), df=df, p_value=float(p_value))

def check_srm(counts: tuple[int, ...], shares: tuple[float, ...]| None=None,
              *, threshold: float = 0.001, on_srm: str="raise") -> SRMResult:
    """Run srm_test and act on a mismatch: raise SampleRatioMismatchError, or warn and go on.

    A p-value strictly below the threshold is a sample ratio mismatch. The default follows
    Kohavi, Tang and Xu (p. 45): the check runs on every experiment, and a strict threshold keeps
    false alarms rare enough for an alarm to be believed.

    Parameters
    ----------
    counts : sequence of int
        Number of users in each group, as for srm_test.
    shares : sequence of float, optional
        Designed share of each group, as for srm_test; None means equal shares.
    threshold : float, default 0.001
        A p-value below it is a mismatch. Must be strictly between 0 and 1.
    on_srm : {"raise", "warn"}, default "raise"
        What to do with a mismatch: "raise" stops the caller; "warn" issues a warning and
        returns the result, for looking at the metrics while debugging the cause.

    Returns
    -------
    SRMResult
        The test result, when there is no mismatch or when on_srm is "warn".

    Raises
    ------
    SampleRatioMismatchError
        If the p-value is below the threshold and on_srm is "raise". The message names the
        counts and the p-value.
    ValueError
        If threshold is not a number strictly between 0 and 1, if on_srm is neither "raise" nor
        "warn", or for any input that srm_test rejects. The arguments are checked before
        anything is computed.

    Warns
    -----
    SampleRatioMismatchWarning
        If the p-value is below the threshold and on_srm is "warn".
    SmallExpectedCountWarning
        Passed on from srm_test when an expected count is below 5.
    """
    if not isinstance(threshold, numbers.Real) or not 0 < threshold < 1:
        raise ValueError("threshold must be a number between 0 and 1")
    if on_srm not in ("raise", "warn"):
        raise ValueError("on_srm should be either 'raise' or 'warn'")

    result = srm_test(counts, shares)

    if result.p_value < threshold:
        if on_srm == "raise":
            raise SampleRatioMismatchError(f'{counts} SRM with p_value: {result.p_value}')
        elif on_srm == "warn":
            warnings.warn(
                "{counts} SRM with p_value: {result.p_value}",
                SampleRatioMismatchWarning,
                stacklevel=2
            )
            return result
    else:
        return result

@dataclass(frozen=True)
class Scorecard:
    """An experiment's metric results, released only after its sample ratio check.

    scorecard builds it. The data class is frozen, but results is an ordinary dict: treat it as
    read-only.

    Attributes
    ----------
    srm : SRMResult
        The sample ratio check: one the experiment passed, or one that failed while on_srm was
        "warn".
    results : dict of str to SupportsPValue or float
        Each metric's name mapped to whatever its test returned, such as a ComparisonResult.
    """

    srm: SRMResult
    results: dict[str, SupportsPValue | float]

def scorecard(treated: np.ndarray, metrics:Mapping[str, tuple[np.ndarray, TwoSampleTest]],
              *, expected_share:float =0.5,threshold: float = 0.001,
              on_srm: str="raise") -> Scorecard:
    """Check the sample ratio first, then compare every metric between control and treatment.

    Every input is validated before the check, and no metric's test is called until the check
    has passed, or has warned with on_srm="warn". Each metric is split into groups with the same
    treated array that the check counted, so every comparison uses the assignment it verified.

    Parameters
    ----------
    treated : np.ndarray
        One-dimensional boolean array with one entry per user, True for treatment users.
    metrics : mapping of str to (np.ndarray, TwoSampleTest)
        Each metric's name mapped to a pair: its values, one per user in the order of treated,
        and the test that compares the groups, such as welch_t_test or z_test_proportions. Pass
        the test itself, not its result; it is called as test(control, treatment).
    expected_share : float, default 0.5
        Designed share of users in treatment, strictly between 0 and 1.
    threshold : float, default 0.001
        Passed to check_srm.
    on_srm : {"raise", "warn"}, default "raise"
        Passed to check_srm.

    Returns
    -------
    Scorecard
        The check's result and every metric's test result.

    Raises
    ------
    ValueError
        If treated is not a one-dimensional boolean array, if expected_share is not strictly
        between 0 and 1, or if a metric does not have exactly one value per user; all of these
        are checked before the sample ratio.
    SampleRatioMismatchError
        If the check finds a mismatch and on_srm is "raise"; no metric is computed then.

    Warns
    -----
    SampleRatioMismatchWarning
        If the check finds a mismatch and on_srm is "warn"; the metrics are computed anyway.
    """
    if treated.ndim > 1:
        raise ValueError("treated array should be one dimensional")
    if treated.dtype != np.bool:
        raise ValueError("treated array should be bool")
    if not isinstance(expected_share, numbers.Real) or not 0 < expected_share < 1:
        raise ValueError("expected_share must be a number between 0 and 1")
    for name, (values, test) in metrics.items():
        if np.asarray(values).ndim != 1 or len(values) != len(treated):
            raise ValueError(f'metric {name} must have one value per user')

    counts = (int((~treated).sum()), int(treated.sum()))
    shares = (1 - expected_share, expected_share)

    srm = check_srm(counts=counts, shares=shares,
                    threshold=threshold, on_srm=on_srm)

    results = dict()

    for name, (values, test) in metrics.items():
        p_value = test(values[~treated], values[treated])
        results[name] = p_value

    return Scorecard(srm=srm, results=results)