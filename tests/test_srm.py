"""Chapter 8: abkit's own exceptions and warnings, the chi-square sample ratio check, and the scorecard.

The oracles are scipy.stats.chisquare, OpenIntro's juror example (section 6.3, pp. 229-235) and
Kohavi, Tang and Xu's Scenario 1 (p. 220).
"""

import dataclasses
import math
import warnings
from statistics import NormalDist

import numpy as np
import pytest
from scipy import stats

from abkit.errors import (
    AbkitError,
    AbkitWarning,
    SampleRatioMismatchError,
    SampleRatioMismatchWarning,
    SmallExpectedCountWarning,
)
from abkit.protocols import SupportsPValue
from abkit.srm import Scorecard, SRMResult, check_srm, scorecard, srm_test
from abkit.ttest import welch_t_test
from abkit.ztest import z_test_proportions


# ---------------------------------------------------------------- the family tree of errors and warnings

def test_abkit_error_is_an_exception():
    assert issubclass(AbkitError, Exception)


def test_srm_error_belongs_to_the_abkit_family():
    assert issubclass(SampleRatioMismatchError, AbkitError)


def test_srm_error_is_not_a_value_error():
    # a design decision: an `except ValueError` written for bad arguments must not swallow an SRM
    assert not issubclass(SampleRatioMismatchError, ValueError)


def test_except_abkit_error_catches_an_srm_error():
    try:
        raise SampleRatioMismatchError("counts (600, 400)")
    except AbkitError as error:
        caught = error
    assert isinstance(caught, SampleRatioMismatchError)
    assert "600" in str(caught)


def test_warnings_have_their_own_family():
    assert issubclass(AbkitWarning, UserWarning)
    assert issubclass(SampleRatioMismatchWarning, AbkitWarning)
    assert issubclass(SmallExpectedCountWarning, AbkitWarning)


def test_the_two_families_are_separate():
    assert not issubclass(AbkitWarning, AbkitError)
    assert not issubclass(SampleRatioMismatchError, Warning)
    assert not issubclass(SampleRatioMismatchWarning, SmallExpectedCountWarning)
    assert not issubclass(SmallExpectedCountWarning, SampleRatioMismatchWarning)


# ---------------------------------------------------------------- srm_test: the statistic and its p-value

def test_equal_counts_give_a_zero_statistic():
    result = srm_test([500, 500])
    assert result.statistic == 0
    assert result.p_value == pytest.approx(1.0)
    assert result.df == 1
    assert result.expected == (500.0, 500.0)


@pytest.mark.parametrize("counts, shares", [
    ([50_400, 49_600], None),
    ([821_588, 815_482], None),
    ([120, 80, 100], None),
    ([8_000, 1_100, 900], [0.8, 0.1, 0.1]),
    ([205, 26, 25, 19], [0.72, 0.07, 0.12, 0.09]),
])
def test_matches_scipy_chisquare(counts, shares):
    observed = np.array(counts)
    expected = observed.sum() * (np.array(shares) if shares else np.full(len(counts), 1 / len(counts)))
    oracle = stats.chisquare(observed, expected)
    result = srm_test(counts, shares)
    assert result.statistic == pytest.approx(oracle.statistic, rel=1e-9)
    assert result.p_value == pytest.approx(oracle.pvalue, rel=1e-9, abs=1e-300)


def test_openintro_jurors():
    result = srm_test([205, 26, 25, 19], [0.72, 0.07, 0.12, 0.09])
    assert result.expected == pytest.approx((198, 19.25, 33, 24.75))
    assert result.statistic == pytest.approx(5.89, abs=0.005)
    assert result.df == 3
    assert result.p_value == pytest.approx(0.1171, abs=0.0005)


def test_kohavi_scenario_1():
    result = srm_test([821_588, 815_482])
    assert result.p_value == pytest.approx(1.8e-6, rel=0.05)


def test_two_groups_statistic_is_z_squared():
    n, treatment = 100_000, 49_600
    z = (treatment - n / 2) / math.sqrt(n * 0.5 * 0.5)
    result = srm_test([n - treatment, treatment])
    assert result.statistic == pytest.approx(z ** 2)
    assert result.statistic == pytest.approx(6.4)


def test_two_groups_p_value_is_the_two_sided_normal_tail():
    n, treatment = 100_000, 49_600
    z = (treatment - n / 2) / math.sqrt(n * 0.25)
    result = srm_test([n - treatment, treatment])
    assert result.p_value == pytest.approx(2 * NormalDist().cdf(-abs(z)), rel=1e-6)


def test_unequal_design_with_matching_counts():
    result = srm_test([900, 100], [0.9, 0.1])
    assert result.statistic == pytest.approx(0)
    assert result.expected == pytest.approx((900, 100))


def test_default_shares_are_equal():
    assert srm_test([30, 30, 40]) == srm_test([30, 30, 40], [1 / 3, 1 / 3, 1 / 3])


def test_df_is_number_of_groups_minus_one():
    assert srm_test([10, 10, 10, 10]).df == 3
    assert srm_test([10, 10, 10]).df == 2


def test_expected_counts_follow_the_shares():
    result = srm_test([7_000, 3_000], [0.6, 0.4])
    assert result.expected == pytest.approx((6_000, 4_000))
    assert result.statistic == pytest.approx(1_000 ** 2 / 6_000 + 1_000 ** 2 / 4_000)


def test_counts_and_expected_are_tuples_of_plain_numbers():
    result = srm_test(np.array([900, 100], dtype=np.int32), [0.9, 0.1])
    assert result.counts == (900, 100)
    assert isinstance(result.counts, tuple) and all(type(c) is int for c in result.counts)
    assert isinstance(result.expected, tuple) and all(type(e) is float for e in result.expected)
    assert type(result.statistic) is float and type(result.p_value) is float and type(result.df) is int


def test_result_is_frozen():
    result = srm_test([500, 480])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.5


def test_result_is_a_supports_p_value():
    assert isinstance(srm_test([500, 480]), SupportsPValue)


def test_input_is_not_modified():
    counts = np.array([520, 480])
    srm_test(counts)
    assert counts.tolist() == [520, 480]


@pytest.mark.parametrize("counts", [
    [100],                  # one group
    [[100, 90], [80, 70]],  # two-dimensional
    [0.49, 0.51],           # shares passed where counts belong
    [100.0, 98.0],          # floats, even whole ones
    [True, False],          # booleans
    [-1, 5],                # negative
    [0, 0],                 # nobody at all
])
def test_rejects_bad_counts(counts):
    with pytest.raises(ValueError):
        srm_test(counts)


def test_shares_instead_of_counts_message_mentions_integers():
    with pytest.raises(ValueError, match="integer"):
        srm_test([0.49, 0.51])


@pytest.mark.parametrize("shares", [
    [0.5, 0.3, 0.2],          # wrong length
    [1.0, 0.0],               # a zero share
    [1.2, -0.2],              # negative
    [float("nan"), 0.5],      # not a number
    [0.45, 0.45],             # does not sum to 1
])
def test_rejects_bad_shares(shares):
    with pytest.raises(ValueError, match="shares"):
        srm_test([100, 100], shares)


def test_small_expected_count_warns_and_still_returns():
    with pytest.warns(SmallExpectedCountWarning):
        result = srm_test([3, 1])
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value == pytest.approx(stats.chi2.sf(1.0, 1))


def test_small_count_warning_points_at_the_caller():
    with pytest.warns(SmallExpectedCountWarning) as record:
        srm_test([3, 1])
    assert record[0].filename == __file__


def test_no_warning_when_every_expected_count_is_at_least_five():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        srm_test([5, 5])
        srm_test([520, 480])


# ---------------------------------------------------------------- check_srm: the policy

def test_check_passes_and_returns_the_test_result():
    assert check_srm([50_000, 50_100]) == srm_test([50_000, 50_100])


def test_check_raises_below_the_threshold():
    with pytest.raises(SampleRatioMismatchError):
        check_srm([50_400, 49_600], threshold=0.05)  # p = 0.0114


def test_default_threshold_is_one_in_a_thousand():
    result = check_srm([50_400, 49_600])           # p = 0.0114 passes 0.001
    assert result.p_value == pytest.approx(0.0114, abs=1e-4)
    with pytest.raises(SampleRatioMismatchError):
        check_srm([821_588, 815_482])               # p = 1.8e-6


def test_the_message_names_the_counts():
    with pytest.raises(SampleRatioMismatchError) as caught:
        check_srm([821_588, 815_482])
    message = str(caught.value)
    assert "821588" in message and "815482" in message


def test_a_p_value_equal_to_the_threshold_is_not_a_mismatch():
    p = srm_test([50_400, 49_600]).p_value
    check_srm([50_400, 49_600], threshold=p)  # strictly below is a mismatch; equal is not
    with pytest.raises(SampleRatioMismatchError):
        check_srm([50_400, 49_600], threshold=math.nextafter(p, 1))


def test_check_uses_the_shares():
    check_srm([8_000, 1_000, 1_000], [0.8, 0.1, 0.1])
    with pytest.raises(SampleRatioMismatchError):
        check_srm([8_000, 1_100, 900], [0.8, 0.1, 0.1])  # chi-square 20 on 2 df, p = 4.5e-5


def test_warn_mode_warns_and_returns_the_result():
    with pytest.warns(SampleRatioMismatchWarning):
        result = check_srm([821_588, 815_482], on_srm="warn")
    assert result.p_value == pytest.approx(1.8e-6, rel=0.05)


def test_warn_mode_warning_points_at_the_caller():
    with pytest.warns(SampleRatioMismatchWarning) as record:
        check_srm([821_588, 815_482], on_srm="warn")
    assert record[0].filename == __file__


def test_warn_mode_is_silent_when_the_ratio_is_fine():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_srm([50_000, 50_100], on_srm="warn")


def test_a_warning_filter_can_turn_the_warning_into_an_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SampleRatioMismatchWarning)
        with pytest.raises(SampleRatioMismatchWarning):
            check_srm([821_588, 815_482], on_srm="warn")


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.5, float("nan"), "0.001", None])
def test_rejects_bad_thresholds(threshold):
    with pytest.raises(ValueError):
        check_srm([500, 500], threshold=threshold)


@pytest.mark.parametrize("on_srm", ["ignore", "RAISE", "", None])
def test_rejects_bad_on_srm(on_srm):
    with pytest.raises(ValueError):
        check_srm([500, 500], on_srm=on_srm)


def test_bad_arguments_are_value_errors_not_srm_errors():
    with pytest.raises(ValueError):
        check_srm([821_588, 815_482], threshold=2)


# ---------------------------------------------------------------- scorecard: the check guards every metric

def exact_split(n, n_treated, seed=0):
    treated = np.zeros(n, dtype=bool)
    treated[np.random.default_rng(seed).permutation(n)[:n_treated]] = True
    return treated


class RecordingTest:
    """A fake TwoSampleTest that remembers its calls and returns a p-value."""

    def __init__(self):
        self.calls = []

    def __call__(self, control, treatment, /):
        self.calls.append((control.copy(), treatment.copy()))
        return 0.5


def test_scorecard_runs_every_metric_when_the_ratio_is_fine():
    rng = np.random.default_rng(1)
    treated = exact_split(1_000, 500)
    revenue = rng.exponential(3.0, 1_000)
    converted = (rng.random(1_000) < 0.1).astype(float)
    card = scorecard(treated, {"revenue": (revenue, welch_t_test), "converted": (converted, z_test_proportions)})
    assert isinstance(card, Scorecard)
    assert set(card.results) == {"revenue", "converted"}
    assert card.results["revenue"] == welch_t_test(revenue[~treated], revenue[treated])
    assert card.results["converted"] == z_test_proportions(converted[~treated], converted[treated])


def test_scorecard_records_the_check():
    treated = exact_split(1_000, 480)
    card = scorecard(treated, {})
    assert card.srm.counts == (520, 480)
    assert card.srm == srm_test([520, 480])
    assert card.results == {}


def test_scorecard_passes_control_first_then_treatment():
    treated = np.array([False, True] * 6)
    values = np.arange(12.0)
    fake = RecordingTest()
    scorecard(treated, {"m": (values, fake)})
    control, treatment = fake.calls[0]
    assert control.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert treatment.tolist() == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def test_scorecard_computes_no_metric_when_the_check_fails():
    treated = exact_split(1_000, 600)  # p = 2.5e-10 against a 50/50 design
    fake = RecordingTest()
    with pytest.raises(SampleRatioMismatchError):
        scorecard(treated, {"revenue": (np.ones(1_000), fake)})
    assert fake.calls == []


def test_scorecard_warn_mode_returns_the_metrics_with_a_warning():
    treated = exact_split(1_000, 600)
    fake = RecordingTest()
    with pytest.warns(SampleRatioMismatchWarning):
        card = scorecard(treated, {"revenue": (np.ones(1_000), fake)}, on_srm="warn")
    assert card.results == {"revenue": 0.5}
    assert len(fake.calls) == 1


def test_scorecard_uses_the_expected_share():
    treated = exact_split(1_000, 300)
    scorecard(treated, {}, expected_share=0.3)
    with pytest.raises(SampleRatioMismatchError):
        scorecard(treated, {})


def test_scorecard_passes_the_threshold_on():
    treated = exact_split(10_000, 4_880)  # p = 0.016
    scorecard(treated, {})
    with pytest.raises(SampleRatioMismatchError):
        scorecard(treated, {}, threshold=0.05)


@pytest.mark.parametrize("treated", [
    np.array([0, 1, 0, 1]),                    # integers, not booleans
    np.array([[True, False], [False, True]]),  # two-dimensional
])
def test_scorecard_rejects_bad_treated(treated):
    with pytest.raises(ValueError):
        scorecard(treated, {})


@pytest.mark.parametrize("share", [0, 1, -0.5, float("nan")])
def test_scorecard_rejects_bad_expected_share(share):
    with pytest.raises(ValueError, match="expected_share"):
        scorecard(exact_split(100, 50), {}, expected_share=share)


def test_scorecard_rejects_a_metric_of_the_wrong_length_and_names_it():
    with pytest.raises(ValueError, match="revenue"):
        scorecard(exact_split(100, 50), {"revenue": (np.ones(99), welch_t_test)})


def test_scorecard_validates_its_inputs_before_it_checks_the_ratio():
    treated = exact_split(1_000, 600)  # would fail the check
    with pytest.raises(ValueError):
        scorecard(treated, {"revenue": (np.ones(999), welch_t_test)})
