"""Chapter 9: Bonferroni, Holm and Benjamini-Hochberg corrections for multiple testing.

The oracles are ISLP's five fund managers (Table 13.3, p. 567; worked on pp. 568, 569 and 575;
lab 13.6.2, p. 586) and statsmodels' multipletests, which the ISLP lab uses. The last tests
simulate many families of p-values and count the errors, as ISLP's lab does in section 13.6.1.
"""

import dataclasses
import math
from statistics import NormalDist

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from abkit.multiple import CorrectionResult, benjamini_hochberg, bonferroni, holm

# The p-values of ISLP's first five fund managers, as printed in the lab (p. 586).
FUND = [0.006, 0.918, 0.012, 0.601, 0.756]

METHODS = [bonferroni, holm, benjamini_hochberg]
NAMES = {bonferroni: "bonferroni", holm: "holm", benjamini_hochberg: "benjamini_hochberg"}
LEVEL_ARGUMENT = {bonferroni: "alpha", holm: "alpha", benjamini_hochberg: "q"}
STATSMODELS_METHOD = {bonferroni: "bonferroni", holm: "holm", benjamini_hochberg: "fdr_bh"}
by_method = pytest.mark.parametrize("correct", METHODS, ids=lambda correct: correct.__name__)


def at_level(correct, p_values, level):
    """Call a correction with its own name for the level: alpha for the first two, q for the last."""
    return correct(p_values, **{LEVEL_ARGUMENT[correct]: level})


def two_sided_p(z):
    return 2 * (1 - NormalDist().cdf(abs(z)))


# ---------------------------------------------------------------- ISLP's five fund managers

def test_bonferroni_multiplies_by_five_and_caps_at_one():
    # lab, p. 586: "simply the fund_mini_pvalues multiplied by 5 and truncated to be less than or equal to 1"
    result = bonferroni(FUND)
    assert result.adjusted == pytest.approx((0.03, 1.0, 0.06, 1.0, 1.0))
    assert result.reject == (True, False, False, False, False)


def test_holm_also_rejects_manager_three():
    # p. 569: p(1) = 0.006 < 0.05/5 and p(2) = 0.012 < 0.05/4, but p(3) = 0.601 > 0.05/3
    result = holm(FUND)
    assert result.adjusted == pytest.approx((0.03, 1.0, 0.048, 1.0, 1.0))
    assert result.reject == (True, False, True, False, False)


def test_benjamini_hochberg_rejects_managers_one_and_three():
    # p. 575: p(1) < 0.05 * 1/5 and p(2) < 0.05 * 2/5; p(3), p(4), p(5) are above their lines
    result = benjamini_hochberg(FUND)
    assert result.adjusted == pytest.approx((0.03, 0.918, 0.03, 0.918, 0.918))
    assert result.reject == (True, False, True, False, False)


def test_a_bigger_q_rejects_everything_at_the_largest_p_value():
    # 0.918 is managers four's and five's adjusted p-value: at q = 0.92, p(5) = 0.918 <= 0.92 * 5/5
    assert benjamini_hochberg(FUND, q=0.92).reject == (True, True, True, True, True)
    assert benjamini_hochberg(FUND, q=0.91).reject == (True, False, True, False, False)


# ---------------------------------------------------------------- what comes back

@by_method
def test_the_result_names_its_method_level_and_p_values(correct):
    result = at_level(correct, FUND, 0.1)
    assert isinstance(result, CorrectionResult)
    assert result.method == NAMES[correct]
    assert result.level == 0.1
    assert result.p_values == tuple(FUND)


@by_method
def test_the_default_level_is_five_percent(correct):
    assert correct(FUND).level == 0.05


@by_method
def test_the_level_is_keyword_only(correct):
    with pytest.raises(TypeError):
        correct(FUND, 0.05)


@by_method
def test_every_field_holds_plain_python_numbers_even_from_numpy_input(correct):
    result = at_level(correct, np.array(FUND), np.float64(0.05))
    assert type(result.level) is float
    for field in (result.p_values, result.adjusted, result.reject):
        assert type(field) is tuple
    assert all(type(value) is float for value in result.p_values + result.adjusted)
    assert all(type(decision) is bool for decision in result.reject)


@by_method
def test_a_list_and_an_array_give_equal_results(correct):
    # == on two data classes compares their fields; with arrays inside it would raise instead
    assert correct(FUND) == correct(np.array(FUND))


@by_method
def test_the_result_is_frozen(correct):
    result = correct(FUND)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reject = (False,) * 5


@by_method
def test_rejects_exactly_where_the_adjusted_p_value_is_at_most_the_level(correct):
    # lab, p. 586: adjusted p-values "can simply be compared to the desired FWER"
    rng = np.random.default_rng(9)
    p_values = rng.random(40) ** 3
    result = at_level(correct, p_values, 0.1)
    assert result.reject == tuple(adjusted <= 0.1 for adjusted in result.adjusted)


# ---------------------------------------------------------------- the mechanics

@by_method
def test_results_come_back_in_the_order_given(correct):
    order = [3, 0, 4, 2, 1]
    shuffled = [FUND[i] for i in order]
    original = correct(FUND)
    result = correct(shuffled)
    assert result.p_values == tuple(shuffled)
    assert result.adjusted == tuple(original.adjusted[i] for i in order)
    assert result.reject == tuple(original.reject[i] for i in order)


@by_method
def test_the_input_array_is_not_changed(correct):
    p_values = np.array(FUND)
    correct(p_values)
    assert p_values.tolist() == FUND


@by_method
def test_a_single_p_value_is_left_alone(correct):
    assert correct([0.03]).adjusted == (0.03,)
    assert correct([0.03]).reject == (True,)
    assert correct([0.07]).reject == (False,)


@by_method
def test_adjusted_p_values_stay_between_the_p_value_and_one(correct):
    rng = np.random.default_rng(3)
    p_values = rng.random(30)
    result = correct(p_values)
    for p, adjusted in zip(result.p_values, result.adjusted):
        assert p - 1e-12 <= adjusted <= 1.0


@by_method
def test_tied_p_values_get_the_same_adjusted_p_value(correct):
    result = correct([0.02, 0.01, 0.30, 0.01, 0.02])
    assert result.adjusted[1] == result.adjusted[3]
    assert result.adjusted[0] == result.adjusted[4]


@by_method
def test_a_p_value_exactly_on_the_line_is_rejected(correct):
    # 0.125 and 0.25 are exact in binary floating point, so the comparison really is an equality:
    # Bonferroni's 2 * 0.125, Holm's 2 * 0.125 and 1 * 0.25, and BH's 2 * 0.125 / 1 and 2 * 0.25 / 2 all equal 0.25
    result = at_level(correct, [0.125, 0.25], 0.25)
    assert result.reject[0] is True


def test_holm_steps_down_while_bonferroni_does_not():
    # the second smallest p-value faces 0.05/2 under Holm but 0.05/3 under Bonferroni
    p_values = [0.01, 0.02, 0.5]
    assert bonferroni(p_values).reject == (True, False, False)
    assert holm(p_values).reject == (True, True, False)


def test_holm_keeps_the_adjusted_p_values_in_the_order_of_the_p_values():
    # raw products 3 * 0.01 = 0.03 and 2 * 0.012 = 0.024: the second is raised to 0.03
    assert holm([0.01, 0.012, 0.9]).adjusted == pytest.approx((0.03, 0.03, 0.9))


def test_benjamini_hochberg_steps_up_past_a_p_value_above_its_line():
    # p(2) = 0.04 > 0.05 * 2/3, yet p(3) = 0.045 <= 0.05 * 3/3, so all three are rejected
    p_values = [0.001, 0.04, 0.045]
    result = benjamini_hochberg(p_values)
    assert result.reject == (True, True, True)
    assert result.adjusted == pytest.approx((0.003, 0.045, 0.045))


def test_benjamini_hochberg_lowers_an_adjusted_p_value_to_the_one_above_it():
    # raw values m * p(j) / j: 3 * 0.02 / 2 = 0.03 but 3 * 0.025 / 3 = 0.025
    assert benjamini_hochberg([0.001, 0.02, 0.025]).adjusted == pytest.approx((0.003, 0.025, 0.025))


# ---------------------------------------------------------------- against statsmodels

@by_method
@pytest.mark.parametrize("seed", range(40))
def test_matches_statsmodels_multipletests(correct, seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 60))
    p_values = np.where(rng.random(m) < 0.5, rng.random(m), rng.random(m) ** 6)
    level = float(rng.choice([0.01, 0.05, 0.1, 0.2]))

    reject, adjusted, _, _ = multipletests(p_values, alpha=level, method=STATSMODELS_METHOD[correct])
    result = at_level(correct, p_values, level)

    assert result.adjusted == pytest.approx(tuple(adjusted), rel=1e-12, abs=0)
    assert result.reject == tuple(bool(r) for r in reject)


def test_the_three_adjusted_p_values_are_ordered():
    # every threshold of BH, q * j / m, is at least Holm's alpha / (m + 1 - j), which is at least alpha / m
    rng = np.random.default_rng(1)
    for _ in range(200):
        p_values = rng.random(int(rng.integers(1, 30))) ** 2
        bh, ho, bo = (correct(p_values).adjusted for correct in METHODS[::-1])
        for p, b, h, f in zip(p_values, bh, ho, bo):
            assert p - 1e-12 <= b <= h + 1e-12 <= f + 2e-12


# ---------------------------------------------------------------- the error rates, by simulation

def test_without_correction_ten_true_nulls_give_a_false_rejection_four_times_in_ten():
    # ISLP (13.5): with m independent true nulls, FWER = 1 - (1 - alpha) ** m = 0.401 for m = 10
    rng = np.random.default_rng(13)
    families = rng.random((4000, 10))
    any_rejected = (families <= 0.05).any(axis=1)
    assert any_rejected.mean() == pytest.approx(1 - 0.95 ** 10, abs=0.03)


def test_bonferroni_and_holm_keep_the_family_wise_error_rate_at_alpha():
    rng = np.random.default_rng(13)
    families = rng.random((4000, 10))
    bonferroni_any = [any(bonferroni(p).reject) for p in families]
    holm_any = [any(holm(p).reject) for p in families]

    # when every null is true, Holm rejects something exactly when Bonferroni does:
    # both start from the smallest p-value against alpha / m
    assert holm_any == bonferroni_any
    assert np.mean(holm_any) <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 4000)


def test_benjamini_hochberg_keeps_the_false_discovery_rate_at_q():
    # 15 true nulls and 5 false ones, whose z-statistics are centred on 3
    rng = np.random.default_rng(21)
    is_null = np.array([True] * 15 + [False] * 5)
    q = 0.1
    proportions, discoveries, holm_discoveries = [], [], []
    for _ in range(2000):
        z = rng.standard_normal(20) + np.where(is_null, 0.0, 3.0)
        p_values = [two_sided_p(value) for value in z]
        reject = np.array(benjamini_hochberg(p_values, q=q).reject)
        false = int((reject & is_null).sum())
        total = int(reject.sum())
        proportions.append(false / total if total else 0.0)  # ISLP footnote 15: V/R is 0 when R = 0
        discoveries.append(int((reject & ~is_null).sum()))
        holm_discoveries.append(int((np.array(holm(p_values, alpha=q).reject) & ~is_null).sum()))

    fdr = np.mean(proportions)
    assert fdr <= q + 3 * np.std(proportions) / math.sqrt(2000)
    # and it buys power: more of the five false nulls are found than under Holm at the same level
    assert np.mean(discoveries) > np.mean(holm_discoveries)


# ---------------------------------------------------------------- bad input

@by_method
@pytest.mark.parametrize("p_values", [[], [[0.01, 0.02]], [0.01, -0.01], [0.5, 1.01], [0.01, float("nan")]],
                         ids=["empty", "a table", "negative", "above one", "nan"])
def test_rejects_bad_p_values(correct, p_values):
    with pytest.raises(ValueError, match="p_values"):
        correct(p_values)


@by_method
@pytest.mark.parametrize("level", [0, 1, -0.1, 1.5, float("nan"), "0.05", None])
def test_rejects_a_bad_level_and_names_it(correct, level):
    name = LEVEL_ARGUMENT[correct]
    with pytest.raises(ValueError, match=rf"\b{name}\b"):
        at_level(correct, FUND, level)
