"""Chapter 7: CUPAC, a model's cross-fitted prediction as the CUPED covariate.

Five pieces are specified here. ``RidgeRegressor`` is your own model, written to scikit-learn's
conventions and checked against scikit-learn's ``Ridge``. ``unfitted_copy`` builds a fresh model from
another model's parameters. ``assign_folds`` and ``cross_fit`` give every user a prediction from a
model that never saw them, checked against ``cross_val_predict``. ``cupac_test`` puts the pieces
together with chapter 6's ``cuped_test``. The ``Regressor`` protocol is checked with mypy.
"""

import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from abkit.cupac import RidgeRegressor, assign_folds, cross_fit, cupac_test, unfitted_copy
from abkit.cuped import UserMetric, cuped_test
from abkit.protocols import Regressor
from abkit.regression import ols
from abkit.ttest import welch_t_test
from abkit.ztest import ComparisonResult

PROJECT = Path(__file__).parents[1]


def make_users(rng, n, effect=0.0):
    """n users with three features; the metric is linear in the first two, the third is noise.

    Half the users, picked at random, are treated, and ``effect`` is added to their metric. The
    features explain 3.25 of the metric's variance of 4.25, so the best covariate has rho**2 = 0.76.
    """
    X = rng.normal(size=(n, 3))
    y = 2.0 + 1.5 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(size=n)
    treated = np.zeros(n, dtype=bool)
    treated[rng.permutation(n)[: n // 2]] = True
    return y + effect * treated, X, treated


@pytest.fixture
def users():
    return make_users(np.random.default_rng(7), 400)


class Memoriser:
    """A model that overfits as much as a model can: it stores every training user's value.

    The first feature is a user id. For a user it has seen it returns that user's own value; for a
    user it has not seen it returns the mean of the users it was trained on.
    """

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y):
        self.memory_ = dict(zip(np.asarray(X)[:, 0].tolist(), np.asarray(y).tolist()))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.memory_.get(user_id, self.mean_) for user_id in np.asarray(X)[:, 0].tolist()])


class TrainingSize:
    """Predicts, for every row, the number of rows it was trained on."""

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y):
        self.n_rows_ = len(y)
        return self

    def predict(self, X):
        return np.full(len(X), float(self.n_rows_))


class Recorder:
    """Keeps a list, shared by all instances with the same ``log``, of the instances that were fitted."""

    def __init__(self, log=None):
        self.log = log

    def get_params(self, deep=True):
        return {"log": self.log}

    def __deepcopy__(self, memo):  # the shared log must stay shared when the parameters are copied
        return Recorder(self.log)

    def fit(self, X, y):
        self.log.append(self)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class SharedList(list):
    """A list that survives copy.deepcopy as itself, so that every copy of a Recorder writes to it."""

    def __deepcopy__(self, memo):
        return self


# ---------------------------------------------------------------- RidgeRegressor: the conventions

def test_init_only_stores_its_argument():
    # scikit-learn's first convention: __init__ stores what it was given and does nothing else
    assert RidgeRegressor().alpha == 1.0
    assert RidgeRegressor(alpha=0.25).alpha == 0.25
    assert RidgeRegressor(alpha=-1).alpha == -1          # no validation here: it happens in fit
    assert RidgeRegressor(alpha="not a number").alpha == "not a number"


def test_learned_attributes_exist_only_after_fit(users):
    y, X, _ = users
    model = RidgeRegressor()
    assert not hasattr(model, "coef_") and not hasattr(model, "intercept_")
    model.fit(X, y)
    assert model.coef_.shape == (3,)
    assert isinstance(model.intercept_, float)


def test_fit_returns_the_model_itself(users):
    y, X, _ = users
    model = RidgeRegressor()
    assert model.fit(X, y) is model


@pytest.mark.parametrize("alpha", [-1, -1e-9, "0.5", None, float("nan")])
def test_fit_rejects_a_bad_alpha(alpha, users):
    y, X, _ = users
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegressor(alpha=alpha).fit(X, y)


def test_get_params_returns_the_constructor_arguments():
    assert RidgeRegressor(alpha=2.0).get_params() == {"alpha": 2.0}
    assert RidgeRegressor(alpha=2.0).get_params(deep=False) == {"alpha": 2.0}


def test_sklearn_can_clone_it(users):
    y, X, _ = users
    fitted = RidgeRegressor(alpha=3.0).fit(X, y)
    copy = clone(fitted)
    assert isinstance(copy, RidgeRegressor) and copy is not fitted
    assert copy.alpha == 3.0
    assert not hasattr(copy, "coef_")


# ---------------------------------------------------------------- RidgeRegressor: the numbers

def test_ridge_by_hand():
    # x = 0, 2, 4 and y = 1, 3, 8: means 2 and 4, sum of squares of x 8, sum of cross products 14
    X, y = np.array([[0.0], [2.0], [4.0]]), np.array([1.0, 3.0, 8.0])
    least_squares = RidgeRegressor(alpha=0.0).fit(X, y)
    assert least_squares.coef_ == pytest.approx([14 / 8])
    assert least_squares.intercept_ == pytest.approx(4 - 2 * 14 / 8)
    penalised = RidgeRegressor(alpha=6.0).fit(X, y)        # 14 / (8 + 6) = 1
    assert penalised.coef_ == pytest.approx([1.0])
    assert penalised.intercept_ == pytest.approx(2.0)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 10.0, 1000.0])
def test_matches_sklearn_ridge(alpha, users):
    y, X, _ = users
    mine, theirs = RidgeRegressor(alpha=alpha).fit(X, y), Ridge(alpha=alpha).fit(X, y)
    assert mine.coef_ == pytest.approx(theirs.coef_, rel=1e-8)
    assert mine.intercept_ == pytest.approx(theirs.intercept_, rel=1e-8)
    assert mine.predict(X[:10]) == pytest.approx(theirs.predict(X[:10]), rel=1e-8)


def test_alpha_zero_is_least_squares(users):
    y, X, _ = users
    model = RidgeRegressor(alpha=0.0).fit(X, y)
    fit = ols(y, np.column_stack([np.ones(len(y)), X]))
    assert model.intercept_ == pytest.approx(fit.coef[0])
    assert model.coef_ == pytest.approx(fit.coef[1:])


def test_the_intercept_is_not_penalised(users):
    y, X, _ = users
    X = X + 5.0                                            # features far from zero
    model = RidgeRegressor(alpha=1e12).fit(X, y)           # a penalty so large that the slopes vanish
    assert model.coef_ == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert model.intercept_ == pytest.approx(y.mean(), abs=1e-4)
    shifted = RidgeRegressor(alpha=10.0).fit(X, y + 100.0)
    assert shifted.coef_ == pytest.approx(RidgeRegressor(alpha=10.0).fit(X, y).coef_)
    assert shifted.intercept_ == pytest.approx(RidgeRegressor(alpha=10.0).fit(X, y).intercept_ + 100.0)


def test_predict_is_intercept_plus_features_times_slopes(users):
    y, X, _ = users
    model = RidgeRegressor(alpha=2.0).fit(X, y)
    new = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    prediction = model.predict(new)
    assert prediction.shape == (2,)
    assert prediction == pytest.approx([model.intercept_, model.intercept_ + new[1] @ model.coef_])


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="fit"):
        RidgeRegressor().predict(np.zeros((2, 3)))


def test_predict_rejects_a_different_number_of_features(users):
    y, X, _ = users
    model = RidgeRegressor().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 2)))


@pytest.mark.parametrize("X, y", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),                               # X is not a table
    ([[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]),                   # y is a table
    ([[1.0], [2.0], [3.0]], [1.0, 2.0]),                              # different numbers of rows
    ([[1.0], [np.nan], [3.0]], [1.0, 2.0, 3.0]),                      # not finite
    ([[1.0], [2.0], [3.0]], [1.0, np.inf, 3.0]),
])
def test_fit_rejects_bad_data(X, y):
    with pytest.raises(ValueError):
        RidgeRegressor().fit(np.array(X), np.array(y))


def test_fit_accepts_lists_and_leaves_its_inputs_alone():
    X, y = [[0.0], [2.0], [4.0]], [1.0, 3.0, 8.0]
    model = RidgeRegressor(alpha=0.0).fit(X, y)
    assert model.coef_ == pytest.approx([1.75])
    assert X == [[0.0], [2.0], [4.0]] and y == [1.0, 3.0, 8.0]
    arrays = np.array(X), np.array(y)
    before = arrays[0].copy(), arrays[1].copy()
    RidgeRegressor().fit(*arrays)
    assert np.array_equal(arrays[0], before[0]) and np.array_equal(arrays[1], before[1])


def test_fitting_again_replaces_what_was_learned(users):
    y, X, _ = users
    model = RidgeRegressor().fit(X, y)
    first = model.coef_.copy()
    model.fit(X, -y)
    assert model.coef_ == pytest.approx(-first)


# ---------------------------------------------------------------- unfitted_copy

def test_unfitted_copy_keeps_the_parameters_and_drops_what_was_learned(users):
    y, X, _ = users
    fitted = RidgeRegressor(alpha=4.0).fit(X, y)
    copy = unfitted_copy(fitted)
    assert type(copy) is RidgeRegressor and copy is not fitted
    assert copy.alpha == 4.0
    assert not hasattr(copy, "coef_")
    assert hasattr(fitted, "coef_")                        # the original is untouched


def test_unfitted_copy_works_on_scikit_learn_models(users):
    y, X, _ = users
    fitted = Ridge(alpha=3.0, fit_intercept=False).fit(X, y)
    copy = unfitted_copy(fitted)
    assert type(copy) is Ridge and copy is not fitted
    assert copy.get_params() == fitted.get_params()
    assert not hasattr(copy, "coef_")


def test_unfitted_copy_does_not_share_mutable_parameters():
    class Weighted:
        def __init__(self, weights=None):
            self.weights = weights

        def get_params(self, deep=True):
            return {"weights": self.weights}

    original = Weighted(weights=[1.0, 2.0])
    copy = unfitted_copy(original)
    assert copy.weights == [1.0, 2.0] and copy.weights is not original.weights


# ---------------------------------------------------------------- assign_folds

@pytest.mark.parametrize("n_users, n_folds", [(10, 3), (100, 5), (7, 7), (1001, 10), (2, 2)])
def test_folds_are_of_near_equal_size(n_users, n_folds):
    folds = assign_folds(n_users, n_folds, np.random.default_rng(0))
    assert folds.shape == (n_users,)
    assert np.issubdtype(folds.dtype, np.integer)
    sizes = np.bincount(folds, minlength=n_folds)
    assert len(sizes) == n_folds                           # every number from 0 to n_folds - 1, no other
    assert sizes.max() - sizes.min() <= 1


def test_folds_are_random_and_come_from_the_generator():
    first = assign_folds(1000, 5, np.random.default_rng(3))
    assert np.array_equal(first, assign_folds(1000, 5, np.random.default_rng(3)))
    assert not np.array_equal(first, assign_folds(1000, 5, np.random.default_rng(4)))
    rng = np.random.default_rng(3)
    assert not np.array_equal(assign_folds(1000, 5, rng), assign_folds(1000, 5, rng))
    assert not np.array_equal(first, np.arange(1000) % 5)  # not the users' order
    assert not np.array_equal(first, np.sort(first))       # not consecutive blocks


@pytest.mark.parametrize("n_users, n_folds", [(10, 1), (10, 0), (10, -2), (4, 5)])
def test_assign_folds_rejects_impossible_requests(n_users, n_folds):
    with pytest.raises(ValueError):
        assign_folds(n_users, n_folds, np.random.default_rng(0))


# ---------------------------------------------------------------- cross_fit

@pytest.mark.parametrize("n_folds, seed", [(5, 0), (2, 1), (10, 2)])
def test_cross_fit_matches_cross_val_predict(n_folds, seed, users):
    y, X, _ = users
    folds = assign_folds(len(y), n_folds, np.random.default_rng(seed))
    expected = cross_val_predict(Ridge(alpha=2.0), X, y, cv=PredefinedSplit(folds))
    assert cross_fit(Ridge(alpha=2.0), X, y, n_folds=n_folds, seed=seed) == pytest.approx(expected)
    assert cross_fit(RidgeRegressor(alpha=2.0), X, y, n_folds=n_folds, seed=seed) == pytest.approx(expected)


def test_no_user_is_predicted_by_a_model_that_saw_them():
    rng = np.random.default_rng(1)
    y = rng.normal(10.0, 1.0, size=103)
    ids = np.arange(103.0).reshape(-1, 1)
    in_sample = Memoriser().fit(ids, y).predict(ids)
    assert in_sample == pytest.approx(y)                   # fitted on everyone, it returns everyone's own value
    out_of_fold = cross_fit(Memoriser(), ids, y)
    assert not np.any(np.isclose(out_of_fold, y))          # cross-fitted, nobody gets their own value back
    assert len(np.unique(out_of_fold)) == 5                # only the five training means


def test_every_fold_model_is_trained_on_all_the_other_folds():
    y = np.arange(103.0)
    X = np.zeros((103, 1))
    folds = assign_folds(103, 5, np.random.default_rng(9))
    own_fold_size = np.bincount(folds)[folds]
    assert cross_fit(TrainingSize(), X, y, n_folds=5, seed=9) == pytest.approx(103 - own_fold_size)


def test_cross_fit_leaves_the_given_model_unfitted(users):
    y, X, _ = users
    model = RidgeRegressor()
    cross_fit(model, X, y)
    assert not hasattr(model, "coef_")
    tree = DecisionTreeRegressor(max_depth=2)
    predictions = cross_fit(tree, X, y)
    assert not hasattr(tree, "tree_")
    assert np.isfinite(predictions).all()


def test_cross_fit_works_with_a_scikit_learn_pipeline(users):
    # a pipeline is an estimator whose parameters are other estimators: its steps must be copied, not shared
    y, X, _ = users
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    folds = assign_folds(len(y), 5, np.random.default_rng(0))
    expected = cross_val_predict(pipeline, X, y, cv=PredefinedSplit(folds))
    assert cross_fit(pipeline, X, y) == pytest.approx(expected)
    assert not hasattr(pipeline.steps[-1][1], "coef_")     # the steps of the given pipeline stay unfitted


def test_cross_fit_fits_one_fresh_model_per_fold(users):
    y, X, _ = users
    log = SharedList()
    given = Recorder(log)
    cross_fit(given, X, y, n_folds=4)
    assert len(log) == 4
    assert len({id(model) for model in log}) == 4          # four different objects
    assert all(model is not given for model in log)


def test_cross_fit_passes_the_parameters_on(users):
    y, X, _ = users
    folds = assign_folds(len(y), 5, np.random.default_rng(0))
    flat = cross_fit(RidgeRegressor(alpha=1e12), X, y)     # slopes of zero: each prediction is a training mean
    training_means = np.array([y[folds != fold].mean() for fold in range(5)])
    assert flat == pytest.approx(training_means[folds], abs=1e-4)
    assert not np.allclose(flat, cross_fit(RidgeRegressor(alpha=0.0), X, y))


def test_cross_fit_is_reproducible_from_its_seed(users):
    y, X, _ = users
    assert np.array_equal(cross_fit(RidgeRegressor(), X, y, seed=5), cross_fit(RidgeRegressor(), X, y, seed=5))
    assert not np.allclose(cross_fit(RidgeRegressor(), X, y, seed=5), cross_fit(RidgeRegressor(), X, y, seed=6))
    assert np.array_equal(cross_fit(RidgeRegressor(), X, y), cross_fit(RidgeRegressor(), X, y, n_folds=5, seed=0))


def test_cross_fit_returns_one_float_per_user_and_leaves_its_inputs_alone(users):
    y, X, _ = users
    X_before, y_before = X.copy(), y.copy()
    predictions = cross_fit(RidgeRegressor(), X, y)
    assert predictions.shape == y.shape and predictions.dtype == np.float64
    assert np.array_equal(X, X_before) and np.array_equal(y, y_before)
    assert cross_fit(RidgeRegressor(), X.tolist(), y.tolist()) == pytest.approx(predictions)


def test_cross_fit_takes_folds_and_seed_by_keyword_only(users):
    y, X, _ = users
    parameters = inspect.signature(cross_fit).parameters
    assert parameters["n_folds"].kind is inspect.Parameter.KEYWORD_ONLY and parameters["n_folds"].default == 5
    assert parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY and parameters["seed"].default == 0
    with pytest.raises(TypeError):
        cross_fit(RidgeRegressor(), X, y, 5)


@pytest.mark.parametrize("change", ["short y", "flat X", "one fold", "too many folds"])
def test_cross_fit_rejects_bad_input(change, users):
    y, X, _ = users
    arguments = {"X": X, "y": y, "n_folds": 5}
    if change == "short y":
        arguments["y"] = y[:-1]
    if change == "flat X":
        arguments["X"] = X[:, 0]
    if change == "one fold":
        arguments["n_folds"] = 1
    if change == "too many folds":
        arguments["n_folds"] = len(y) + 1
    with pytest.raises(ValueError):
        cross_fit(RidgeRegressor(), arguments["X"], arguments["y"], n_folds=arguments["n_folds"])


# ---------------------------------------------------------------- cupac_test

@pytest.mark.parametrize("n_folds, seed, alpha", [(5, 0, 0.05), (3, 4, 0.1)])
def test_cupac_is_cuped_with_the_cross_fitted_prediction(n_folds, seed, alpha):
    y, X, treated = make_users(np.random.default_rng(11), 400, effect=0.5)
    prediction = cross_fit(RidgeRegressor(), X, y, n_folds=n_folds, seed=seed)   # on all users together
    expected = cuped_test(UserMetric(y[~treated], prediction[~treated]),
                          UserMetric(y[treated], prediction[treated]), alpha=alpha)
    result = cupac_test(y, X, treated, RidgeRegressor(), n_folds=n_folds, seed=seed, alpha=alpha)
    assert isinstance(result, ComparisonResult)
    assert result.estimate == pytest.approx(expected.estimate)
    assert result.standard_error == pytest.approx(expected.standard_error)
    assert result.p_value == pytest.approx(expected.p_value)
    assert (result.ci_low, result.ci_high) == pytest.approx((expected.ci_low, expected.ci_high))
    assert result.df == pytest.approx(expected.df)
    assert result.alpha == alpha


def test_cupac_estimates_treatment_minus_control():
    y, X, treated = make_users(np.random.default_rng(12), 2000, effect=1.0)
    result = cupac_test(y, X, treated, RidgeRegressor())
    assert result.estimate == pytest.approx(1.0, abs=4 * result.standard_error)
    assert result.estimate > 0.7
    swapped = cupac_test(y, X, ~treated, RidgeRegressor())
    assert swapped.estimate == pytest.approx(-result.estimate)


def test_the_standard_error_shrinks_by_the_out_of_fold_correlation():
    y, X, treated = make_users(np.random.default_rng(13), 4000)
    raw = welch_t_test(y[~treated], y[treated])
    result = cupac_test(y, X, treated, RidgeRegressor())
    rho = np.corrcoef(y, cross_fit(RidgeRegressor(), X, y))[0, 1]
    assert rho ** 2 == pytest.approx(3.25 / 4.25, abs=0.03)
    assert result.standard_error / raw.standard_error == pytest.approx(np.sqrt(1 - rho ** 2), abs=0.01)


def test_features_that_predict_nothing_cost_nothing():
    rng = np.random.default_rng(14)
    y, _, treated = make_users(rng, 4000, effect=0.3)
    noise = rng.normal(size=(4000, 3))                     # unrelated to the metric
    raw = welch_t_test(y[~treated], y[treated])
    result = cupac_test(y, noise, treated, RidgeRegressor())
    assert result.standard_error == pytest.approx(raw.standard_error, rel=0.01)
    assert result.estimate == pytest.approx(raw.estimate, abs=0.1 * raw.standard_error)


def test_cupac_holds_five_percent_when_there_is_no_effect():
    rng = np.random.default_rng(15)
    rejections = 0
    for _ in range(600):
        y, X, treated = make_users(rng, 300)
        rejections += cupac_test(y, X, treated, RidgeRegressor()).p_value < 0.05
    assert 0.025 <= rejections / 600 <= 0.08               # 5% plus or minus about three Monte Carlo SE


def test_cupac_is_unbiased_when_there_is_an_effect():
    rng = np.random.default_rng(16)
    estimates = []
    for _ in range(400):
        y, X, treated = make_users(rng, 300, effect=0.5)
        estimates.append(cupac_test(y, X, treated, RidgeRegressor()).estimate)
    monte_carlo_se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    assert np.mean(estimates) == pytest.approx(0.5, abs=3.5 * monte_carlo_se)


def test_an_in_sample_memoriser_erases_the_effect_and_cross_fitting_keeps_it():
    rng = np.random.default_rng(17)
    y, _, treated = make_users(rng, 2000, effect=1.0)
    ids = np.arange(2000.0).reshape(-1, 1)
    raw = welch_t_test(y[~treated], y[treated])
    # In-sample, the memoriser's prediction is the user's own value. The covariate then carries the
    # whole effect, theta is 1, and the adjustment subtracts the effect from the treatment group.
    in_sample = Memoriser().fit(ids, y).predict(ids)
    metric = UserMetric(y, in_sample)
    assert metric.theta == pytest.approx(1.0)
    adjusted = metric.adjust()
    assert adjusted[treated].mean() - adjusted[~treated].mean() == pytest.approx(0.0, abs=1e-9)
    # Cross-fitted, the same model can only return a training mean, and the effect survives.
    result = cupac_test(y, ids, treated, Memoriser())
    assert result.estimate == pytest.approx(raw.estimate, abs=0.5 * raw.standard_error)
    assert result.estimate > 0.7


def test_cupac_leaves_the_given_model_unfitted(users):
    y, X, treated = users
    model = RidgeRegressor()
    cupac_test(y, X, treated, model)
    assert not hasattr(model, "coef_")


def test_cupac_works_with_a_scikit_learn_model(users):
    y, X, treated = users
    result = cupac_test(y, X, treated, DecisionTreeRegressor(max_depth=3, random_state=0))
    raw = welch_t_test(y[~treated], y[treated])
    assert np.isfinite([result.estimate, result.standard_error, result.p_value]).all()
    assert result.standard_error < raw.standard_error


def test_cupac_takes_its_options_by_keyword_only(users):
    y, X, treated = users
    parameters = inspect.signature(cupac_test).parameters
    for name, default in [("n_folds", 5), ("seed", 0), ("alpha", 0.05)]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY and parameters[name].default == default
    with pytest.raises(TypeError):
        cupac_test(y, X, treated, RidgeRegressor(), 5)


@pytest.mark.parametrize("change", ["integer mask", "short mask", "flat features", "short values", "alpha"])
def test_cupac_rejects_bad_input(change, users):
    y, X, treated = users
    arguments = {"values": y, "features": X, "treated": treated, "alpha": 0.05}
    if change == "integer mask":
        arguments["treated"] = treated.astype(int)         # 0 and 1 would index two users, not two groups
    if change == "short mask":
        arguments["treated"] = treated[:-1]
    if change == "flat features":
        arguments["features"] = X[:, 0]
    if change == "short values":
        arguments["values"] = y[:-1]
    if change == "alpha":
        arguments["alpha"] = 1.5
    with pytest.raises(ValueError):
        cupac_test(arguments["values"], arguments["features"], arguments["treated"], RidgeRegressor(),
                   alpha=arguments["alpha"])


# ---------------------------------------------------------------- the Regressor protocol, checked by mypy

PREAMBLE = """
    from typing import Any
    import numpy as np
    from abkit.cupac import RidgeRegressor, cross_fit
    from abkit.protocols import Regressor
"""


def mypy_result(code, tmp_path):
    """Run mypy on a small module written to tmp_path; return (accepted, output)."""
    assert importlib.util.find_spec("mypy") is not None, "mypy isn't installed: run `uv add --dev mypy`"
    check = tmp_path / "regressor_check.py"
    check.write_text(textwrap.dedent(PREAMBLE) + textwrap.dedent(code))
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(PROJECT / "pyproject.toml"), str(check)],
        cwd=PROJECT, capture_output=True, text=True,
        env={**os.environ, "MYPYPATH": str(PROJECT / "src")},
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def test_ridge_regressor_satisfies_the_protocol(tmp_path):
    accepted, output = mypy_result("""
        model: Regressor = RidgeRegressor(alpha=2.0)
        predictions: np.ndarray = cross_fit(RidgeRegressor(), np.zeros((10, 2)), np.zeros(10), n_folds=2)
    """, tmp_path)
    assert accepted, output


def test_the_protocol_does_not_care_what_the_parameters_are_called(tmp_path):
    accepted, output = mypy_result("""
        class Mean:
            def get_params(self, deep: bool = True) -> dict[str, Any]:
                return {}
            def fit(self, features: np.ndarray, target: np.ndarray) -> "Mean":
                self.mean_ = float(target.mean())
                return self
            def predict(self, features: np.ndarray) -> np.ndarray:
                return np.full(len(features), self.mean_)

        model: Regressor = Mean()
    """, tmp_path)
    assert accepted, output


@pytest.mark.parametrize("broken_class", [
    """
        class Broken:                                       # no predict
            def get_params(self, deep: bool = True) -> dict[str, Any]:
                return {}
            def fit(self, X: np.ndarray, y: np.ndarray) -> "Broken":
                return self
    """,
    """
        class Broken:                                       # no get_params: nothing to build a copy from
            def fit(self, X: np.ndarray, y: np.ndarray) -> "Broken":
                return self
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X))
    """,
    """
        class Broken:                                       # predict returns text
            def get_params(self, deep: bool = True) -> dict[str, Any]:
                return {}
            def fit(self, X: np.ndarray, y: np.ndarray) -> "Broken":
                return self
            def predict(self, X: np.ndarray) -> str:
                return "high"
    """,
])
def test_mypy_rejects_a_class_that_is_not_a_regressor(broken_class, tmp_path):
    accepted, output = mypy_result(broken_class + """
        model: Regressor = Broken()
    """, tmp_path)
    assert not accepted
    assert "Regressor" in output


def test_fit_and_predict_are_positional_only_in_the_protocol():
    for method in (Regressor.fit, Regressor.predict):
        kinds = [p.kind for name, p in inspect.signature(method).parameters.items() if name != "self"]
        assert kinds and all(kind is inspect.Parameter.POSITIONAL_ONLY for kind in kinds)


def test_the_protocol_is_static_only():
    # Regressor is not runtime-checkable: abkit trusts the type checker and the first call to fit
    with pytest.raises(TypeError):
        isinstance(RidgeRegressor(), Regressor)
