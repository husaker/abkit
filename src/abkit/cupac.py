import numpy as np
from typing import Any, Self
from abkit.protocols import Regressor
import copy
from abkit.ztest import ComparisonResult
from abkit.cuped import cuped_test, UserMetric
import numbers


class RidgeRegressor:
    """Ridge regression written to scikit-learn's conventions: least squares with a penalty on the slopes.

    The slopes solve ISLP's eq. (6.5) on centred data, (Xc'Xc + alpha * I) beta = Xc'yc, so the
    intercept is not penalised. The constructor only stores its argument; alpha and the data
    are checked in fit, and what the model learns appears in fit as attributes ending in an
    underscore.

    Parameters
    ----------
    alpha : float, default 1.0
        Weight of the penalty on the squared slopes, zero or more; 0 gives ordinary least
        squares.

    Attributes
    ----------
    alpha : float
        The penalty weight, exactly as it was passed in.
    coef_ : np.ndarray
        One slope per feature. Exists only after fit.
    intercept_ : float
        The intercept, mean(y) - mean(X) @ coef_. Exists only after fit.
    """

    coef_: np.ndarray # declared for mypy, they do not exist until fit runs
    intercept_: np.ndarray
    alpha: float

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor arguments by name.

        Parameters
        ----------
        deep : bool, default True
            Accepted for compatibility with scikit-learn; this model has no nested models, so
            it makes no difference.

        Returns
        -------
        dict[str, Any]
            ``{"alpha": alpha}``, enough to build an unfitted copy of the model.
        """
        return {"alpha": self.alpha}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Learn the slopes and the intercept from the data.

        Parameters
        ----------
        X : np.ndarray
            Features, one row per user and one column per feature. Lists are accepted, and the
            input is not modified.
        y : np.ndarray
            Target, one value per row of X.

        Returns
        -------
        Self
            The model itself, with coef_ and intercept_ set.

        Raises
        ------
        ValueError
            If alpha is not a number of zero or more, if X is not two-dimensional or y not
            one-dimensional, if they differ in the number of rows or hold fewer than two, or if
            any value is nan or infinite.
        """
        if not isinstance(self.alpha, numbers.Real) or not self.alpha >= 0:
            raise ValueError("alpha must be a number that is zero or positive")
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")
        if X.size <= 1 or y.size <= 1:
            raise ValueError("X and y must have at least two elements")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("values can't be infinity")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("values can't be NaNs")

        n_rows, n_features = X.shape

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y, axis=0)

        Xc = X - X_mean
        yc = y - y_mean

        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        b = Xc.T @ yc
        slopes = np.linalg.solve(A, b)

        intercept = float(y.mean() - X.mean(axis=0) @ slopes)

        self.coef_ = slopes
        self.intercept_ = intercept

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target for each row of X.

        Parameters
        ----------
        X : np.ndarray
            Features, one row per user, with as many columns as the model was fitted on.

        Returns
        -------
        np.ndarray
            One prediction per row, intercept_ + X @ coef_.

        Raises
        ------
        ValueError
            If the model has not been fitted, if X has a different number of features than in
            fit, or if any value is nan or infinite.
        """
        X = np.array(X, dtype=float)
        if np.isinf(X).any():
            raise ValueError("values can't be infinity")
        if np.isnan(X).any():
            raise ValueError("values can't be NaNs")
        if not hasattr(self, "coef_"):
            raise ValueError("this model has not been fitted yet: call fit first")

        return self.intercept_ + X @ self.coef_

def unfitted_copy(model: Regressor) -> Regressor:
    """Build a new, unfitted model of the same class with the same settings.

    The copy is made by calling the model's class with the model's own constructor arguments,
    which is what scikit-learn's clone does. Every value from get_params(deep=False) is
    deep-copied first, so a setting that is a list or another model is not shared between the
    copy and the original.

    Parameters
    ----------
    model : Regressor
        Any model that follows scikit-learn's conventions, fitted or not. It is not modified.

    Returns
    -------
    Regressor
        A new object of the same class with the same settings and nothing learned.
    """

    settings = {}

    for name, value in model.get_params(deep=False).items():
        settings[name] = copy.deepcopy(value)
    mould = type(model)

    model_new = mould(**settings)

    return model_new

def assign_folds(n_users: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Give every user a fold number, at random, in folds of near-equal size.

    A random permutation of the users is taken modulo n_folds, so the fold sizes differ by at
    most one.

    Parameters
    ----------
    n_users : int
        Number of users.
    n_folds : int
        Number of folds, at least 2 and at most n_users.
    rng : np.random.Generator
        Source of randomness; a seeded generator makes the assignment reproducible.

    Returns
    -------
    np.ndarray
        One integer from 0 to n_folds - 1 per user.

    Raises
    ------
    ValueError
        If there are fewer than two folds, or more folds than users.
    """

    if n_users < n_folds:
        raise ValueError("n_users can't be less than n_folds")
    if n_folds < 2:
        raise ValueError("n_folds can't be less than 2")

    return rng.permutation(n_users) % n_folds




def cross_fit(model: Regressor, X: np.ndarray, y: np.ndarray, *, n_folds: int = 5,
                 seed: int = 0) -> np.ndarray:
    """Predict every user's target with a model that was fitted without that user.

    The users are put into n_folds folds. For each fold, an unfitted copy of the model is
    fitted on the users of all the other folds and predicts the users of this one, so no
    prediction comes from a model that has seen the user's own target. This is what keeps a
    flexible model from carrying the treatment effect into the covariate.

    Parameters
    ----------
    model : Regressor
        The model to copy for each fold. It is never fitted itself.
    X : np.ndarray
        Features, one row per user and one column per feature.
    y : np.ndarray
        Target, one value per row of X.
    n_folds : int, default 5
        Number of folds.
    seed : int, default 0
        Seed for the fold assignment; the same seed gives the same folds.

    Returns
    -------
    np.ndarray
        One float per user, in the users' order.

    Raises
    ------
    ValueError
        If X is not two-dimensional or y not one-dimensional, if they differ in the number of
        rows, if any value is nan or infinite, or if n_folds is below 2 or above the number of
        users.
    """

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if y.ndim != 1:
        raise ValueError("y must be a 1-dimensional array")
    if X.size <= 1 or y.size <= 1:
        raise ValueError("X and y must have at least two elements")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("values can't be infinity")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("values can't be NaNs")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have the same number of rows")

    n_users = X.shape[0]
    rng = np.random.default_rng(seed)

    folds = assign_folds(n_users, n_folds, rng)

    preds = np.zeros(n_users)

    for fold in range(n_folds):
        held_out = folds == fold
        fold_model = unfitted_copy(model)
        fold_model.fit(X[~held_out], y[~held_out])
        preds[held_out] = fold_model.predict(X[held_out])

    return preds

def cupac_test(values: np.ndarray, features: np.ndarray, treated: np.ndarray,
               model: Regressor, *, n_folds: int = 5, seed: int = 0, alpha: float = 0.05) -> ComparisonResult:
    """Compare two means with a model's cross-fitted prediction as the CUPED covariate.

    The model is cross-fitted once on all users of both groups together, and the two groups
    are then compared with cuped_test, the prediction taking the place of the covariate.

    Parameters
    ----------
    values : np.ndarray
        The metric, one value per user, for both groups.
    features : np.ndarray
        One row of pre-experiment features per user.
    treated : np.ndarray
        Boolean array, True for treatment users.
    model : Regressor
        An unfitted model; it is copied for each fold and never fitted itself.
    n_folds, seed : int
        Passed to cross_fit; defaults 5 and 0.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    ComparisonResult
        Difference in adjusted means, treatment minus control, as returned by cuped_test.

    Raises
    ------
    ValueError
        If treated is not a one-dimensional boolean array with one entry per user, or for
        any input cross_fit or cuped_test rejects.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in [0, 1]")
    if treated.dtype != bool:
        raise ValueError("treated must be bool")
    if treated.ndim > 1:
        raise ValueError("treated must be one dimensional")
    if treated.size != values.size:
        raise ValueError("treated and values must have the same size")

    predictions = cross_fit(model=model, X=features, y=values, n_folds=n_folds, seed=seed)

    control = UserMetric(values = values[~treated], covariate =  predictions[~treated])
    treatment = UserMetric(values = values[treated], covariate =  predictions[treated])

    results = cuped_test(control=control, treatment=treatment, alpha=alpha)

    return results





X = np.array([[0.0], [2.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
y = np.array([1.0, 3.0, 8.0, 1.0, 1.0, 2.0, 3.0, 9.0, 5.0])
alpha = 0.0

model = RidgeRegressor(alpha=alpha)
model.fit(X, y)

preds = cross_fit(model=model, X=X, y=y)
print(preds)


