from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OLSResult:
    """Coefficients of an OLS fit with three kinds of standard errors.

    For an intercept plus a treatment dummy, the dummy's classical SE equals the
    pooled-variance t-test's, HC0 the unpooled z-test's (for a 0/1 outcome), and
    HC2 Welch's.

    Attributes
    ----------
    coef : np.ndarray
        Estimated coefficients, one per column of X.
    classical_se : np.ndarray
        Standard errors that assume every row has the same residual variance.
    robust_se : np.ndarray
        HC0 (White) standard errors, which let each row have its own variance.
    hc2_se : np.ndarray
        HC2 standard errors: HC0 with each squared residual divided by (1 - leverage).
    """

    coef: np.ndarray
    classical_se: np.ndarray
    robust_se: np.ndarray
    hc2_se: np.ndarray

def ols(y: np.ndarray, X: np.ndarray) -> OLSResult:
    """Fit OLS and return the coefficients with classical, HC0 and HC2 standard errors.

    Solves (X^T X) beta = X^T y with X as given, so include a column of ones if you
    want an intercept.

    Parameters
    ----------
    y : np.ndarray
        Outcome, one value per row.
    X : np.ndarray
        Design matrix of shape (n_rows, n_columns), e.g. a column of ones and a
        treatment dummy.

    Returns
    -------
    OLSResult
        Coefficients and their standard errors.

    Raises
    ------
    ValueError
        If X and y have different numbers of rows, or X doesn't have more rows than
        columns.
    """

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.shape[0] != X.shape[0]:
        raise ValueError('X and y must have the same number of rows')
    if y.shape[0] <= X.shape[1]:
        raise ValueError('There should be more rows than columns')


    # X.T @ X counts users, X.T @ X is a table of how much every pair of columns overlaps. Each entry is Σ (column j × column k) over all users
    # np.linalg.inv gives the matrix that “divides” by that table, the way 1/5 undoes 5,
    # so that inv(X.T @ X) @ X.T @ y solves the two equations
    # and returns β₀ (control’s rate) and β₁ (the difference).

    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y  # [0.2, 0.3]   ← control rate, difference
    resid = y - X @ coef  # control converters 0.8, non-converters -0.2;

    n, k = X.shape
    sigma2 = resid @ resid / (n - k)  # resid @ resid is Σ residual², which is RSS, divide by n − k rather than n because we used up k numbers fitting the coefficients
    classical_se = np.sqrt(np.diag(xtx_inv) * sigma2) # SE of the slope coefficietn B1

    meat = (X * (resid**2)[:, None]).T @ X  # Xᵀ diag(e²) X, without an n×n matrix, resid[:, None] turns the residuals into a column, so each row of X is multiplied by its own e².
    robust_se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))  # The robust SE of the slope, is exactly z-test’s unpooled SE. Each user keeps their own squared residual instead of sharing one σ²

    h = np.sum((X @ xtx_inv) * X, axis=1)  # leverage of each row, shape (n,)
    meat2 = (X * (resid ** 2 / (1 - h))[:, None]).T @ X  # like HC0's meat, with corrected e²
    hc2_se = np.sqrt(np.diag(xtx_inv @ meat2 @ xtx_inv))

    return OLSResult(coef=coef, classical_se=classical_se, robust_se=robust_se, hc2_se=hc2_se)


