import numpy as np

def cap_upper(control: np.ndarray, treatment: np.ndarray,
              quantile: float=0.99) -> tuple[np.ndarray, np.ndarray, float]:

    """Cap both groups at one upper threshold computed from the pooled data.

    Values above the threshold are replaced by the threshold. Both groups share it,
    so the capped metric means the same thing in each group.

    Parameters
    ----------
    control, treatment : np.ndarray
        One value per user in each group.
    quantile : float, default 0.99
        Quantile of the pooled values used as the threshold, in (0, 1].

    Returns
    -------
    capped_control : np.ndarray
        Control values with anything above the threshold set to the threshold.
    capped_treatment : np.ndarray
        The same for treatment.
    threshold : float
        The cap.

    Raises
    ------
    ValueError
        If quantile is outside (0, 1].
    """

    if quantile <= 0 or quantile > 1:
        raise ValueError("quantile must be between 0 and 1")
    control_copy = np.asarray(control, dtype=float)
    treatment_copy = np.asarray(treatment, dtype=float)

    arr = np.concatenate((control_copy, treatment_copy))

    threshold = np.quantile(arr, quantile)

    capped_control = np.where(control_copy >= threshold, threshold, control_copy)
    capped_treatment = np.where(treatment_copy >= threshold, threshold, treatment_copy)

    return capped_control, capped_treatment, threshold