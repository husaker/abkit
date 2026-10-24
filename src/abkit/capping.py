import numpy as np

def cap_upper(control, treatment, quantile=0.99):

    if quantile <= 0 or quantile > 1:
        raise ValueError("quantile must be between 0 and 1")
    control_copy = np.asarray(control, dtype=float)
    treatment_copy = np.asarray(treatment, dtype=float)

    arr = np.concatenate((control_copy, treatment_copy))

    threshold = np.quantile(arr, quantile)

    capped_control = np.where(control_copy >= threshold, threshold, control_copy)
    capped_treatment = np.where(treatment_copy >= threshold, threshold, treatment_copy)

    return capped_control, capped_treatment, threshold