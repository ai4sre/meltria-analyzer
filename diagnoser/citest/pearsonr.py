import numpy as np
from scipy import stats


def ci_test_pearsonr(dm: np.ndarray, x: int, y: int, s: set[int], **kwargs):
    assert 'corr_matrix' in kwargs
    # cm: np.ndarray = kwargs['corr_matrix']
    s = list(s)
    if len(s) == 0:
        coef, p_val = stats.pearsonr(dm[:, x], dm[:, y])
    else:
        X_coef = np.linalg.lstsq(dm[:, s], dm[:, x], rcond=None)[0]
        Y_coef = np.linalg.lstsq(dm[:, s], dm[:, y], rcond=None)[0]

        res_X = dm[:, x] - dm[:, s].dot(X_coef)
        res_Y = dm[:, y] - dm[:, s].dot(Y_coef)
        coef, p_val = stats.pearsonr(res_X, res_Y)
    return p_val
