import numpy as np
from scipy import stats


def ci_test_pearsonar(data_matrix: np.ndarray, x: int, y: int, s: set[int], **kwargs):
    assert 'corr_matrix' in kwargs
    cm: np.ndarray = kwargs['corr_matrix']
    s = list(s)
    if len(s) == 0:
        coef, p_val = stats.pearsonr(cm[x], cm[y])
    else:
        X_coef = np.linalg.lstsq(cm[:, s], cm[:, x], rcond=None)[0]
        Y_coef = np.linalg.lstsq(cm[:, s], cm[:, y], rcond=None)[0]

        res_X = cm[:, x] - cm[:, s].dot(X_coef)
        res_Y = cm[:, y] - cm[:, s].dot(Y_coef)
        coef, p_val = stats.pearsonr(res_X, res_Y)
    return p_val
