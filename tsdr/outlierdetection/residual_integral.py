import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def residual_integral(x: np.ndarray, bkp: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the residual integral of the given time series.
    """
    if np.all(x == x[0]):
        return np.array([0.0]), np.array([])

    if bkp == 0:
        ols = OLS(x, add_constant(np.arange(1, x.size + 1))).fit()
        std = ols.resid.std()
        if std == 0:
            std = 1
        errors = ols.resid / std
    else:
        train_x, test_x = np.split(x, [bkp])
        ols = OLS(train_x, add_constant(np.arange(1, train_x.size+1))).fit()
        test_pred = ols.predict(exog=add_constant(np.arange(train_x.size+1, x.size+1)))
        pred_x = np.append(ols.predict(), test_pred)
        errors = (test_pred - test_x) / (pred_x - x).std()

    intersected_idxs: np.ndarray = np.where(np.diff(np.sign(errors)))[0] + 1
    sections = np.split(errors, indices_or_sections=intersected_idxs)
    return np.array([np.sum(errs**2) for errs in sections]), sections


def residual_integral_max(x: np.ndarray, bkp: int = 0) -> tuple[float, list[tuple[int, float]]]:
    rsses, secs = residual_integral(x, bkp=bkp)
    max_rss_sec_idx: np.intp = np.argmax(rsses)
    max_rss_start: int = 0 if max_rss_sec_idx == 0 else np.sum([sec.size for sec in secs[:max_rss_sec_idx]])
    max_rss_start += bkp
    max_rss: float = rsses[max_rss_sec_idx]
    max_rss_sec: list[tuple[int, float]] = [(max_rss_start+i, v) for i, v in enumerate(secs[max_rss_sec_idx])]
    return max_rss, max_rss_sec
