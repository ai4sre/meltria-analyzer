import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def residual_integral(x: np.ndarray, standarized_resid=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the residual integral of the given time series.
    """
    if np.all(x == x[0]):
        return np.array([0.0]), np.array([])
    ols = OLS(x, add_constant(np.arange(1, x.size + 1))).fit()
    resids = ols.resid
    if standarized_resid:
        std = ols.resid.std()
        if std == 0:
            std = 1
        resids = ols.resid / std
    intersected_idxs: np.ndarray = np.where(np.diff(np.sign(resids)))[0] + 1
    sections = np.split(resids, indices_or_sections=intersected_idxs)
    return np.array([np.sum(sec**2) for sec in sections]), sections


def residual_integral_max(x: np.ndarray) -> tuple[float, tuple[int, int]]:
    rsses, secs = residual_integral(x)
    max_rss_idx = np.argmax(rsses)
    max_rss_start = np.sum([sec.size for sec in secs[:max_rss_idx]]) - 1
    max_rss_range = (max_rss_start, max_rss_start + secs[max_rss_idx].size + 1)
    max_rss = rsses[max_rss_idx]
    return max_rss, max_rss_range
