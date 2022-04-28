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


def residual_integral_max(x: np.ndarray) -> tuple[float, list[tuple[int, float]]]:
    rsses, secs = residual_integral(x)
    max_rss_sec_idx: np.intp = np.argmax(rsses)
    max_rss_start: int = 0 if max_rss_sec_idx == 0 else np.sum([sec.size for sec in secs[:max_rss_sec_idx]])
    max_rss: float = rsses[max_rss_sec_idx]
    max_rss_sec: list[tuple[int, float]] = [(max_rss_start+i, v) for i, v in enumerate(secs[max_rss_sec_idx])]
    return max_rss, max_rss_sec
