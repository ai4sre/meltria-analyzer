from typing import Any

import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.base.prediction import PredictionResults


class AROutlierDetector:
    maxlag: int

    def __init__(self, maxlag: int = 0):
        self.maxlag = maxlag

    def score(self, x: np.ndarray, regression: str = 'c', ic: str = 'aic') -> np.ndarray:
        maxlag = int(x.size * 0.2) if self.maxlag == 0 else self.maxlag
        sel = ar_select_order(x, maxlag=maxlag, trend=regression, ic=ic, old_names=False)
        model_fit = sel.model.fit()
        r: int = 0
        if model_fit.ar_lags is None or len(model_fit.ar_lags) > 0:
            r = model_fit.ar_lags[-1]
        sig2 = model_fit.sigma2
        preds: np.ndarray = model_fit.get_prediction().predicted_mean
        scores: np.ndarray = np.zeros(x.size, dtype=np.float32)
        for i, (xi, pred) in enumerate(zip(x[r:], preds[r:])):
            scores[r + i] = (xi - pred) ** 2 / sig2
        return scores

    def detect(self, x: np.ndarray, threshold: float) -> list[float]:
        return [s for s in self.score(x) if s >= threshold]

    def detect_by_fitting_dist(self, scores: np.ndarray, threshold: float) -> list[tuple[int, float]]:
        abn_th = chi2.interval(1-threshold, 1)[1]
        anomalies: list[tuple[int, float]] = []
        for i, a in enumerate(scores):
            if a > abn_th:
                anomalies.append((i, a))
        return anomalies
