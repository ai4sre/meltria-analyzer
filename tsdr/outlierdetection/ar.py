import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.ar_model import (AutoReg, AutoRegResultsWrapper,
                                      ar_select_order)


class AROutlierDetector:
    def __init__(self, samples: np.ndarray, maxlag: int = 0):
        self._samples = samples
        self._maxlag = int(self._samples.size * 0.2) if maxlag == 0 else maxlag
        self._model: AutoRegResultsWrapper = None
        self._lag = 0

    def fit(
        self,
        regression: str = 'n',
        lag: int = 0,
        ic: str = 'bic',
    ) -> None:
        autolag: bool = lag == 0
        if autolag:
            sel = ar_select_order(self._samples, maxlag=self._maxlag, trend=regression, ic=ic)
            model_fit = sel.model.fit()
            self._model = model_fit
            if model_fit.ar_lags is not None and len(model_fit.ar_lags) > 0:
                self._lag = model_fit.ar_lags[-1]
        else:
            self._lag = lag
            model = AutoReg(endog=self._samples, lags=lag, trend=regression)
            self._model = model.fit()

    def predict(self, dynamic: bool = False) -> tuple[np.ndarray, float]:
        pred_results = self._model.get_prediction(dynamic=dynamic)
        preds = pred_results.predicted_mean
        # remove the plots for the lag. And read through the first value because the prediction line is shifted by 1 plot for some reason.
        preds = preds[self._lag+1:]
        var = pred_results.var_pred_mean
        sig2: float = var[self._lag]
        if sig2 == 0:
            return np.empty([]), 0
        return preds, sig2

    def anomaly_scores(self, **kwargs) -> np.ndarray:
        preds, sig2 = self.predict(**kwargs)
        scores: np.ndarray = np.zeros(self._samples.size, dtype=np.float32)
        for i, (xi, pred) in enumerate(zip(self._samples[self._lag:], preds)):
            scores[self._lag+i] = (xi - pred) ** 2 / sig2
        return scores

    @classmethod
    def detect_by_fitting_dist(
        cls,
        scores: np.ndarray,
        threshold: float,
    ) -> tuple[list[tuple[int, float]], float]:
        abn_th = chi2.interval(1-threshold, 1)[1]
        anomalies: list[tuple[int, float]] = []
        for i, a in enumerate(scores):
            if a > abn_th:
                anomalies.append((i, a))
        return anomalies, abn_th
