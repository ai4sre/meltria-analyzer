import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.ar_model import (AutoReg, AutoRegResultsWrapper,
                                      ar_select_order)
from statsmodels.tsa.base.prediction import PredictionResults


class AROutlierDetector:
    def __init__(self, samples: np.ndarray, maxlag: int = 0):
        self._samples = samples
        self._maxlag = int(self._samples.size * 0.2) if maxlag == 0 else maxlag
        self._model: AutoReg = None
        self._fit_model: AutoRegResultsWrapper = None
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
            self._model = sel.model
            self._fit_model = sel.model.fit()
            if self._fit_model.ar_lags is not None and len(self._fit_model.ar_lags) > 0:
                self._lag = self._model.ar_lags[-1]
        else:
            self._lag = lag
            self._model = AutoReg(endog=self._samples, lags=lag, trend=regression)
            self._fit_model = self._model.fit()

    def predict_in_sample(self) -> tuple[np.ndarray, float]:
        pred_results: PredictionResults = self._fit_model.get_prediction()
        preds = pred_results.predicted_mean
        # remove the plots for the lag. And read through the first value because the prediction line is shifted by 1 plot for some reason.
        preds = preds[self._lag+1:]
        var = pred_results.var_pred_mean
        sig2: float = var[self._lag]
        if sig2 == 0:
            return np.empty([]), 0
        return preds, sig2

    def predict_out_of_sample(self, test_sample_size: int, dynamic: bool = False) -> tuple[np.ndarray, float]:
        preds = self._fit_model.forecast(steps=test_sample_size)
        sig2: float = self._fit_model.sigma2  # var[self._lag]
        if sig2 == 0:
            return np.empty([]), 0
        return preds, sig2

    def anomaly_scores_in_sample(self) -> np.ndarray:
        preds, sig2 = self.predict_in_sample()
        scores: np.ndarray = np.zeros(self._samples.size, dtype=np.float32)
        for i, (xi, pred) in enumerate(zip(self._samples[self._lag:], preds)):
            scores[self._lag+i] = (xi - pred) ** 2 / sig2
        return scores

    def anomaly_scores_out_of_sample(self, test_samples: np.ndarray, dynamic=False) -> np.ndarray:
        preds, sig2 = self.predict_out_of_sample(test_samples.size, dynamic)
        scores: np.ndarray = np.zeros(test_samples.size, dtype=np.float32)
        if preds.size <= 1:
            return scores
        for i, (xi, pred) in enumerate(zip(test_samples, preds)):
            scores[i] = (xi - pred) ** 2 / sig2
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
