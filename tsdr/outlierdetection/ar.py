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

    def predict_both_of_sample(self, test_samples_size: int, dynamic: bool = False) -> tuple[np.ndarray, float]:
        pred_results: PredictionResults = self._fit_model.get_prediction(
            start=0,
            end=self._samples.size+test_samples_size+1,
            dynamic=dynamic,
        )
        preds = pred_results.predicted_mean
        # remove the plots for the lag. And read through the first value because the prediction line is shifted by 1 plot for some reason.
        preds = preds[self._lag+1:]
        var = pred_results.var_pred_mean
        sig2: float = var[self._lag]
        if sig2 == 0:
            return np.empty([]), 0
        return preds, sig2

    def anomaly_scores_both_of_sample(self, test_samples: np.ndarray, dynamic: bool = False) -> tuple[np.ndarray,np.ndarray]:
        preds, sig2 = self.predict_both_of_sample(test_samples.size, dynamic)
        scores: np.ndarray = np.zeros(self._samples.size + test_samples.size, dtype=np.float32)
        if preds.size <= 1:
            return scores, preds
        actuals: np.ndarray = np.concatenate([self._samples, test_samples])[self._lag:]
        for i, (xi, pred) in enumerate(zip(actuals, preds)):
            scores[i] = (xi - pred) ** 2 / sig2
        return scores, preds

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
        # mean, var = scores.mean(), scores.var()
        # m_mo = 2*mean**2 / (var - mean**2)
        # s_mo = (var - mean**2) / 2*mean
        # if m_mo < 1:
        #     m_mo = 1
        # if s_mo < 1:
        #     s_mo = 1
        abn_th = chi2.interval(alpha=1-threshold, df=1, scale=1)[1]
        anomalies: list[tuple[int, float]] = []
        for i, a in enumerate(scores):
            if a > abn_th:
                anomalies.append((i, a))
        return anomalies, abn_th

    @classmethod
    def detect_by_fitting_gaussian(
        cls,
        scores: np.ndarray,
        sigma: int = 2,
    ) -> list[tuple[int, float]]:
        mean = scores.mean()
        lower, upper = mean - sigma * scores.std(), scores.mean() + sigma * scores.std()
        anomalies: list[tuple[int, float]] = []
        for (i, v) in enumerate(scores):
            if v <= lower or v >= upper:
                anomalies.append((i, v))
        return anomalies

    @classmethod
    def detect_by_mse_gaussian(
        cls,
        scores: np.ndarray,
        sigma: int = 2,
    ) -> list[tuple[int, float]]:
        mses = np.array([np.sum(scores[:i]) / i for i, score in enumerate(scores, start=1)])
        mse_mean = mses.mean()
        mse_std = mses.std()
        lower, upper = mse_mean - sigma * mse_std, mse_mean + sigma * mse_std
        anomalies: list[tuple[int, float]] = []
        for (i, v) in enumerate(mses):
            if v <= lower or v >= upper:
                anomalies.append((i, v))
        return anomalies
