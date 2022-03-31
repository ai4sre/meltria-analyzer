import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.ar_model import (AutoReg, AutoRegResultsWrapper,
                                      ar_select_order)


class AROutlierDetector:
    maxlag: int

    def __init__(self, maxlag: int = 0):
        self.maxlag = maxlag

    def score(
        self,
        x: np.ndarray,
        actuals: np.ndarray = None,
        regression: str = 'c',
        autolag: bool = True,
        ic: str = 'aic',
        lag: int = 0,
        dynamic_prediction: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, AutoRegResultsWrapper]:
        """
        Estimate the anomaly scores for the datapoints in x.

        Parameters
        ----------

        Returns
        -------
            numpy.ndarray
                Anomaly scores
            numpy.ndarray
                Predicted values
            AutoRegResults
                Learned Model
        """

        r: int = lag
        if autolag:
            maxlag = int(x.size * 0.2) if self.maxlag == 0 else self.maxlag
            sel = ar_select_order(x, maxlag=maxlag, trend=regression, ic=ic, old_names=False)
            model_fit = sel.model.fit()
            if model_fit.ar_lags is not None and len(model_fit.ar_lags) > 0:
                r = model_fit.ar_lags[-1]
        else:
            model = AutoReg(endog=x, lags=lag, trend=regression, old_names=False)
            model_fit = model.fit()

        sig2 = model_fit.sigma2
        if sig2 == 0:
            return [], [], model_fit
        if actuals is None:
            preds: np.ndarray = model_fit.get_prediction(dynamic=dynamic_prediction).predicted_mean
        else:
            preds: np.ndarray = model_fit.get_prediction(end=len(actuals)-1, dynamic=dynamic_prediction).predicted_mean
        if actuals is None:
            scores: np.ndarray = np.zeros(x.size, dtype=np.float32)
            for i, (xi, pred) in enumerate(zip(x[r:], preds[r:])):
                scores[r + i] = (xi - pred) ** 2 / sig2
        else:
            scores: np.ndarray = np.zeros(actuals.size, dtype=np.float32)
            for i, (xi, pred) in enumerate(zip(actuals, preds[r:])):
                scores[r + i] = (xi - pred) ** 2 / sig2
        return scores, preds, model_fit

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
