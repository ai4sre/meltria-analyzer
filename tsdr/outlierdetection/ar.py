import numpy as np
from statsmodels.tsa.ar_model import ar_select_order


class AROutlierDetector:
    maxlag: int

    def __init__(self, maxlag: int = 0):
        self.maxlag = maxlag

    def score(self, x: np.ndarray, regression: str = 'c', ic: str = 'aic', include_nan: bool = False) -> list[float]:
        maxlag = int(x.size * 0.2) if self.maxlag == 0 else self.maxlag
        sel = ar_select_order(x, maxlag=maxlag, trend=regression, ic=ic, old_names=False)
        model_fit = sel.model.fit()
        r: int = 0
        if model_fit.ar_lags is None or len(model_fit.ar_lags) > 0:
            r = model_fit.ar_lags[-1]
        sig2 = model_fit.sigma2

        pred = model_fit.get_prediction()
        preds = pred.summary_frame()[r:]['mean'].to_numpy()

        scores: list[float] = []
        for i, xi in enumerate(x[r:]):
            if i >= preds.size:
                break
            scores.append((xi - preds[i]) ** 2 / sig2)
        return [np.nan * r] + scores if include_nan else scores

    def find_anomalies(self, x: np.ndarray, threshold: float) -> list[float]:
        return [s for s in self.score(x) if s >= threshold]
