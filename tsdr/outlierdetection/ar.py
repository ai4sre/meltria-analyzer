import numpy as np
from statsmodels.tsa.ar_model import ar_select_order


class AROutlierDetector:
    maxlag: int

    def __init__(self, maxlag: int = 0):
        self.maxlag = maxlag

    def score(self, x: np.ndarray) -> list[float]:
        maxlag = int(x.size * 0.2) if self.maxlag == 0 else self.maxlag
        sel = ar_select_order(x, maxlag=maxlag, trend='ct', ic='aic')
        model_fit = sel.model.fit()
        r = 0 if model_fit.ar_lags is None else model_fit.ar_lags[-1]
        sig2 = model_fit.sigma2

        pred = model_fit.get_prediction()
        preds = pred.summary_frame()[r:]['mean']

        scores: list[float] = []
        for i, xi in enumerate(x):
            if i >= x.size - r:
                break
            scores.append((xi - preds[i+r]) ** 2 / sig2)
        return scores
