import random
import warnings

import numpy as np
import sklearn.mixture


class FluxInferAD(object):
    def __init__(self, x: np.ndarray):
        self.x: np.ndarray = x

    def smooth(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        x_: np.ndarray = np.copy(self.x)

        while True:
            reshaped_x = x_.reshape(-1, 1)
            with warnings.catch_warnings():
                # Supress sklearn's warning about covariance matrix being singular
                # 'ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (2). Possibly due to duplicate points in X.'
                warnings.simplefilter("ignore")
                labels = sklearn.mixture.GaussianMixture(n_components=2).fit(reshaped_x).predict(reshaped_x)

            # Calculate segmentation boundaries.
            seg_bounds: np.ndarray = np.argwhere(np.abs(np.diff(labels)) == 1).flatten() + 1
            segs: list[np.ndarray] = np.split(x_, indices_or_sections=seg_bounds)
            k: int = len(segs)
            if k <= 2:
                return x_, segs, seg_bounds

            global_i = len(segs[0])
            changed = False
            for j in range(1, len(segs)-1):  # skip the first and last segment
                if len(segs[j]) < len(segs[j+1]):
                    for i in range(len(segs[j])):
                        v = random.choice(list(segs[j+1]))
                        segs[j][i], x_[global_i+i] = v, v
                        changed = True
                global_i += len(segs[j])

            if not changed:
                return x_, segs, seg_bounds

    def detect_anomaly(self, sigma: int = 3) -> bool:
        segs = self.smooth()[1]
        k = len(segs)
        if k == 0:
            return False
        mean_k2, std_k2 = np.mean(segs[k-2]), np.std(segs[k-2])
        if std_k2 == 0:
            return False
        zscores = np.frompyfunc(lambda x: (x - mean_k2) / std_k2, 1, 1)(segs[k-1])
        return np.abs(np.mean(zscores)) > sigma*np.std(zscores)
