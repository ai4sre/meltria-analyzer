import functools

import numpy as np


class KNNOutlierDetector:
    """
    Outlier Detector for time series data with kNN algorithm.

    >>> knn = KNNOutlierDetector(2, 1)
    >>> data = np.array([1, 2, 10, 2, 1])
    >>> knn.score(data)
    [0.0, 1.4142135623730951, 8.06225774829855, 8.06225774829855, 1.4142135623730951]
    """
    w: int = 0
    k: int = 1

    def __init__(self, w: int, k: int) -> None:
        self.w = w
        self.k = k

    def score(self, data: np.ndarray) -> list[float]:
        windows: list[np.ndarray] = self.sliding_windows(data)
        scores: list[float] = []
        for t in range(len(windows)):
            distances: list[float] = []
            for window in windows:
                distances.append(self.dist(windows[t], window))
            distances.sort()
            s: float = functools.reduce(lambda x, y: x+y, distances[1:self.k+1], 0.0) / self.k
            scores.append(s)
        # Adjust the size of input/output list
        return [0.0] * (self.w-1) + scores

    def sliding_windows(self, data: np.ndarray) -> list[np.ndarray]:
        num: int = data.size - self.w + 1
        return [data[t:t+self.w] for t in range(num)]

    def dist(self, v1: np.ndarray, v2: np.ndarray) -> float:
        sum: float = 0.0
        for i, v in enumerate(v1):
            sum += (v - v2[i]) * (v - v2[i])
        return np.sqrt(sum)

    def has_anomaly(self, data: np.ndarray, threashold: float) -> bool:
        for s in self.score(data):
            if s > threashold:
                return True
        return False
