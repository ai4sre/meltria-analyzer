import numpy as np
from tsdr.outlierdetection.knn import KNNOutlierDetector


def test_knn_outlier_detector_score():
    knn = KNNOutlierDetector(w=2, k=1)
    input = np.array([1, 2, 10, 2, 1])
    assert knn.score(input)[1:] == [np.nan, 1.4142135623730951, 8.06225774829855, 8.06225774829855, 1.4142135623730951][1:]


def test_knn_outlier_detector_find_anomalies():
    knn = KNNOutlierDetector(w=2, k=1)
    input = np.array([1, 2, 10, 2, 1])
    anomalies = knn.find_anomalies(input, 3)
    assert len(anomalies) == 2
