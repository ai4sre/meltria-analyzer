import numpy as np
from tsdr.outlierdetection.ar import AROutlierDetector


def test_ar_outlier_detector_score():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    assert ar.score(input)[1:] == [np.nan, 0.08986889267696001, 3.7183484546650334, 3.8075135043425483, 0.6158008427697705, 0.5654338656623322, 0.32463645172583305, 0.17123483665528313, 0.26297645945662845, 0.03599698801645843, 0.3254092017134403, 0.4524605268734048, 1.6303199754423074][1:]


def test_ar_outlier_detector_find_anomalies():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    anomalies = ar.find_anomalies(input, 1.5)
    assert len(anomalies) == 3
