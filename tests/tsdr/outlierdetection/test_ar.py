from pprint import pp, pprint

import numpy as np
from tsdr.outlierdetection.ar import AROutlierDetector


def test_ar_outlier_detector_score():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    got = ar.score(input)[0]
    expected = np.array(
        [
            0.0, 0.23356132, 0.21187489, 0.23356132, 0.36147678,
            0.43586820, 0.39039865, 0.29404113, 0.01675449, 0.02341777,
            0.01691183, 0.11720429, 0.0,
        ], dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-06)


def test_ar_outlier_detector_detect_by_fitting_dist():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    anomalies = ar.detect_by_fitting_dist(input, 0.01)
    assert len(anomalies) == 3
