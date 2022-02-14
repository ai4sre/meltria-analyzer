from pprint import pp, pprint

import numpy as np
from tsdr.outlierdetection.ar import AROutlierDetector


def test_ar_outlier_detector_score():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    got = ar.score(input)[0]
    print(got)
    expected = np.array(
        [
            0.0, 0.0898689, 3.7183485, 3.8075135, 0.61580086, 0.56543386,
            0.32463646, 0.17123483, 0.26297647, 0.03599699, 0.3254092, 0.45246053, 1.63032,
        ], dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-06)


def test_ar_outlier_detector_detect_by_fitting_dist():
    ar = AROutlierDetector()
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 7.2, 9.0, 12.1])
    anomalies = ar.detect_by_fitting_dist(input, 0.01)
    assert len(anomalies) == 3
