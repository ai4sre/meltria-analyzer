from pprint import pp, pprint

import numpy as np
from tsdr.outlierdetection.ar import AROutlierDetector


def test_ar_outlier_detector_anomaly_scores():
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 22.2, 9.0, 12.1])
    ar = AROutlierDetector(input)
    ar.fit(regression='n', autolag=True, ic='bic')
    got = ar.anomaly_scores()
    expected = np.array(
        [
            0.000000e+00, 9.473844e-03, 2.368461e-01, 9.473844e-03,
            2.368461e-03, 5.921152e-04, 1.515815e-03, 5.329037e-03,
            5.921152e-02, 5.456934e-02, 1.167272e+00, 1.918453e-01,
            0.000000e+00,
        ], dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-06)


def test_ar_outlier_detector_detect_by_fitting_dist():
    input = np.array([1, 2, 10, 2, 1, 0.5, 0.8, 1.5, 5.0, 4.8, 42.2, 9.0, 12.1])
    ar = AROutlierDetector(input)
    ar.fit(regression='n', autolag=True, ic='bic')
    got = ar.anomaly_scores()
    anomalies, abn_th = AROutlierDetector.detect_by_fitting_dist(got, 0.01)
    assert len(anomalies) == 1
