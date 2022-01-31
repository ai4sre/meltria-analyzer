import numpy as np
import pytest

from tests.testseries.sockshop import testcases_of_sockshop
from tsdr import tsdr


@pytest.mark.parametrize("take_log", [True, False])
@pytest.mark.parametrize("unit_root_model", ['adf', 'pp'])
@pytest.mark.parametrize("unit_root_alpha", [0.01])
@pytest.mark.parametrize("unit_root_regression", ['c', 'ct'])
@pytest.mark.parametrize("post_cv", [True, False])
@pytest.mark.parametrize("cv_threshold", [0.1, 0.5])
@pytest.mark.parametrize("post_knn", [True, False])
@pytest.mark.parametrize("knn_threshold", [0.01])
def test_unit_root_based_model(
    take_log,
    unit_root_model,
    unit_root_alpha,
    unit_root_regression,
    post_cv,
    cv_threshold,
    post_knn,
    knn_threshold,
):
    gots: dict[str, bool] = {}
    for case in testcases_of_sockshop:
        got: bool = tsdr.unit_root_based_model(
            series=np.array(case['datapoints']),
            tsifter_step1_take_log=take_log,
            tsifter_step1_unit_root_model=unit_root_model,
            tsifter_step1_unit_root_alpla=unit_root_alpha,
            tsifter_step1_unit_root_regression=unit_root_regression,
            tsifter_step1_post_cv=post_cv,
            tsifter_step1_cv_threshold=cv_threshold,
            tsifter_step1_post_knn=post_knn,
            tsifter_step1_knn_threshold=knn_threshold,
        )
        gots[case['name']] = (got == case['is_unstationality'])
    assert not (False in gots.values())


@pytest.mark.parametrize("ar_regression", ['c', 'ct'])
@pytest.mark.parametrize("ar_anomaly_score_threshold", [10, 50, 80])
@pytest.mark.parametrize("cv_threshold", [0.1, 0.5])
def test_ar_based_ad_model(
    ar_regression,
    ar_anomaly_score_threshold,
    cv_threshold,
):
    gots: dict[str, bool] = {}
    for case in testcases_of_sockshop:
        got: bool = tsdr.ar_based_ad_model(
            series=np.array(case['datapoints']),
            tsifter_step1_ar_regression=ar_regression,
            tsifter_step1_ar_anomaly_score_threshold=ar_anomaly_score_threshold,
            tsifter_step1_cv_threshold=cv_threshold,
        )
        gots[case['name']] = (got == case['is_unstationality'])
    assert not (False in gots.values())
