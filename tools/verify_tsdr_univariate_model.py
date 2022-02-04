#!/usr/bin/env python3

import itertools

import numpy as np
from tsdr import tsdr
from tsdr.testseries.sockshop import testcases_of_sockshop


class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    RESET = '\033[0m'


def verify_unit_root_test_model():
    passed_items: list[tuple] = []
    for item in itertools.product(
        [True, False],  # take_log,
        ['adf', 'pp'],  # unit_root_model
        [0.01],         # unit_root_alpha
        ['c', 'ct'],    # unit_root_regression
        [True, False],  # post_cvs
        [0.1, 0.5],     # cv_threshold
        [True, False],  # post_knn
        [0.01],         # knn_threshold
    ):
        take_log, \
            unit_root_model, \
            unit_root_alpha, \
            unit_root_regression, \
            post_cv, \
            cv_threshold, \
            post_knn, \
            knn_threshold = item
        gots: dict[str, bool] = {}
        for case in testcases_of_sockshop:
            got: bool = tsdr.unit_root_based_model(
                series=np.array(case['datapoints']),
                tsifter_step1_pre_cv=post_cv,
                tsifter_step1_cv_threshold=cv_threshold,
                tsifter_step1_take_log=take_log,
                tsifter_step1_unit_root_model=unit_root_model,
                tsifter_step1_unit_root_alpla=unit_root_alpha,
                tsifter_step1_unit_root_regression=unit_root_regression,
                tsifter_step1_post_knn=post_knn,
                tsifter_step1_knn_threshold=knn_threshold,
            )
            gots[case['name']] = (got == case['is_unstationality'])
        if (False in gots.values()):
            for k, v in gots.items():
                if v is False:
                    print(f"{Color.RED}REJECT:{Color.RESET} {k} params: {item}")
        else:
            passed_items.append(item)
    for item in passed_items:
        print(f"{Color.GREEN}PASS:{Color.RESET} params: {item}", item)


def verify_ar_based_ad_model():
    passed_items: list[tuple] = []
    for item in itertools.product(
        ['c', 'ct'],    # ar_regression
        [5, 10, 20],    # ar_anomaly_score_threshold
        [0.1, 0.5],     # cv_threshold
    ):
        ar_regression, \
            ar_anomaly_score_threshold, \
            cv_threshold = item
        gots: dict[str, bool] = {}
        for case in testcases_of_sockshop:
            got: bool = tsdr.ar_based_ad_model(
                series=np.array(case['datapoints']),
                tsifter_step1_ar_regression=ar_regression,
                tsifter_step1_ar_anomaly_score_threshold=ar_anomaly_score_threshold,
                tsifter_step1_cv_threshold=cv_threshold,
            )
            gots[case['name']] = (got == case['is_unstationality'])
        if (False in gots.values()):
            for k, v in gots.items():
                if v is False:
                    print(f"{Color.RED}REJECT:{Color.RESET} {k} params: {item}")
        else:
            passed_items.append(item)
    for item in passed_items:
        print(f"{Color.GREEN}PASS:{Color.RESET} params: {item}", item)


def main():
    print("--> unit_root_test")
    verify_unit_root_test_model()
    print("--> ar_based_ad")
    verify_ar_based_ad_model()


if __name__ == '__main__':
    main()
