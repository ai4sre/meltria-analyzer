#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import cpu_count

from lib.metrics import check_cause_metrics
from tsdr import tsdr

DIST_THRESHOLDS = [0.001, 0.01, 0.1]


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    args = parser.parse_args()

    results = defaultdict(lambda: defaultdict(
        lambda: defaultdict(dict)
    ))
    for metrics_file in args.metricsfiles:
        data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file)
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        for thresh in DIST_THRESHOLDS:
            key = f"{chaos_type}:{chaos_comp}"

            logging.info(f">> Running tsdr {metrics_file} [{key}] dict_threshold:{thresh} ...")

            elapsedTime, reduced_df, _, _ = tsdr.run_tsdr(
                data_df=data_df,
                method=tsdr.TSIFTER_METHOD,
                max_workers=cpu_count(),
                tsifter_adf_alpha=tsdr.SIGNIFICANCE_LEVEL,
                tsifter_clustering_threshold=thresh,
            )
            ok, _ = check_cause_metrics(
                metrics=list(reduced_df.columns),
                chaos_type=chaos_type,
                chaos_comp=chaos_comp,
            )
            results[key]['dict_threshold'][thresh] = {
                'found_cause': ok,
                'reduction_performance': {'reduced_series_num': len(reduced_df.columns)},
                'execution_time': round(elapsedTime['step1'] + elapsedTime['step2'], 2),
            }

    json.dump(results, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
