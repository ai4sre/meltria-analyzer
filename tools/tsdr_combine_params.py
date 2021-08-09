#!/usr/bin/env python3

import argparse
import json
import os
import sys
from multiprocessing import cpu_count

sys.path.append(os.path.dirname(__file__) + "/../tsdr")
import tsdr

sys.path.append(os.path.dirname(__file__) + "/../lib")
from metrics import check_cause_metrics

DIST_THRESHOLDS = [0.001, 0.01, 0.1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    args = parser.parse_args()

    summary = {}
    for metrics_file in args.metricsfiles:
        # read metric
        data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file)
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        for thresh in DIST_THRESHOLDS:
            _, reduced_df, _, _ = tsdr.run_tsdr(
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
            key = f"{chaos_type}:{chaos_comp}"
            summary.setdefault(key, {})
            summary[key][thresh] = ok

    json.dump(summary, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
