#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import cpu_count

from lib.metrics import check_cause_metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from tsdr import tsdr

DIST_THRESHOLDS = [0.001, 0.01, 0.1]
ADF_ALPHAS = [0.01, 0.02, 0.05]


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    args = parser.parse_args()

    y_trues = defaultdict(list)
    y_preds = defaultdict(list)
    results = defaultdict(lambda: defaultdict(dict))
    for metrics_file in args.metricsfiles:
        data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file)
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        for alpha in ADF_ALPHAS:
            for thresh in DIST_THRESHOLDS:
                key = f"{chaos_type}:{chaos_comp}"

                logging.info(f">> Running tsdr {metrics_file} [{key}] dist_threshold:{thresh} ...")

                elapsedTime, reduced_df, metrics_dimension, _ = tsdr.run_tsdr(
                    data_df=data_df,
                    method=tsdr.TSIFTER_METHOD,
                    max_workers=cpu_count(),
                    tsifter_adf_alpha=alpha,
                    tsifter_clustering_threshold=thresh,
                )
                ok, _ = check_cause_metrics(
                    metrics=list(reduced_df.columns),
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                )
                param_key = f"adf_alpha:{alpha},dist_threshold:{thresh}"
                y_trues[param_key].append(1)
                y_preds[param_key].append(1 if ok else 0)
                results[key][param_key] = {
                    'found_cause': ok,
                    'reduction_performance': {
                        'reduced_series_num': {
                            'step0': metrics_dimension['total'][0],
                            'step1': metrics_dimension['total'][1],
                            'step2': metrics_dimension['total'][2],
                        },
                    },
                    'execution_time': round(elapsedTime['step1'] + elapsedTime['step2'], 2),
                }

    for alpha in ADF_ALPHAS:
        for thresh in DIST_THRESHOLDS:
            param_key = f"adf_alpha:{alpha},dist_threshold:{thresh}"
            y_true, y_pred = y_trues[param_key], y_preds[param_key]
            tn, fp, fn, tp = confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=[0, 1],
            ).ravel()
            results['evaluation'][param_key] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
            }

    json.dump(results, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
