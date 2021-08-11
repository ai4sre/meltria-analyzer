#!/usr/bin/env python3

import argparse
import json
import logging
import statistics
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
    parser.add_argument('--dist-thresholds',
                        default=DIST_THRESHOLDS,
                        type=lambda s: [float(i) for i in s.split(',')],
                        help='distance thresholds')
    parser.add_argument('--adf-alphas',
                        default=ADF_ALPHAS,
                        type=lambda s: [float(i) for i in s.split(',')],
                        help='sigificance levels for ADF test')
    parser.add_argument('--out', help='output file path')
    args = parser.parse_args()

    y_trues = defaultdict(lambda: {
        'step1': [],
        'step2': [],
    })
    y_preds = defaultdict(lambda: {
        'step1': [],
        'step2': [],
    })
    reductions = defaultdict(lambda: {
        'step1': [],
        'step2': [],
    })
    results = defaultdict(lambda: defaultdict(dict))
    for metrics_file in args.metricsfiles:
        data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file)
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        for alpha in args.adf_alphas:
            for thresh in args.dist_thresholds:
                key = f"{chaos_type}:{chaos_comp}"

                logging.info(f">> Running tsdr {metrics_file} [{key}] dist_threshold:{thresh} ...")

                elapsedTime, reduced_df_by_step, metrics_dimension, _ = tsdr.run_tsdr(
                    data_df=data_df,
                    method=tsdr.TSIFTER_METHOD,
                    max_workers=cpu_count(),
                    tsifter_adf_alpha=alpha,
                    tsifter_clustering_threshold=thresh,
                )

                param_key = f"adf_alpha:{alpha},dist_threshold:{thresh}"

                has_cause_metrics = {'step1': False, 'step2': False}
                for step, df in reduced_df_by_step.items():
                    ok, _ = check_cause_metrics(
                        metrics=list(df.columns),
                        chaos_type=chaos_type,
                        chaos_comp=chaos_comp,
                    )
                    has_cause_metrics[step] = ok
                    y_trues[param_key][step].append(1)
                    y_preds[param_key][step].append(1 if ok else 0)

                series_num: int = metrics_dimension['total'][0]
                step1_series_num: int = metrics_dimension['total'][1]
                step2_series_num: int = metrics_dimension['total'][2]
                results[key][param_key] = {
                    'found_cause': has_cause_metrics,
                    'reduction_performance': {
                        'reduced_series_num': {
                            'step0': series_num,
                            'step1': step1_series_num,
                            'step2': step2_series_num,
                        },
                    },
                    'execution_time': round(elapsedTime['step1'] + elapsedTime['step2'], 2),
                }

                reductions[param_key]['step1'].append(1 - (step1_series_num / series_num))
                reductions[param_key]['step2'].append(1 - (step2_series_num / series_num))

    for alpha in args.adf_alphas:
        for thresh in args.dist_thresholds:
            param_key = f"adf_alpha:{alpha},dist_threshold:{thresh}"
            for step in ['step1', 'step2']:
                y_true, y_pred = y_trues[param_key][step], y_preds[param_key][step]
                tn, fp, fn, tp = confusion_matrix(
                    y_true=y_true, y_pred=y_pred, labels=[0, 1],
                ).ravel()
                results['evaluation'][param_key][step] = {
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred),
                    'reduction_rate': statistics.mean(reductions[param_key][step]),
                }

    if args.out is None:
        json.dump(results, sys.stdout, indent=4)
    else:
        with open(args.out, mode='w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()
