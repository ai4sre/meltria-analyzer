#!/usr/bin/env python3

import argparse
import logging
import os
import statistics
from collections import defaultdict
from multiprocessing import cpu_count

import neptune.new as neptune
import pandas as pd
from lib.metrics import check_tsdr_ground_truth_by_route
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tsdr import tsdr

# algorithms
STEP1_METHODS = ['df', 'adf']

# parameters
DIST_THRESHOLDS = [0.001, 0.01, 0.1]
STEP1_ALPHAS = [0.01, 0.02, 0.05]


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    parser.add_argument("--dataset-id",
                        type=str,
                        help='dataset id like "b2qdj"')
    parser.add_argument('--step1-method',
                        default='df',
                        choices=STEP1_METHODS,
                        help='step1 method')
    parser.add_argument('--dist-thresholds',
                        default=DIST_THRESHOLDS,
                        type=lambda s: [float(i) for i in s.split(',')],
                        help='distance thresholds')
    parser.add_argument('--step1-alphas',
                        default=STEP1_ALPHAS,
                        type=lambda s: [float(i) for i in s.split(',')],
                        help='sigificance levels for step1 test')
    parser.add_argument('--out', help='output file path')
    parser.add_argument('--neptune-mode',
                        choices=['online', 'offline', 'debug'],
                        default='online',
                        help='specify neptune mode')
    args = parser.parse_args()

    run = neptune.init(mode=args.neptune_mode)
    run['dataset/id'] = args.dataset_id

    dataset = pd.DataFrame()
    for metrics_file in args.metricsfiles:
        # https://docs.neptune.ai/api-reference/neptune#.init

        try:
            data_df, _, metrics_meta = tsdr.read_metrics_json(metrics_file)
        except ValueError as e:
            logging.warning(f">> Skip {metrics_file} because of {e}")
            continue
        chaos_type: str = metrics_meta['injected_chaos_type']
        chaos_comp: str = metrics_meta['chaos_injected_component']
        data_df['chaos_type'] = chaos_type
        data_df['chaos_comp'] = chaos_comp
        data_df['metrics_file'] = os.path.basename(metrics_file)
        dataset = dataset.append(data_df)

    dataset.set_index(['chaos_type', 'chaos_comp', 'metrics_file'], inplace=True)

    for alpha in args.step1_alphas:
        for thresh in args.dist_thresholds:
            run['parameters'] = {
                'step1_model': args.step1_method,
                'step1_alpha': alpha,
                'step2_dist_threshold': thresh,
            }

            scores_df = pd.DataFrame()

            for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
                case = f"{chaos_type}:{chaos_comp}"
                param_key = f"step1_alpha:{alpha},dist_threshold:{thresh}"

                y_true_by_step: dict[str, list[int]] = defaultdict(lambda: list())
                y_pred_by_step: dict[str, list[int]] = defaultdict(lambda: list())
                reductions: dict[str, list[float]] = defaultdict(lambda: list())

                for (metrics_file), data_df in sub_df.groupby(level=0):
                    logging.info(f">> Running tsdr {metrics_file} {case} {param_key} ...")

                    elapsedTime, reduced_df_by_step, metrics_dimension, _ = tsdr.run_tsdr(
                        data_df=data_df,
                        method=tsdr.TSIFTER_METHOD,
                        max_workers=cpu_count(),
                        tsifter_step1_method=args.step1_method,
                        tsifter_step1_alpha=alpha,
                        tsifter_clustering_threshold=thresh,
                    )

                    series_num: dict[str, float] = {
                        'total': metrics_dimension['total'][0],
                        'step1': metrics_dimension['total'][1],
                        'step2': metrics_dimension['total'][2],
                    }

                    for step, df in reduced_df_by_step.items():
                        ok, found_metrics = check_tsdr_ground_truth_by_route(
                            metrics=list(df.columns),
                            chaos_type=chaos_type,
                            chaos_comp=chaos_comp,
                        )
                        y_true_by_step[step].append(1)
                        y_pred_by_step[step].append(1 if ok else 0)
                        reductions[step].append(1 - (series_num[step] / series_num['total']))

                for step, y_true in y_true_by_step.items():
                    y_pred = y_pred_by_step[step]
                    tn, fp, fn, tp = confusion_matrix(
                        y_true=y_true, y_pred=y_pred, labels=[0, 1],
                    ).ravel()
                    accuracy = accuracy_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    reduction_rate = statistics.mean(reductions[step])
                    scores = {
                        'tn': tn,
                        'fp': fp,
                        'fn': fn,
                        'tp': tp,
                        'accuracy': accuracy,
                        'recall': recall,
                        'reduction_rate': reduction_rate,
                    }
                    label = {
                        'chaos_type': chaos_type,
                        'chaos_comp': chaos_comp,
                        'step': step,
                    }
                    meta_key = f"data/{chaos_type}/{chaos_comp}/{step}/scores"
                    for k, v in scores.items():
                        run[meta_key+'/'+k] = v
                    scores_df = scores_df.append(
                        pd.DataFrame(dict(label, **scores), index=['chaos_type', 'chaos_comp']))

            # multiindexing
            scores_df.set_index(['chaos_type', 'chaos_comp'], inplace=True)
            # TODO: aggregate scores by chaos cases
            tn = scores_df['tn'].sum()
            fp = scores_df['fp'].sum()
            fn = scores_df['fn'].sum()
            tp = scores_df['tp'].sum()
            run['scores/tn'] = tn
            run['scores/fp'] = fp
            run['scores/fn'] = fn
            run['scores/tp'] = tp
            run['scores/accuracy'] = (tp + tn) / (tn + fp + fn + tp)
            run['scores/reduction_rate'] = scores_df['reduction_rate'].mean()


if __name__ == '__main__':
    main()
