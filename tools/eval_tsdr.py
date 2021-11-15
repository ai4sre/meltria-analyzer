#!/usr/bin/env python3

import argparse
import logging
import os
import statistics
from collections import defaultdict
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import neptune.new as neptune
import pandas as pd
from lib.metrics import check_tsdr_ground_truth_by_route
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tsdr import tsdr

# algorithms
STEP1_METHODS = ['df', 'adf']


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
    parser.add_argument('--step1-alpha',
                        type=float,
                        default=0.01,
                        help='sigificance levels for step1 test')
    parser.add_argument('--dist-threshold',
                        type=float,
                        default=0.001,
                        help='distance thresholds')
    parser.add_argument('--out', help='output file path')
    parser.add_argument('--neptune-mode',
                        choices=['async', 'offline', 'debug'],
                        default='async',
                        help='specify neptune mode')
    args = parser.parse_args()

    # Setup neptune.ai client
    run = neptune.init(mode=args.neptune_mode)
    run['dataset/id'] = args.dataset_id
    run['dataset/num_metrics_files'] = len(args.metricsfiles)
    run['parameters'] = {
        'step1_model': args.step1_method,
        'step1_alpha': args.step1_alpha,
        'step2_dist_threshold': args.dist_threshold,
    }

    dataset = pd.DataFrame()
    for metrics_file in args.metricsfiles:
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

    scores_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'step',
                 'tn', 'fp', 'fn', 'tp', 'accuracy', 'recall',
                 'reduction_rate'],
        index=['chaos_type', 'chaos_comp', 'step']
    ).dropna()
    tests_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'metrics_file', 'step', 'ok', 'found_metrics'],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'step'],
    ).dropna()

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        case = f"{chaos_type}:{chaos_comp}"
        y_true_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        y_pred_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        reductions: dict[str, list[float]] = defaultdict(lambda: list())

        for (metrics_file), data_df in sub_df.groupby(level=2):
            logging.info(f">> Running tsdr {metrics_file} {case} ...")

            elapsedTime, reduced_df_by_step, metrics_dimension, _ = tsdr.run_tsdr(
                data_df=data_df,
                method=tsdr.TSIFTER_METHOD,
                max_workers=cpu_count(),
                tsifter_step1_method=args.step1_method,
                tsifter_step1_alpha=args.step1_alpha,
                tsifter_clustering_threshold=args.dist_threshold,
            )

            series_num: dict[str, float] = {
                'total': metrics_dimension['total'][0],
                'step1': metrics_dimension['total'][1],
                'step2': metrics_dimension['total'][2],
            }

            step: str
            df: pd.DataFrame
            for step, df in reduced_df_by_step.items():
                ok, found_metrics = check_tsdr_ground_truth_by_route(
                    metrics=list(df.columns),
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                )
                y_true_by_step[step].append(1)
                y_pred_by_step[step].append(1 if ok else 0)
                reductions[step].append(1 - (series_num[step] / series_num['total']))
                tests_df = tests_df.append(
                    pd.Series(
                        [chaos_type, chaos_comp, metrics_file, step, ok, ','.join(found_metrics)],
                        index=tests_df.columns,
                    ),
                    ignore_index=True,
                )

                # upload found_metrics plot images to neptune.ai
                if len(found_metrics) < 1:
                    continue
                found_metrics.sort()
                fig, axes = plt.subplots(nrows=len(found_metrics), ncols=1)
                # reset_index removes extra index texts from the generated figure.
                df[found_metrics].reset_index().plot(subplots=True, figsize=(6, 6), sharex=False, ax=axes)
                fig.suptitle(f"{chaos_type}_{chaos_comp}_{metrics_file}")
                run['tests/figures'].log(neptune.types.File.as_image(fig))
                plt.close(fig=fig)

        for step, y_true in y_true_by_step.items():
            y_pred = y_pred_by_step[step]
            tn, fp, fn, tp = confusion_matrix(
                y_true=y_true, y_pred=y_pred, labels=[0, 1],
            ).ravel()
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            reduction_rate = statistics.mean(reductions[step])
            scores_df = scores_df.append(
                pd.Series(
                    [chaos_type, chaos_comp, step, tn, fp, fn, tp, accuracy, recall, reduction_rate],
                    index=scores_df.columns,
                ), ignore_index=True,
            )

    run['tests/table'].upload(neptune.types.File.as_html(tests_df))

    tn = scores_df['tn'].sum()
    fp = scores_df['fp'].sum()
    fn = scores_df['fn'].sum()
    tp = scores_df['tp'].sum()
    run['scores/tn'] = tn
    run['scores/fp'] = fp
    run['scores/fn'] = fn
    run['scores/tp'] = tp
    run['scores/accuracy'] = (tp + tn) / (tn + fp + fn + tp)
    run['scores/reduction_rate'] = {
        'mean': scores_df['reduction_rate'].mean(),
        'max': scores_df['reduction_rate'].max(),
        'min': scores_df['reduction_rate'].min(),
    }
    run['scores/table'].upload(neptune.types.File.as_html(scores_df))
    df_by_chaos_type = scores_df.groupby(['chaos_type', 'step']).agg(
        {
            'tn': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'tp': 'sum',
            'reduction_rate': 'mean',
        },
    )
    df_by_chaos_type['accuracy'] = (df_by_chaos_type['tp'] + df_by_chaos_type['tn']) / (df_by_chaos_type['tn'] + df_by_chaos_type['fp'] + df_by_chaos_type['fn'] + df_by_chaos_type['tp'])
    run['scores/table_grouped_by_chaos_type'] = df_by_chaos_type

    df_by_chaos_comp = scores_df.groupby(['chaos_comp', 'step']).agg(
        {
            'tn': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'tp': 'sum',
            'reduction_rate': 'mean',
        },
    )
    df_by_chaos_comp['accuracy'] = (df_by_chaos_comp['tp'] + df_by_chaos_comp['tn']) / (df_by_chaos_comp['tn'] + df_by_chaos_comp['fp'] + df_by_chaos_comp['fn'] + df_by_chaos_comp['tp'])
    run['scores/table_grouped_by_chaos_comp'] = df_by_chaos_comp

    run.stop()


if __name__ == '__main__':
    main()
