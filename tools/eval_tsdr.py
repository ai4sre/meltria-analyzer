#!/usr/bin/env python3

import argparse
import logging
import math
import os
import statistics
from collections import defaultdict
from concurrent import futures
from multiprocessing import cpu_count
from typing import Optional

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
from lib.metrics import check_tsdr_ground_truth_by_route
from neptune.new.integrations.python_logger import NeptuneHandler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tsdr import tsdr

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('root_experiment')
logger.setLevel(logging.INFO)

# algorithms
STEP1_METHODS = ['df', 'adf']


class DatasetRecord:
    """A record of dataset"""
    chaos_comp: str     # chaos-injected component
    chaos_type: str     # injected chaos type
    metrics_file: str   # path of metrics file
    data_df: pd.DataFrame

    def __init__(self, chaos_type: str, chaos_comp: str, metrics_file: str, data_df: pd.DataFrame):
        self.chaos_comp = chaos_comp
        self.chaos_type = chaos_type
        self.metrics_file = metrics_file
        self.data_df = data_df

    def chaos_case(self) -> str:
        return f"{self.chaos_comp}/{self.chaos_type}"

    def chaos_case_file(self) -> str:
        return f"{self.metrics_file} of {self.chaos_case()}"

    def metrics_names(self) -> list[str]:
        return list(self.data_df.columns)


def read_metrics_file(metrics_file: str,
                      exclude_middleware_metrics: bool = False) -> Optional[pd.DataFrame]:
    logger.info(f">> Loading metrics file {metrics_file} ...")
    try:
        data_df, _, metrics_meta = tsdr.read_metrics_json(
            metrics_file,
            exclude_middlewares=exclude_middleware_metrics,
        )
    except ValueError as e:
        logger.warning(f">> Skip {metrics_file} because of {e}")
        return None
    chaos_type: str = metrics_meta['injected_chaos_type']
    chaos_comp: str = metrics_meta['chaos_injected_component']
    data_df['chaos_type'] = chaos_type
    data_df['chaos_comp'] = chaos_comp
    data_df['metrics_file'] = os.path.basename(metrics_file)
    data_df['grafana_dashboard_url'] = metrics_meta['grafana_dashboard_url']
    return data_df


def load_dataset(metrics_files: list[str], exclude_middleware_metrics: bool = False) -> pd.DataFrame:
    dataset = pd.DataFrame()
    with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        future_list = []
        for metrics_file in metrics_files:
            future_list.append(executor.submit(read_metrics_file, metrics_file, exclude_middleware_metrics))
        for future in futures.as_completed(future_list):
            data_df = future.result()
            if data_df is not None:
                dataset = dataset.append(data_df)
    return dataset.set_index(['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url'])


def log_plots_as_image(run: neptune.Run, record: DatasetRecord) -> None:
    """ Upload found_metrics plot images to neptune.ai. """

    _, ground_truth_metrics = check_tsdr_ground_truth_by_route(
        metrics=record.metrics_names(),  # pre-reduced data frame
        chaos_type=record.chaos_type,
        chaos_comp=record.chaos_comp,
    )
    if len(ground_truth_metrics) < 1:
        return
    ground_truth_metrics.sort()
    fig, axes = plt.subplots(nrows=len(ground_truth_metrics), ncols=1)
    # reset_index removes extra index texts from the generated figure.
    record.data_df[ground_truth_metrics].reset_index().plot(subplots=True, figsize=(6, 6), sharex=False, ax=axes)
    fig.suptitle(record.chaos_case_file())
    run[f"dataset/figures/{record.chaos_case}"].log(neptune.types.File.as_image(fig))
    plt.close(fig=fig)


def log_clustering_plots_as_image(run: neptune.Run,
                                  rep_metric: str, sub_metrics: list[str],
                                  record: DatasetRecord) -> None:
    """ Upload clustered time series plots to neptune.ai """
    clustered_metrics: list[str] = [rep_metric] + sub_metrics
    fig, axes = plt.subplots(nrows=len(clustered_metrics), ncols=1)
    # reset_index removes extra index texts from the generated figure.
    record.data_df[clustered_metrics].reset_index().plot(
        subplots=True, figsize=(6, 6), sharex=False, ax=axes)
    fig.suptitle(
        f"{record.chaos_case_file()}    rep:{rep_metric}")
    run[f"tests/clustering/ts_figures/{record.chaos_case()}"].log(
        neptune.types.File.as_image(fig))
    plt.close(fig=fig)


def log_non_clustered_plots_as_image(run: neptune.Run,
                                     record: DatasetRecord,
                                     non_clustered_reduced_df: pd.DataFrame) -> None:
    """ Upload non-clustered time series plots to neptune.ai """

    num_non_clustered_plots = len(non_clustered_reduced_df.columns)
    fig, axes = plt.subplots(
        nrows=math.ceil(num_non_clustered_plots/6),
        ncols=6,
        figsize=(6*6, num_non_clustered_plots),
        squeeze=False,  # always return 2D-array axes
    )
    # Match the numbers axes and non-clustered columns
    axes = trim_axs(axes, num_non_clustered_plots)
    # reset_index removes extra index texts from the generated figure.
    non_clustered_reduced_df.reset_index().plot(
        subplots=True, figsize=(6, 6), sharex=False, ax=axes)
    fig.suptitle(
        f"{record.chaos_case_file()} - non-clustered metrics")
    run[f"tests/clustering/non_clustered_metrics_ts_figures/{record.chaos_case()}"].log(
        neptune.types.File.as_image(fig)
    )
    plt.close(fig=fig)


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def get_scores_by_index(scores_df: pd.DataFrame, indexes: list[str]) -> pd.DataFrame:
    df = scores_df.groupby(indexes).agg({
        'tn': 'sum',
        'fp': 'sum',
        'fn': 'sum',
        'tp': 'sum',
        'reduction_rate': 'mean',
    })
    df['accuracy'] = (df['tp'] + df['tn']) / (df['tn'] + df['fp'] + df['fn'] + df['tp'])
    return df


def eval_tsdr(run: neptune.Run, metrics_files: list[str]):
    dataset: pd.DataFrame = load_dataset(
        metrics_files,
        run['parameters']['exclude_middleware_metrics'].fetch(),  # The type mismatch should be fixed
    )
    logger.info("Dataset loading complete")

    clustering_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'metrics_file', 'representative_metric', 'sub_metrics'],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'representative_metric', 'sub_metrics'],
    ).dropna()
    non_clustered_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'metrics_file', 'non_clustered_metrics'],
        index=['chaos_type', 'chaos_comp', 'metrics_file'],
    ).dropna()
    scores_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'step',
                 'tn', 'fp', 'fn', 'tp', 'accuracy', 'recall',
                 'num_series', 'reduction_rate'],
        index=['chaos_type', 'chaos_comp', 'step']
    ).dropna()
    tests_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'metrics_file', 'step', 'ok', 'num_series', 'found_metrics', 'grafana_dashboard_url'],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url', 'step'],
    ).dropna()

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        y_true_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        y_pred_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        num_series: dict[str, list[float]] = defaultdict(lambda: list())

        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            logger.info(f">> Uploading plot figures of {record.chaos_case_file()} ...")
            log_plots_as_image(run, record)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            elapsedTime, reduced_df_by_step, metrics_dimension, clustering_info = tsdr.run_tsdr(
                data_df=data_df,
                method=tsdr.TSIFTER_METHOD,
                max_workers=cpu_count(),
                tsifter_step1_method=run['parameters']['step1_model'].fetch(),
                tsifter_step1_alpha=run['parameters']['step1_alpha'].fetch(),
                tsifter_clustering_threshold=run['parameters']['step2_dist_threshold'].fetch(),
            )

            num_series_each_step: dict[str, float] = {
                'total': metrics_dimension['total'][0],
                'step1': metrics_dimension['total'][1],
                'step2': metrics_dimension['total'][2],
            }
            num_series['total'].append(num_series_each_step['total'])
            num_series_str: str = '/'.join(
                [f"{num_series_each_step[s]}" for s in ['total', 'step1', 'step2']]
            )

            for step, df in reduced_df_by_step.items():
                ok, found_metrics = check_tsdr_ground_truth_by_route(
                    metrics=list(df.columns),
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                )
                y_true_by_step[step].append(1)
                y_pred_by_step[step].append(1 if ok else 0)
                num_series[step].append(num_series_each_step[step])
                tests_df = tests_df.append(
                    pd.Series(
                        [
                            chaos_type, chaos_comp, metrics_file, step, ok,
                            num_series_str, ','.join(found_metrics), grafana_dashboard_url,
                        ], index=tests_df.columns,
                    ), ignore_index=True,
                )

            logger.info(f">> Uploading clustered plots of {record.chaos_case_file()} ...")
            with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                for representative_metric, sub_metrics in clustering_info.items():
                    executor.submit(log_clustering_plots_as_image,
                                    run, representative_metric, sub_metrics, record)
                    clustering_df = clustering_df.append(
                        pd.Series(
                            [
                                chaos_type, chaos_comp, metrics_file,
                                representative_metric, ','.join(sub_metrics),
                            ], index=clustering_df.columns,
                        ), ignore_index=True,
                    )

            logger.info(f">> Uploading non-clustered plots of {record.chaos_case_file()} ...")
            rep_metrics: list[str] = list(clustering_info.keys())
            non_clustered_reduced_df: pd.DataFrame = reduced_df_by_step['step2'].drop(columns=rep_metrics)
            log_non_clustered_plots_as_image(run, record, non_clustered_reduced_df)
            non_clustered_df = non_clustered_df.append(
                pd.Series(
                    [
                        chaos_type, chaos_comp, metrics_file,
                        ','.join(non_clustered_reduced_df.columns),
                    ], index=non_clustered_df.columns,
                ), ignore_index=True,
            )

        mean_num_series_str: str = '/'.join(
            [f"{statistics.mean(num_series[s])}" for s in ['total', 'step1', 'step2']]
        )
        for step, y_true in y_true_by_step.items():
            y_pred = y_pred_by_step[step]
            tn, fp, fn, tp = confusion_matrix(
                y_true=y_true, y_pred=y_pred, labels=[0, 1],
            ).ravel()
            accuracy: float = accuracy_score(y_true, y_pred)
            recall: float = recall_score(y_true, y_pred)
            mean_reduction_rate: float = 1 - np.mean(np.divide(num_series[step], num_series['total']))
            scores_df = scores_df.append(
                pd.Series([
                    chaos_type, chaos_comp, step,
                    tn, fp, fn, tp, accuracy, recall,
                    mean_num_series_str, mean_reduction_rate],
                    index=scores_df.columns,
                ), ignore_index=True,
            )

    run['tests/clustering/clustered_table'].upload(neptune.types.File.as_html(clustering_df))
    run['tests/clustering/non_clustered_table'].upload(neptune.types.File.as_html(non_clustered_df))

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

    scores_df_by_chaos_type = get_scores_by_index(scores_df, ['chaos_type', 'step'])
    run['scores/table_grouped_by_chaos_type'].upload(neptune.types.File.as_html(scores_df_by_chaos_type))
    scores_df_by_chaos_comp = get_scores_by_index(scores_df, ['chaos_comp', 'step'])
    run['scores/table_grouped_by_chaos_comp'].upload(neptune.types.File.as_html(scores_df_by_chaos_comp))

    logger.info(tests_df.head())
    logger.info(scores_df.head())
    logger.info(scores_df_by_chaos_type.head())
    logger.info(scores_df_by_chaos_comp.head())
    logger.info(clustering_df.head())


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("metricsfiles",
                        nargs='+',
                        help="metrics output JSON file")
    parser.add_argument("--dataset-id",
                        type=str,
                        help='dataset id like "b2qdj"')
    parser.add_argument("--exclude-middleware-metrics",
                        action='store_true',
                        help='a flag for excluding middleware metrics')
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
    run: neptune.Run = neptune.init(mode=args.neptune_mode)
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)
    run['dataset/id'] = args.dataset_id
    run['dataset/num_metrics_files'] = len(args.metricsfiles)
    run['parameters'] = {
        'exclude_middleware_metrics': args.exclude_middleware_metrics,
        'step1_model': args.step1_method,
        'step1_alpha': args.step1_alpha,
        'step2_dist_threshold': args.dist_threshold,
    }

    eval_tsdr(run, args.metricsfiles)

    run.stop()


if __name__ == '__main__':
    main()
