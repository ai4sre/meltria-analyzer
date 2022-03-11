#!/usr/bin/env python3

import logging
import math
import os
import statistics
from collections import defaultdict
from multiprocessing import cpu_count

import hydra
import matplotlib.pyplot as plt
import meltria.loader as meltria_loader
import neptune.new as neptune
import numpy as np
import pandas as pd
from eval import groundtruth
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tsdr import tsdr

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('root_experiment')
logger.setLevel(logging.INFO)

# algorithms
STEP1_METHODS = ['df', 'adf']


class TimeSeriesPlotter:
    run: neptune.Run
    enable_upload_plots: bool
    logger: logging.Logger

    def __init__(
        self,
        run: neptune.Run,
        enable_upload_plots: bool,
        logger: logging.Logger,
    ) -> None:
        self.run = run
        self.enable_upload_plots = enable_upload_plots
        self.logger = logger

    def log_plots_as_image(self, record: DatasetRecord) -> None:
        """ Upload found_metrics plot images to neptune.ai. """
        if not self.enable_upload_plots:
            return

        self.logger.info(f">> Uploading plot figures of {record.chaos_case_file()} ...")

        _, ground_truth_metrics = groundtruth.check_tsdr_ground_truth_by_route(
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
        self.run[f"dataset/figures/{record.chaos_case()}"].log(neptune.types.File.as_image(fig))
        plt.close(fig=fig)

    def log_clustering_plots_as_image(
        self,
        rep_metric: str,
        sub_metrics: list[str],
        metrics_df: pd.DataFrame,
        record: DatasetRecord,
    ) -> None:
        """ Upload clustered time series plots to neptune.ai """

        if not self.enable_upload_plots:
            return

        clustered_metrics: list[str] = [rep_metric] + sub_metrics
        fig, axes = plt.subplots(
            nrows=len(clustered_metrics),
            ncols=1,
            figsize=(6, len(clustered_metrics) * 1.5),
        )
        # reset_index removes extra index texts from the generated figure.
        metrics_df[clustered_metrics].reset_index(drop=True).plot(
            subplots=True, sharex=False, ax=axes,
        )
        fig.suptitle(f"{record.chaos_case_file()}  rep:{rep_metric}")
        self.run[f"tests/clustering/ts_figures/{record.chaos_case()}"].log(
            neptune.types.File.as_image(fig))
        plt.close(fig=fig)

    def log_non_clustered_plots_as_image(
        self,
        record: DatasetRecord,
        non_clustered_reduced_df: pd.DataFrame,
    ) -> None:
        """ Upload non-clustered time series plots to neptune.ai """

        if not self.enable_upload_plots:
            return

        logger.info(f">> Uploading non-clustered plots of {record.chaos_case_file()} ...")

        num_non_clustered_plots = len(non_clustered_reduced_df.columns)
        if num_non_clustered_plots == 0:
            return None
        fig, axes = plt.subplots(
            nrows=math.ceil(num_non_clustered_plots/3),
            ncols=3,
            figsize=(6*3, num_non_clustered_plots * 0.8),
            squeeze=False,  # always return 2D-array axes
        )
        # Match the numbers axes and non-clustered columns
        axes = self.trim_axs(axes, num_non_clustered_plots)
        # reset_index removes extra index texts from the generated figure.
        non_clustered_reduced_df.reset_index(drop=True).plot(
            subplots=True, figsize=(6, 6), sharex=False, sharey=False, ax=axes,
        )
        fig.suptitle(f"{record.chaos_case_file()} - non-clustered metrics")
        self.run[f"tests/clustering/non_clustered_metrics_ts_figures/{record.chaos_case()}"].log(
            neptune.types.File.as_image(fig)
        )
        plt.close(fig=fig)

    @classmethod
    def trim_axs(cls, axs, N):
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
        'elapsed_time': 'mean',
    })
    df['accuracy'] = (df['tp'] + df['tn']) / (df['tn'] + df['fp'] + df['fn'] + df['tp'])
    return df


def eval_tsdr(run: neptune.Run, cfg: DictConfig):
    ts_plotter: TimeSeriesPlotter = TimeSeriesPlotter(
        run=run,
        enable_upload_plots=(cfg.neptune.mode != 'debug') or cfg.upload_plots,
        logger=logger,
    )

    dataset: pd.DataFrame = meltria_loader.load_dataset(
        cfg.metrics_files,
        cfg.exclude_middleware_metrics,
    )[0]
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
                 'num_series', 'reduction_rate', 'elapsed_time'],
        index=['chaos_type', 'chaos_comp', 'step']
    ).dropna()
    tests_df = pd.DataFrame(
        columns=[
            'chaos_type', 'chaos_comp', 'metrics_file', 'step', 'ok',
            'num_series', 'elapsed_time', 'found_metrics', 'grafana_dashboard_url'
        ],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url', 'step'],
    ).dropna()

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        y_true_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        y_pred_by_step: dict[str, list[int]] = defaultdict(lambda: list())
        num_series: dict[str, list[float]] = defaultdict(lambda: list())
        elapsed_time: dict[str, list[float]] = defaultdict(lambda: list())

        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            ts_plotter.log_plots_as_image(record)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            reducer: tsdr.Tsdr
            tsdr_param = {
                'tsifter_step2_clustering_threshold': cfg.step2.dist_threshold,
                'tsifter_step2_clustered_series_type': cfg.step2.clustered_series_type,
                'tsifter_step2_clustering_dist_type': cfg.step2.clustering_dist_type,
                'tsifter_step2_clustering_choice_method': cfg.step2.clustering_choice_method,
                'tsifter_step2_clustering_linkage_method': cfg.step2.clustering_linkage_method,
            }
            if cfg.step1.model_name == 'unit_root_test':
                tsdr_param.update({
                    'tsifter_step1_pre_cv': cfg.step1.pre_cv,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_take_log': cfg.step1.take_log,
                    'tsifter_step1_unit_root_model': cfg.step1.unit_root_model,
                    'tsifter_step1_unit_root_alpha': cfg.step1.unit_root_alpha,
                    'tsifter_step1_unit_root_regression': cfg.step1.unit_root_regression,
                    'tsifter_step1_post_od_model': cfg.step1.post_od_model,
                    'tsifter_step1_post_od_threshold': cfg.step1.post_od_threshold,
                })
                reducer = tsdr.Tsdr(tsdr.unit_root_based_model, **tsdr_param)
            elif cfg.step1.model_name == 'ar_based_ad':
                tsdr_param.update({
                    'tsifter_step1_ar_regression': cfg.step1.ar_regression,
                    'tsifter_step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_ar_dynamic_prediction': cfg.step1.ar_dynamic_prediction,
                })
                reducer = tsdr.Tsdr(tsdr.ar_based_ad_model, **tsdr_param)

            elapsed_time_by_step, reduced_df_by_step, metrics_dimension, clustering_info = reducer.run(
                series=data_df,
                max_workers=cpu_count(),
            )

            elapsed_time_by_step['total'] = elapsed_time_by_step['step1'] + elapsed_time_by_step['step2']
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
                elapsed_time[step].append(elapsed_time_by_step[step])
                tests_df = tests_df.append(
                    pd.Series(
                        [
                            chaos_type, chaos_comp, metrics_file, step, ok,
                            num_series_str, elapsed_time_by_step[step],
                            ','.join(found_metrics), grafana_dashboard_url,
                        ], index=tests_df.columns,
                    ), ignore_index=True,
                )

            if ts_plotter.enable_upload_plots:
                logger.info(f">> Uploading clustered plots of {record.chaos_case_file()} ...")
            pre_clustered_reduced_df = reduced_df_by_step['step1']
            for representative_metric, sub_metrics in clustering_info.items():
                ts_plotter.log_clustering_plots_as_image(
                    representative_metric, sub_metrics, pre_clustered_reduced_df, record,
                )
                clustering_df = clustering_df.append(
                    pd.Series(
                        [
                            chaos_type, chaos_comp, metrics_file,
                            representative_metric, ','.join(sub_metrics),
                        ], index=clustering_df.columns,
                    ), ignore_index=True,
                )

            rep_metrics: list[str] = list(clustering_info.keys())
            post_clustered_reduced_df = reduced_df_by_step['step2']
            non_clustered_reduced_df: pd.DataFrame = post_clustered_reduced_df.drop(columns=rep_metrics)
            ts_plotter.log_non_clustered_plots_as_image(record, non_clustered_reduced_df)
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
            mean_elapsed_time: float = np.mean(elapsed_time[step])
            scores_df = scores_df.append(
                pd.Series([
                    chaos_type, chaos_comp, step,
                    tn, fp, fn, tp, accuracy, recall,
                    mean_num_series_str, mean_reduction_rate, mean_elapsed_time],
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
    run['scores/elapsed_time'] = {
        'mean': scores_df['elapsed_time'].mean(),
        'max': scores_df['elapsed_time'].max(),
        'min': scores_df['elapsed_time'].min(),
    }
    run['scores/table'].upload(neptune.types.File.as_html(scores_df))

    scores_df_by_chaos_type = get_scores_by_index(scores_df, ['chaos_type', 'step'])
    run['scores/table_grouped_by_chaos_type'].upload(neptune.types.File.as_html(scores_df_by_chaos_type))
    scores_df_by_chaos_comp = get_scores_by_index(scores_df, ['chaos_comp', 'step'])
    run['scores/table_grouped_by_chaos_comp'].upload(neptune.types.File.as_html(scores_df_by_chaos_comp))

    logger.info(tests_df.head())
    logger.info(scores_df.head())
    logger.info(scores_df_by_chaos_type)
    logger.info(scores_df_by_chaos_comp)
    logger.info(clustering_df.head())


@hydra.main(config_path='../conf/tsdr', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['TSDR_NEPTUNE_PROJECT'],
        api_token=os.environ['TSDR_NEPTUNE_API_TOKEN'],
        mode=cfg.neptune.mode,
    )
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)
    run['dataset/id'] = cfg.dataset_id
    run['dataset/num_metrics_files'] = len(cfg.metrics_files)
    params = {
        'exclude_middleware_metrics': cfg.exclude_middleware_metrics,
        'step2_dist_threshold': cfg.step2.dist_threshold,
        'step2_clustered_series_type': cfg.step2.clustered_series_type,
        'step2_clustering_dist_type': cfg.step2.clustering_dist_type,
        'step2_clustering_choice_method': cfg.step2.clustering_choice_method,
        'step2_clustering_linkage_method': cfg.step2.clustering_linkage_method,
    }
    if cfg.step1.model_name == 'unit_root_test':
        params.update({
            'step1_model_name': cfg.step1.model_name,
            'step1_take_log': cfg.step1.take_log,
            'step1_unit_root_model': cfg.step1.unit_root_model,
            'step1_alpha': cfg.step1.unit_root_alpha,
            'step1_regression': cfg.step1.unit_root_regression,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_post_od_model': cfg.step1.post_od_model,
            'step1_post_od_threshold': cfg.step1.post_od_threshold,
        })
    elif cfg.step1.model_name == 'ar_based_ad':
        params.update({
            'step1_model_name': cfg.step1.model_name,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_ar_regression': cfg.step1.ar_regression,
            'step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
            'step1_ar_dynamic_prediction': cfg.step1.ar_dynamic_prediction,
        })
    run['parameters'] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_tsdr(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
