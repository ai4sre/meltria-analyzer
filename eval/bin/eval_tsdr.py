#!/usr/bin/env python3

import logging
import os
import statistics
from collections import defaultdict
from concurrent import futures
from functools import reduce
from multiprocessing import cpu_count
from multiprocessing.sharedctypes import Value
from operator import add

import eval.priorknowledge as pk
import holoviews as hv
import hydra
import meltria.loader as meltria_loader
import neptune.new as neptune
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from bokeh.embed import file_html
from bokeh.resources import CDN
from eval import groundtruth
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tsdr import tsdr

hv.extension('bokeh')


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

    def log_plots_as_html(self, record: DatasetRecord) -> None:
        """ Upload found_metrics plot images to neptune.ai.
        """
        if not self.enable_upload_plots:
            return
        self.logger.info(f">> Uploading plot figures of {record.chaos_case_file()} ...")
        if (gtdf := record.ground_truth_metrics_frame()) is None:
            return
        html = self.generate_html_time_series(
            record, gtdf, title=f'Chart of time series metrics {record.chaos_case_full()}')
        self.run[f"dataset/figures/{record.chaos_case_full()}"].upload(
            neptune.types.File.from_content(html, extension='html'),
        )

    def log_clustering_plots_as_html(
        self,
        clustering_info: dict[str, list[str]],
        non_clustered_reduced_df: pd.DataFrame,
        record: DatasetRecord,
        anomaly_points: dict[str, np.ndarray],
    ) -> None:
        """ Upload clustered time series plots to neptune.ai.
        """
        if not self.enable_upload_plots:
            return

        # Parallelize plotting of clustered and no-clustered metrics.
        with futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_list: dict[futures.Future, str] = {}
            f = executor.submit(
                self.get_html_of_clustered_series_plots,
                clustering_info=clustering_info,
                record=record,
                anomaly_points=anomaly_points,
            )
            future_list[f] = f"tests/clustering/time_series_plots/{record.chaos_case_full()}.clustered"
            f = executor.submit(
                self.get_html_of_non_clustered_series_plots,
                non_clustered_reduced_df=non_clustered_reduced_df,
                record=record,
                anomaly_points=anomaly_points,
            )
            future_list[f] = f"tests/clustering/time_series_plots/{record.chaos_case_full()}.no_clustered"
            for future in futures.as_completed(future_list):
                neptune_path: str = future_list[future]
                html: str = future.result()
                self.run[neptune_path].upload(
                    neptune.types.File.from_content(html, extension='html'),
                )

    @classmethod
    def get_html_of_clustered_series_plots(
        cls,
        clustering_info: dict[str, list[str]],
        record: DatasetRecord,
        anomaly_points: dict[str, np.ndarray],
    ) -> str:
        """ Upload clustered time series plots to neptune.ai.
        """
        logger.info(f">> Uploading clustering plots of {record.chaos_case_file()} ...")
        figures: list[hv.Overlay] = []
        for rep_metric, sub_metrics in clustering_info.items():
            clustered_metrics: list[str] = [rep_metric] + sub_metrics
            fig: hv.Overlay = cls.generate_figure_time_series(
                data=record.data_df[clustered_metrics],
                anomaly_points=anomaly_points,
                title=f'Chart of time series metrics {record.chaos_case_full()} / rep:{rep_metric}',
                width_and_height=(800, 400),
            )
            figures.append(fig)
        final_fig = reduce(add, figures)
        return file_html(hv.render(final_fig), CDN, record.chaos_case_full())

    @classmethod
    def get_html_of_non_clustered_series_plots(
        cls,
        record: DatasetRecord,
        non_clustered_reduced_df: pd.DataFrame,
        anomaly_points: dict[str, np.ndarray],
    ) -> str:
        """ Upload non-clustered time series plots to neptune.ai.
        """
        logger.info(f">> Uploading non-clustered plots of {record.chaos_case_file()} ...")
        if len(non_clustered_reduced_df.columns) == 0:
            return None

        figures: list[hv.Overlay] = []
        for service, metrics in pk.group_metrics_by_service(list(non_clustered_reduced_df.columns)).items():
            fig: hv.Overlay = cls.generate_figure_time_series(
                data=non_clustered_reduced_df[metrics],
                anomaly_points=anomaly_points,
                title=f'Chart of time series metrics {record.chaos_case_full()} / {service} [no clustered]',
                width_and_height=(800, 400),
            )
            figures.append(fig)
        final_fig = reduce(add, figures)
        return file_html(hv.render(final_fig), CDN, record.chaos_case_full())

    @classmethod
    def generate_html_time_series(
        cls,
        record: DatasetRecord,
        data: pd.DataFrame,
        title: str,
        anomaly_points: dict[str, np.ndarray] = {},
    ) -> str:
        fig = cls.generate_figure_time_series(data, title=title, anomaly_points=anomaly_points)
        return file_html(hv.render(fig), CDN, record.chaos_case_full())

    @classmethod
    def generate_figure_time_series(
        cls,
        data: pd.DataFrame,
        title: str,
        width_and_height: tuple[int, int] = (1200, 600),
        anomaly_points: dict[str, np.ndarray] = {},
    ) -> hv.Overlay:
        hv_curves = []
        for column in data.columns:
            vals: np.ndarray = scipy.stats.zscore(data[column].to_numpy())
            df = pd.DataFrame(data={
                'x': np.arange(vals.size),
                'y': vals,
                'label': column,  # to show label with hovertool
            })
            line = hv.Curve(df, label=column).opts(tools=['hover', 'tap'])
            if (points := anomaly_points.get(column)) is None:
                hv_curves.append(line)
            else:
                ap = np.array([(p[0], vals[p[0]]) for p in points])
                hv_curves.append(line * hv.Points(ap).opts(color='red', size=8, marker='x'))
        return hv.Overlay(hv_curves).opts(
            title=title,
            tools=['hover', 'tap'],
            width=width_and_height[0], height=width_and_height[1],
            xlabel='time', ylabel='zscore',
            fontsize={'legend': 8},
            show_grid=True, legend_limit=100,
            show_legend=True, legend_position='right', legend_muted=True,
        )


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


def save_scores(
    run: neptune.Run,
    scores_df: pd.DataFrame, tests_df: pd.DataFrame,
    clustering_df: pd.DataFrame, non_clustered_df: pd.DataFrame,
) -> None:
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

            ts_plotter.log_plots_as_html(record)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            tsdr_param = {
                'tsifter_step2_clustering_threshold': cfg.step2.dist_threshold,
                'tsifter_step2_clustered_series_type': cfg.step2.clustered_series_type,
                'tsifter_step2_clustering_dist_type': cfg.step2.clustering_dist_type,
                'tsifter_step2_clustering_choice_method': cfg.step2.clustering_choice_method,
                'tsifter_step2_clustering_linkage_method': cfg.step2.clustering_linkage_method,
            }
            if cfg.step1.model_name == 'cv':
                tsdr_param.update({
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                })
                reducer = tsdr.Tsdr(tsdr.cv_model, **tsdr_param)
            elif cfg.step1.model_name == 'unit_root_test':
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
                    'tsifter_step1_pre_cv': cfg.step1.pre_cv,
                    'tsifter_step1_smoother': cfg.step1.smoother,
                    'tsifter_step1_smoother_ma_window_size': cfg.step1.smoother_ma_window_size,
                    'tsifter_step1_smoother_binner_window_size': cfg.step1.smoother_binner_window_size,
                    'tsifter_step1_ar_regression': cfg.step1.ar_regression,
                    'tsifter_step1_ar_lag': cfg.step1.ar_lag,
                    'tsifter_step1_ar_ic': cfg.step1.ar_ic,
                    'tsifter_step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_ar_dynamic_prediction': cfg.step1.ar_dynamic_prediction,
                })
                reducer = tsdr.Tsdr(tsdr.ar_based_ad_model, **tsdr_param)
            elif cfg.step1.model_name == 'hotteling_t2':
                tsdr_param.update({
                    'tsifter_step1_pre_cv': cfg.step1.pre_cv,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_hotteling_threshold': cfg.step1.hotteling_threshold,
                })
                reducer = tsdr.Tsdr(tsdr.hotteling_t2_model, **tsdr_param)
            elif cfg.step1.model_name == 'sst':
                tsdr_param.update({
                    'tsifter_step1_pre_cv': cfg.step1.pre_cv,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_sst_threshold': cfg.step1.sst_threshold,
                })
                reducer = tsdr.Tsdr(tsdr.sst_model, **tsdr_param)
            elif cfg.step1.model_name == 'differencial_of_anomaly_score':
                tsdr_param.update({
                    'tsifter_step1_pre_cv': cfg.step1.pre_cv,
                    'tsifter_step1_cv_threshold': cfg.step1.cv_threshold,
                    'tsifter_step1_ar_regression': cfg.step1.ar_regression,
                    'tsifter_step1_ar_lag': cfg.step1.ar_lag,
                    'tsifter_step1_ar_ic': cfg.step1.ar_ic,
                    'tsifter_step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
                    'tsifter_step1_changepoint_topk': cfg.step1.changepoint_topk,
                })
                reducer = tsdr.Tsdr(tsdr.differencial_of_anomaly_score_model, **tsdr_param)
            else:
                raise ValueError(f'Invalid name of step1 mode: {cfg.step1.model_name}')

            elapsed_time_by_step, reduced_df_by_step, metrics_dimension, clustering_info, anomaly_points = reducer.run(
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
                ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
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

            for representative_metric, sub_metrics in clustering_info.items():
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
            non_clustered_df = non_clustered_df.append(
                pd.Series(
                    [
                        chaos_type, chaos_comp, metrics_file,
                        ','.join(non_clustered_reduced_df.columns),
                    ], index=non_clustered_df.columns,
                ), ignore_index=True,
            )

            ts_plotter.log_clustering_plots_as_html(clustering_info, non_clustered_reduced_df, record, anomaly_points)

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

    save_scores(run, scores_df, tests_df, clustering_df, non_clustered_df)


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
    if cfg.step1.model_name == 'cv':
        params.update({
            'step1_model_name': cfg.step1.model_name,
            'step1_cv_threshold': cfg.step1.cv_threshold,
        })
    elif cfg.step1.model_name == 'unit_root_test':
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
            'step1_pre_cv': cfg.step1.pre_cv,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_ar_regression': cfg.step1.ar_regression,
            'step1_ar_lag': cfg.step1.ar_lag,
            'step1_ar_ic': cfg.step1.ar_ic,
            'step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
            'step1_ar_dynamic_prediction': cfg.step1.ar_dynamic_prediction,
            'step1_smoother': cfg.step1.smoother,
            'step1_smoother_ma_window_size': cfg.step1.smoother_ma_window_size,
            'step1_smoother_binner_window_size': cfg.step1.smoother_binner_window_size,
        })
    elif cfg.step1.model_name == 'hotteling_t2':
        params.update({
            'step1_model_name': cfg.step1.model_name,
            'step1_pre_cv': cfg.step1.pre_cv,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_hotteling_threshold': cfg.step1.hotteling_threshold,
        })
    elif cfg.step1.model_name == 'sst':
        params.update({
            'step1_pre_cv': cfg.step1.pre_cv,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_sst_threshold': cfg.step1.sst_threshold,
        })
    elif cfg.step1.model_name == 'differencial_of_anomaly_score':
        params.update({
            'step1_model_name': cfg.step1.model_name,
            'step1_pre_cv': cfg.step1.pre_cv,
            'step1_cv_threshold': cfg.step1.cv_threshold,
            'step1_ar_regression': cfg.step1.ar_regression,
            'step1_ar_lag': cfg.step1.ar_lag,
            'step1_ar_ic': cfg.step1.ar_ic,
            'step1_ar_anomaly_score_threshold': cfg.step1.ar_anomaly_score_threshold,
            'step1_changepoint_topk': cfg.step1.changepoint_topk,
        })
    else:
        raise ValueError(f'Unknown model name: {cfg.step1.model_name}')
    run['parameters'] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_tsdr(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
