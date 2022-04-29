#!/usr/bin/env python3

import logging
import os
from concurrent import futures
from functools import reduce
from multiprocessing import cpu_count
from operator import add
from typing import Any

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
from tsdr import tsdr

hv.extension('bokeh')


# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger('root_experiment')
logger.setLevel(logging.INFO)


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
        if len(figures) == 0:
            return ''
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
            return ''

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


def save_scores(
    run: neptune.Run,
    tests: list[dict[str, Any]], clustering: list[dict[str, Any]], non_clustered: list[dict[str, Any]],
) -> None:
    clustering_df = pd.DataFrame(clustering).set_index(
        ['chaos_type', 'chaos_comp', 'metrics_file', 'representative_metric', 'sub_metrics'])
    non_clustered_df = pd.DataFrame(non_clustered).set_index(['chaos_type', 'chaos_comp', 'metrics_file'])
    tests_df = pd.DataFrame(tests).set_index(
        ['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url', 'step'])

    run['tests/clustering/clustered_table'].upload(neptune.types.File.as_html(clustering_df))
    run['tests/clustering/non_clustered_table'].upload(neptune.types.File.as_html(non_clustered_df))

    run['scores/summary'].upload(neptune.types.File.as_html(tests_df))

    def agg_score(x: pd.DataFrame) -> pd.Series:
        tp = int(x['ok'].sum())
        fn = int((~x['ok']).sum())
        d = {
            'tp': tp,
            'fn': fn,
            'accuracy': tp / x.size,
            'reduction_rate_mean': (1 - x['num_series_reduced'] / x['num_series_total']).mean(),
            'reduction_rate_max': (1 - x['num_series_reduced'] / x['num_series_total']).max(),
            'reduction_rate_min': (1 - x['num_series_reduced'] / x['num_series_total']).min(),
            'elapsed_time': x['elapsed_time'].mean(),
            'elapsed_time_max': x['elapsed_time'].max(),
            'elapsed_time_min': x['elapsed_time'].min(),
        }
        return pd.Series(d)

    scores_by_step = tests_df.groupby('step').apply(agg_score).reset_index().set_index('step')
    scores_by_chaos_type = tests_df.groupby(
        ['chaos_type', 'step']).apply(agg_score).reset_index().set_index(['chaos_type', 'step'])
    scores_by_chaos_comp = tests_df.groupby(
        ['chaos_comp', 'step']).apply(agg_score).reset_index().set_index(['chaos_comp', 'step'])
    scores_by_chaos_type_and_comp = tests_df.groupby(
        ['chaos_type', 'chaos_comp', 'step'],
    ).apply(agg_score).reset_index().set_index(['chaos_type', 'chaos_comp', 'step'])
    total_scores: pd.Series = scores_by_step.loc['step2']

    run['scores'] = total_scores.to_dict()
    run['scores/summary_by_step'].upload(neptune.types.File.as_html(scores_by_step))
    run['scores/summary_by_chaos_type'].upload(neptune.types.File.as_html(scores_by_chaos_type))
    run['scores/summary_by_chaos_comp'].upload(neptune.types.File.as_html(scores_by_chaos_comp))
    run['scores/summary_by_chaos_type_and_chaos_comp'].upload(neptune.types.File.as_html(scores_by_chaos_type_and_comp))
    for df in [scores_by_step, scores_by_chaos_type, scores_by_chaos_comp, scores_by_chaos_type_and_comp]:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n"+df.to_string())


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

    clustering_records: list[dict[str, Any]] = []
    non_clustered_records: list[dict[str, Any]] = []
    tests_records: list[dict[str, Any]] = []

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            ts_plotter.log_plots_as_html(record)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

            tsdr_param = {f'step1_{k}': v for k, v in OmegaConf.to_container(cfg.step1, resolve=True).items()}
            tsdr_param.update({f'step2_{k}': v for k, v in OmegaConf.to_container(cfg.step2, resolve=True).items()})
            reducer = tsdr.Tsdr(cfg.step1.model_name, **tsdr_param)
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

            for step, df in reduced_df_by_step.items():
                ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(
                    metrics=list(df.columns),
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                )
                tests_records.append({
                    'chaos_type': chaos_type, 'chaos_comp': chaos_comp, 'metrics_file': metrics_file, 'step': step,
                    'ok': ok,
                    'num_series_total': num_series_each_step['total'],
                    'num_series_reduced': num_series_each_step[step],
                    'elapsed_time': elapsed_time_by_step[step],
                    'found_metrics': ','.join(found_metrics),
                    'grafana_dashboard_url': grafana_dashboard_url,
                })

            for representative_metric, sub_metrics in clustering_info.items():
                clustering_records.append({
                    'chaos_type': chaos_type, 'chaos_comp': chaos_comp, 'metrics_file': metrics_file,
                    'representative_metric': representative_metric,
                    'sub_metrics': ','.join(sub_metrics),
                })

            rep_metrics: list[str] = list(clustering_info.keys())
            post_clustered_reduced_df = reduced_df_by_step['step2']
            non_clustered_reduced_df: pd.DataFrame = post_clustered_reduced_df.drop(columns=rep_metrics)
            non_clustered_records.append({
                'chaos_type': chaos_type, 'chaos_comp': chaos_comp, 'metrics_file': metrics_file,
                'non_clustered_metrics': ','.join(non_clustered_reduced_df.columns),
            })

            ts_plotter.log_clustering_plots_as_html(clustering_info, non_clustered_reduced_df, record, anomaly_points)

    save_scores(run, tests_records, clustering_records, non_clustered_records)


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
    }

    # Hydra parameters are passed to the Neptune.ai run object
    pycfg_step1 = OmegaConf.to_container(cfg.step1, resolve=True)
    params.update({f'step1_{k}': v for k, v in pycfg_step1.items()})
    pycfg_step2 = OmegaConf.to_container(cfg.step2, resolve=True)
    params.update({f'step2_{k}': v for k, v in pycfg_step2.items()})

    run['parameters'] = params
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_tsdr(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
