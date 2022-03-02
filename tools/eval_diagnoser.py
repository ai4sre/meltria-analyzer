#!/usr/bin/env python3

import logging
import sys
from multiprocessing import cpu_count

import hydra
import matplotlib.pyplot as plt
import meltria.loader as meltria_loader
import neptune.new as neptune
import networkx as nx
import pandas as pd
from diag_cause import diag
from lib import metrics
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from tsdr import tsdr

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def eval_diagnoser(run: neptune.Run, cfg: DictConfig) -> None:
    dataset, mappings_by_metrics_file = meltria_loader.load_dataset(
        cfg.metrics_files,
        cfg.exclude_middleware_metrics,
    )
    logger.info("Dataset loading complete")

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            logger.info(f">> Running diagnose {record.chaos_case_file()} ...")

            reducer = tsdr.Tsdr(tsdr.ar_based_ad_model, **{
                'tsifter_step1_ar_regression': cfg.tsdr.step1.ar_regression,
                'tsifter_step1_ar_anomaly_score_threshold': cfg.tsdr.step1.ar_anomaly_score_threshold,
                'tsifter_step1_cv_threshold': cfg.tsdr.step1.cv_threshold,
                'tsifter_step1_ar_dynamic_prediction': cfg.tsdr.step1.ar_dynamic_prediction,
                'tsifter_step2_clustering_threshold': cfg.tsdr.step2.dist_threshold,
                'tsifter_step2_clustered_series_type': cfg.tsdr.step2.clustered_series_type,
                'tsifter_step2_clustering_dist_type': cfg.tsdr.step2.clustering_dist_type,
                'tsifter_step2_clustering_choice_method': cfg.tsdr.step2.clustering_choice_method,
                'tsifter_step2_clustering_linkage_method': cfg.tsdr.step2.clustering_linkage_method,
            })
            _, reduced_df_by_step, metrics_dimension, _ = reducer.run(
                series=data_df,
                max_workers=cpu_count(),
            )
            reduced_df: pd.DataFrame = reduced_df_by_step['step2']

            causal_graph, stats = diag.run(
                reduced_df, mappings_by_metrics_file[record.metrics_file], **{
                    'pc_library': cfg.params.pc_library,
                    'pc_citest': cfg.params.pc_citest,
                    'pc_citest_alpha': cfg.params.pc_citest_alpha,
                    'pc_variant': cfg.params.pc_variant,
                }
            )

            logger.info("--> Checking causal graph including chaos-injected metrics")
            is_cause_metrics, cause_metric_nodes = metrics.check_cause_metrics(
                list(causal_graph.nodes()), record.chaos_type, record.chaos_comp,
            )
            if is_cause_metrics:
                logger.info(f"Found cause metric {cause_metric_nodes} in '{chaos_comp}' '{chaos_type}'")
            else:
                logger.info(f"Not found cause metric in '{chaos_comp}' '{chaos_type}'")

            img: bytes = nx.nx_agraph.to_agraph(causal_graph).draw(prog='sfdp', format='png')
            run[f"results/causal_graphs/{record.chaos_case()}"].log(neptune.types.File.from_content(img))
            run[f"results/causal_graphs/stats_{record.chaos_case()}"].log(stats)


@hydra.main(config_path='../conf/diagnoser', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(mode=cfg.neptune.mode)
    npt_handler = NeptuneHandler(run=run)
    logger.addHandler(npt_handler)
    run['dataset/id'] = cfg.dataset_id
    run['dataset/num_metrics_files'] = len(cfg.metrics_files)
    run['parameters'] = {
        'pc_library': cfg.params.pc_library,
        'pc_citest': cfg.params.pc_citest,
        'pc_citest_alpha': cfg.params.pc_citest_alpha,
        'pc_variant': cfg.params.pc_variant,
    }
    # TODO: run['tsdr/parameters'] = {}
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_diagnoser(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
