#!/usr/bin/env python3

import logging
import os
from multiprocessing import cpu_count

import diagnoser.metric_node as mn
import hydra
import meltria.loader as meltria_loader
import neptune.new as neptune
import networkx as nx
import numpy as np
import pandas as pd
from diagnoser import diag
from eval import groundtruth
from meltria.loader import DatasetRecord
from neptune.new.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from pyvis.network import Network
from sklearn.metrics import accuracy_score
from tsdr import tsdr

# see https://docs.neptune.ai/api-reference/integrations/python-logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def set_visual_style_to_graph(G: nx.DiGraph) -> None:
    """Set graph style followed by The valid properties
    https://pyvis.readthedocs.io/en/latest/tutorial.html#adding-list-of-nodes-with-properties
    >>> ['size', 'value', 'title', 'x', 'y', 'label', 'color']
    """
    for node in G.nodes:
        if node.is_root():
            color = "orange"
            size = 20
        elif node.is_service():
            color = "blue"
            size = 15
        elif node.is_middleware():
            color = "purple"
            size = 10
        elif node.is_container():
            color = "green"
            size = 10
        else:
            color = "grey"
            size = 10
        G.nodes[node]["color"] = color
        G.nodes[node]["size"] = size
        G.nodes[node]["label"] = node.label
    for u, v in G.edges:
        G.edges[u, v]["color"] = "black"


def log_causal_graph(run: neptune.Run, causal_graph: nx.DiGraph, record: DatasetRecord) -> None:
    set_visual_style_to_graph(causal_graph)

    nwg = Network(
        directed=True, height='1000px', width='1000px',
        heading=record.chaos_case_file(),
    )
    # piviz assert isinstance(n_id, str) or isinstance(n_id, int)
    relabeled_mapping = mn.MetricNodes.from_list_of_metric_node(list(causal_graph.nodes)).node_to_label()
    relabeled_graph = nx.relabel_nodes(causal_graph, relabeled_mapping, copy=True)
    nwg.from_nx(relabeled_graph)
    nwg.toggle_physics(True)
    html_path = os.path.join(os.getcwd(), record.basename_of_metrics_file() + '.nw_graph.html')
    nwg.write_html(html_path)
    run[f"tests/causal_graphs_html/{record.chaos_case()}"].upload(neptune.types.File(html_path))

    img: bytes = nx.nx_agraph.to_agraph(causal_graph).draw(prog='sfdp', format='png')
    run[f"tests/causal_graphs/{record.chaos_case()}"].log(neptune.types.File.from_content(img))


def eval_diagnoser(run: neptune.Run, cfg: DictConfig) -> None:
    dataset, mappings_by_metrics_file = meltria_loader.load_dataset(
        cfg.metrics_files,
        cfg.exclude_middleware_metrics,
    )
    logger.info("Dataset loading complete")

    scores_df = pd.DataFrame(
        columns=['chaos_type', 'chaos_comp', 'accuracy', 'elapsed_time'],
        index=['chaos_type', 'chaos_comp']
    ).dropna()
    tests_df = pd.DataFrame(
        columns=[
            'chaos_type', 'chaos_comp', 'metrics_file', 'num_series',
            'init_g_num_nodes', 'init_g_num_edges', 'g_num_nodes', 'g_num_edges', 'g_density', 'g_flow_hierarchy',
            'building_graph_elapsed_sec', 'routes', 'found_cause_metrics', 'grafana_dashboard_url',
        ],
        index=['chaos_type', 'chaos_comp', 'metrics_file', 'grafana_dashboard_url'],
    ).dropna()

    for (chaos_type, chaos_comp), sub_df in dataset.groupby(level=[0, 1]):
        y_pred: list[int] = []
        graph_building_elapsed_secs: list[float] = []

        for (metrics_file, grafana_dashboard_url), data_df in sub_df.groupby(level=[2, 3]):
            record = DatasetRecord(chaos_type, chaos_comp, metrics_file, data_df)

            logger.info(f">> Running tsdr {record.chaos_case_file()} ...")

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

            logger.info(f">> Running diagnosis of {record.chaos_case_file()} ...")

            try:
                causal_graph, stats = diag.run(
                    reduced_df, mappings_by_metrics_file[record.metrics_file], **{
                        'pc_library': cfg.params.pc_library,
                        'pc_citest': cfg.params.pc_citest,
                        'pc_citest_alpha': cfg.params.pc_citest_alpha,
                        'pc_variant': cfg.params.pc_variant,
                    }
                )
            except ValueError as e:
                logger.error(e)
                logger.info(f">> Skip because of error {record.chaos_case_file()}")
                continue

            # Check whether cause metrics exists in the causal graph
            _, found_cause_nodes = groundtruth.check_cause_metrics(
                mn.MetricNodes.from_list_of_metric_node(list(causal_graph.nodes)), chaos_type, chaos_comp,
            )

            logger.info(f">> Checking causal graph including chaos-injected metrics of {record.chaos_case_file()}")
            graph_ok, routes = groundtruth.check_causal_graph(causal_graph, chaos_type, chaos_comp)
            if not graph_ok:
                logger.info(f"wrong causal graph in {record.chaos_case_file()}")
            y_pred.append(1 if graph_ok else 0)
            graph_building_elapsed_secs.append(stats['building_graph_elapsed_sec'])
            tests_df = tests_df.append(
                pd.Series(
                    [
                        chaos_type, chaos_comp, metrics_file, metrics_dimension['total'][2],
                        stats['init_graph_nodes_num'], stats['init_graph_edges_num'],
                        stats['causal_graph_nodes_num'], stats['causal_graph_edges_num'],
                        stats['causal_graph_density'], stats['causal_graph_flow_hierarchy'],
                        stats['building_graph_elapsed_sec'],
                        ', '.join([route.liststr() for route in routes]),
                        found_cause_nodes.liststr(), grafana_dashboard_url,
                    ], index=tests_df.columns,
                ), ignore_index=True,
            )
            log_causal_graph(run, causal_graph, record)

        accuracy = accuracy_score([1] * len(y_pred), y_pred)
        scores_df = scores_df.append(
            pd.Series([
                chaos_type, chaos_comp, accuracy, np.mean(graph_building_elapsed_secs),
                ], index=scores_df.columns,
            ), ignore_index=True,
        )

    run['tests/table'].upload(neptune.types.File.as_html(tests_df))
    logger.info(tests_df)
    run['scores/table'].upload(neptune.types.File.as_html(scores_df))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(scores_df)


@hydra.main(config_path='../conf/diagnoser', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup neptune.ai client
    run: neptune.Run = neptune.init(
        project=os.environ['DIAGNOSER_NEPTUNE_PROJECT'],
        api_token=os.environ['DIAGNOSER_NEPTUNE_API_TOKEN'],
        mode=cfg.neptune.mode,
    )
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
    run['tsdr/parameters'] = OmegaConf.to_container(cfg.tsdr)
    run.wait()  # sync parameters for 'async' neptune mode

    logger.info(OmegaConf.to_yaml(cfg))

    eval_diagnoser(run, cfg)

    run.stop()


if __name__ == '__main__':
    main()
