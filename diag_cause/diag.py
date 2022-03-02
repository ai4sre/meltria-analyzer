import argparse
import base64
import json
import os
import sys
from datetime import datetime
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pcalg
from IPython.display import Image
from lib.metrics import (CONTAINER_CALL_GRAPH, ROOT_METRIC_LABEL,
                         SERVICE_CONTAINERS, check_cause_metrics)
from pgmpy import estimators

from .citest.fisher_z import ci_test_fisher_z
from .citest.fisher_z_pgmpy import fisher_z

SIGNIFICANCE_LEVEL = 0.05

TARGET_DATA: dict[str, list[str]] = {
    "containers": [],  # all
    "services": ["throughput", "latency"],
    "nodes": [
        "node_cpu_seconds_total",
        "node_disk_io_now",
        "node_filesystem_avail_bytes",
        "node_memory_MemAvailable_bytes",
        "node_network_receive_bytes_total",
        "node_network_transmit_bytes_total"
    ],
    # "middlewares": "all"}
}


def filter_by_target_metrics(data_df: pd.DataFrame) -> pd.DataFrame:
    """Filter by specified target metrics
    """
    containers_df, services_df, nodes_df, middlewares_df = None, None, None, None
    if 'containers' in TARGET_DATA:
        containers_df = data_df.filter(
            regex=f"^c-.+({'|'.join(TARGET_DATA['containers'])})$")
    if 'services' in TARGET_DATA:
        services_df = data_df.filter(
            regex=f"^s-.+({'|'.join(TARGET_DATA['services'])})$")
    if 'nodes' in TARGET_DATA:
        nodes_df = data_df.filter(
            regex=f"^n-.+({'|'.join(TARGET_DATA['nodes'])})$")
    if 'middlewares' in TARGET_DATA:
        # TODO: middleware
        middlewares_df = data_df.filter(
            regex=f"^m-.+({'|'.join(TARGET_DATA['middlewares'])})$")
    return pd.concat([containers_df, services_df, nodes_df], axis=1)


def read_data_file(tsdr_result_file: os.PathLike
                   ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tsdr_result = json.load(open(tsdr_result_file))
    reduced_df = pd.DataFrame.from_dict(tsdr_result['reduced_metrics_raw_data'])
    df = filter_by_target_metrics(reduced_df)
    return df, tsdr_result['metrics_dimension'], \
        tsdr_result['clustering_info'], tsdr_result['components_mappings'], \
        tsdr_result['metrics_meta']


def build_no_paths(labels, mappings):
    containers_list, services_list, nodes_list = [], [], []
    for v in labels.values():
        if v.startswith('c-'):
            container_name = v.split("_")[0].replace("c-", "")
            if container_name not in containers_list:
                containers_list.append(container_name)
        elif v.startswith('s-'):
            service_name = v.split("_")[0].replace("s-", "")
            if service_name not in services_list:
                services_list.append(service_name)
        elif v.startswith('n-'):
            node_name = v.split("_")[0].replace("n-", "")
            if node_name not in nodes_list:
                nodes_list.append(node_name)

    containers_metrics = {}
    for c in containers_list:
        nodes = []
        for k, v in labels.items():
            if v.startswith(f"c-{v}_"):
                nodes.append(k)
        containers_metrics[c] = nodes

    services_metrics = {}
    for s in services_list:
        nodes = []
        for k, v in labels.items():
            if v.startswith(f"s-{s}_"):
                nodes.append(k)
        services_metrics[s] = nodes

    nodes_metrics = {}
    for n in nodes_list:
        nodes = []
        for k, v in labels.items():
            if v.startswith(f"n-{n}_"):
                nodes.append(k)
        nodes_metrics[n] = nodes

    # Share host
    nodes_containers = {}
    for node, containers in mappings["nodes-containers"].items():
        for container in containers:
            if container == "nsenter":
                continue
            nodes_containers[container] = node

    # C-C
    no_paths = []
    no_deps_C_C_pair = []
    for i, j in combinations(containers_list, 2):
        if j not in CONTAINER_CALL_GRAPH[i] and nodes_containers[i] != nodes_containers[j]:
            no_deps_C_C_pair.append([i, j])
    for pair in no_deps_C_C_pair:
        for i in containers_metrics[pair[0]]:
            for j in containers_metrics[pair[1]]:
                no_paths.append([i, j])
    print("No dependence C-C pairs: {}, No paths: {}".format(len(no_deps_C_C_pair), len(no_paths)))

    # S-S
    no_deps_S_S_pair = []
    for i, j in combinations(services_list, 2):
        has_comm = False
        for c1 in SERVICE_CONTAINERS[i]:
            for c2 in SERVICE_CONTAINERS[j]:
                if c2 in CONTAINER_CALL_GRAPH[c1]:
                    has_comm = True
        if not has_comm:
            no_deps_S_S_pair.append([i, j])
    for pair in no_deps_S_S_pair:
        for i in services_metrics[pair[0]]:
            for j in services_metrics[pair[1]]:
                no_paths.append([i, j])
    print("No dependence S-S pairs: {}, No paths: {}".format(len(no_deps_S_S_pair), len(no_paths)))

    # N-N
    no_deps_N_N_pair = []
    for i, j in combinations(nodes_list, 2):
        no_deps_N_N_pair.append([i, j])
        for n1 in nodes_metrics[i]:
            for n2 in nodes_metrics[j]:
                no_paths.append([n1, n2])
    print("No dependence N-N pairs: {}, No paths: {}".format(len(no_deps_N_N_pair), len(no_paths)))

    # C-N
    for node in nodes_list:
        for con, host_node in nodes_containers.items():
            if node != host_node:
                for n1 in nodes_metrics[node]:
                    if con not in containers_metrics:
                        continue
                    for c2 in containers_metrics[con]:
                        no_paths.append([n1, c2])
    print("[C-N] No paths: {}".format(len(no_paths)))

    # S-N
    for service in SERVICE_CONTAINERS:
        host_list = []
        for con in SERVICE_CONTAINERS[service]:
            if nodes_containers[con] not in host_list:
                host_list.append(nodes_containers[con])
        for node in nodes_list:
            if node not in host_list:
                if service not in services_metrics:
                    continue
                for s1 in services_metrics[service]:
                    for n2 in nodes_metrics[node]:
                        no_paths.append([s1, n2])
    print("[S-N] No paths: {}".format(len(no_paths)))

    # C-S
    for service in SERVICE_CONTAINERS:
        for con in containers_metrics:
            if con not in SERVICE_CONTAINERS[service]:
                if service not in services_metrics:
                    continue
                for s1 in services_metrics[service]:
                    for c2 in containers_metrics[con]:
                        no_paths.append([s1, c2])
    print("[C-S] No paths: {}".format(len(no_paths)))
    return no_paths


def prepare_init_graph(reduced_df, no_paths):
    dm = reduced_df.values
    print("Shape of data matrix: {}".format(dm.shape))
    init_g = nx.Graph()
    node_ids = range(len(reduced_df.columns))
    init_g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        init_g.add_edge(i, j)
    print("Number of edges in complete graph : {}".format(init_g.number_of_edges()))
    for no_path in no_paths:
        init_g.remove_edge(no_path[0], no_path[1])
    print("Number of edges in init graph : {}".format(init_g.number_of_edges()))
    return init_g


def build_causal_graph_with_pcalg(dm, labels, init_g, alpha, pc_stable):
    """
    Build causal graph with PC algorithm.
    """
    cm = np.corrcoef(dm.T)
    pc_method = 'stable' if pc_stable else None
    (G, sep_set) = pcalg.estimate_skeleton(
        indep_test_func=ci_test_fisher_z,
        data_matrix=dm,
        alpha=alpha,
        corr_matrix=cm,
        init_graph=init_g,
        method=pc_method,
    )
    G = pcalg.estimate_cpdag(skel_graph=G, sep_set=sep_set)

    G = nx.relabel_nodes(G, labels)

    return find_dags(G)


def build_causal_graphs_with_pgmpy(
    df: pd.DataFrame,
    alpha: float,
    pc_stable: bool,
) -> nx.Graph:
    c = estimators.PC(data=df)
    pc_method = 'stable' if pc_stable else None
    g = c.estimate(
        variant=pc_method,
        ci_test=fisher_z,
        significance_level=alpha,
        return_type='pdag',
    )
    return find_dags(g)


def find_dags(G: nx.Graph) -> nx.Graph:
    # Exclude nodes that have no path to "s-front-end_latency" for visualization
    remove_nodes = []
    undirected_G = G.to_undirected()
    nodes: nx.classes.reportviews.NodeView = G.nodes
    for node in nodes:
        if not nx.has_path(undirected_G, node, ROOT_METRIC_LABEL):
            remove_nodes.append(node)
            continue
        if node.startswith('s-'):
            color = "red"
        elif node.startswith('c-'):
            color = "blue"
        elif node.startswith('m-'):
            color = "purple"
        else:
            color = "green"
        G.nodes[node]["color"] = color
    G.remove_nodes_from(remove_nodes)
    return G


def run(dataset: pd.DataFrame, mappings, **kwargs) -> nx.Graph:
    dataset = filter_by_target_metrics(dataset)
    if ROOT_METRIC_LABEL not in dataset.columns:
        raise ValueError(f"dataset has no root metric node: {ROOT_METRIC_LABEL}")

    labels: dict[int, str] = {i: v for i, v in enumerate(dataset.columns)}
    no_paths = build_no_paths(labels, mappings)
    init_g = prepare_init_graph(dataset, no_paths)
    if kwargs['pc_library'] == 'pcalg':
        g = build_causal_graph_with_pcalg(
            dataset.values, labels, init_g,
            kwargs['pc_citest_alpha'],
            kwargs['pc_variant'],
        )
    elif kwargs['pc_library'] == 'pgmpy':
        g = build_causal_graphs_with_pgmpy(
            dataset, kwargs['pc_citest_alpha'], kwargs['pc_variant'],
        )
    else:
        raise ValueError('library should be pcalg or pgmpy')
    return g


def diag(tsdr_file, citest_alpha, pc_stable, library, out_dir):
    """[deprecated]
    """
    reduced_df, metrics_dimension, clustering_info, mappings, metrics_meta = \
        read_data_file(tsdr_file)
    if ROOT_METRIC_LABEL not in reduced_df.columns:
        raise ValueError(
            f"{tsdr_file} has no root metric node: {ROOT_METRIC_LABEL}")

    labels = {}
    for i in range(len(reduced_df.columns)):
        labels[i] = reduced_df.columns[i]

    print("--> Building no paths", file=sys.stderr)
    no_paths = build_no_paths(labels, mappings)

    print("--> Preparing initial graph", file=sys.stderr)
    init_g = prepare_init_graph(reduced_df, no_paths)

    print("--> Building causal graph", file=sys.stderr)
    if library == 'pcalg':
        g = build_causal_graph_with_pcalg(
            reduced_df.values, labels, init_g, citest_alpha, pc_stable)
    elif library == 'pgmpy':
        g = build_causal_graphs_with_pgmpy(
            reduced_df, citest_alpha, pc_stable)
    else:
        raise ValueError('library should be pcalg or pgmpy')

    print("--> Checking causal graph including chaos-injected metrics", file=sys.stderr)
    chaos_type = metrics_meta['injected_chaos_type']
    chaos_comp = metrics_meta['chaos_injected_component']
    is_cause_metrics, cause_metric_nodes = check_cause_metrics(
        list(g.nodes()), chaos_type, chaos_comp)
    if is_cause_metrics:
        print(
            f"Found cause metric {cause_metric_nodes} in '{chaos_comp}' '{chaos_type}'", file=sys.stderr)
    else:
        print(
            f"Not found cause metric in '{chaos_comp}' '{chaos_type}'", file=sys.stderr)

    agraph = nx.nx_agraph.to_agraph(g)
    img = agraph.draw(prog='sfdp', format='png')
    if out_dir is None:
        Image(img)
    else:
        id = os.path.splitext(os.path.basename(tsdr_file))[0]
        out_dir = os.path.join(out_dir, id)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        imgfile = os.path.join(out_dir, ts) + '.png'
        plt.savefig(imgfile)
        print(
            f"Saved the file of causal graph image to {imgfile}", file=sys.stderr)

        metadata = {
            'metrics_meta': metrics_meta,
            'parameters': {
                'pc-stable': pc_stable,
                'citest_alpha': citest_alpha,
            },
            'causal_graph_stats': {
                'cause_metric_nodes': cause_metric_nodes,
                'nodes_num': g.number_of_nodes(),
                'edges_num': g.number_of_edges(),
            },
            'metrics_dimension': metrics_dimension,
            'clustering_info': clustering_info,
            # convert base64 encoded bytes to string to serialize it as json
            'raw_image': base64.b64encode(img).decode('utf-8'),
        }
        metafile = os.path.join(out_dir, ts) + '.json'
        with open(metafile, mode='w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved the file of metadata to {metafile}", file=sys.stderr)
        return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsdr_resultfile", help="results file of tsdr")
    parser.add_argument("--citest-alpha",
                        default=SIGNIFICANCE_LEVEL,
                        type=float,
                        help="alpha value of independence test for building causality graph")
    parser.add_argument("--pc-stable",
                        action='store_true',
                        help='whether to use stable method of PC-algorithm')
    parser.add_argument("--library",
                        default='pcalg',
                        help='pcalg or pgmpy')
    parser.add_argument("--out-dir",
                        help='output directory for saving graph image and metadata from tsdr')
    args = parser.parse_args()

    diag(args.tsdr_resultfile, args.citest_alpha,
         args.pc_stable, args.library, args.out_dir)
