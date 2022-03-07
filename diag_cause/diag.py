import json
import os
import time
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pcalg
from lib.metrics import (CONTAINER_CALL_DIGRAPH, CONTAINER_CALL_GRAPH,
                         ROOT_METRIC_LABEL, SERVICE_CALL_DIGRAPH,
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


def build_no_paths(labels: dict[int, str], mappings: dict[str, Any]) -> list[list[int]]:
    """Build unnecessary edge paths in causal graph based on the prior knowledge.
    """
    containers_list, services_list, nodes_list = [], [], []
    for v in labels.values():
        if v.startswith('c-'):
            container_name: str = v.split("_")[0].replace("c-", "")
            if container_name not in containers_list:
                containers_list.append(container_name)
        elif v.startswith('s-'):
            service_name: str = v.split("_")[0].replace("s-", "")
            if service_name not in services_list:
                services_list.append(service_name)
        elif v.startswith('n-'):
            node_name: str = v.split("_")[0].replace("n-", "")
            if node_name not in nodes_list:
                nodes_list.append(node_name)

    containers_metrics: dict[str, list[int]] = {}
    for c in containers_list:
        nodes = []
        for k, v in labels.items():
            if v.startswith(f"c-{v}_"):
                nodes.append(k)
        containers_metrics[c] = nodes

    services_metrics: dict[str, list[int]] = {}
    for s in services_list:
        nodes = []
        for k, v in labels.items():
            if v.startswith(f"s-{s}_"):
                nodes.append(k)
        services_metrics[s] = nodes

    nodes_metrics: dict[str, list[int]] = {}
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
        for cm_i in containers_metrics[pair[0]]:
            for cm_j in containers_metrics[pair[1]]:
                no_paths.append([cm_i, cm_j])

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
        for sm_i in services_metrics[pair[0]]:
            for sm_j in services_metrics[pair[1]]:
                no_paths.append([sm_i, sm_j])

    # N-N
    no_deps_N_N_pair = []
    for i, j in combinations(nodes_list, 2):
        no_deps_N_N_pair.append([i, j])
        for nm1 in nodes_metrics[i]:
            for nm2 in nodes_metrics[j]:
                no_paths.append([nm1, nm2])

    # C-N
    for node in nodes_list:
        for con, host_node in nodes_containers.items():
            if node != host_node:
                for nm1 in nodes_metrics[node]:
                    if con not in containers_metrics:
                        continue
                    for cm2 in containers_metrics[con]:
                        no_paths.append([nm1, cm2])

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
                for sm1 in services_metrics[service]:
                    for nm2 in nodes_metrics[node]:
                        no_paths.append([sm1, nm2])

    # C-S
    for service in SERVICE_CONTAINERS:
        for con in containers_metrics:
            if con not in SERVICE_CONTAINERS[service]:
                if service not in services_metrics:
                    continue
                for sm1 in services_metrics[service]:
                    for cm2 in containers_metrics[con]:
                        no_paths.append([sm1, cm2])

    return no_paths


def prepare_init_graph(data_df: pd.DataFrame, no_paths) -> nx.Graph:
    init_g = nx.Graph()
    node_ids = range(len(data_df.columns))
    init_g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        init_g.add_edge(i, j)
    for no_path in no_paths:
        init_g.remove_edge(no_path[0], no_path[1])
    return init_g


def nx_reverse_edge_direction(G: nx.DiGraph, u, v):
    attr = G[u][v]
    G.remove_edge(u, v)
    G.add_edge(v, u, attr=attr) if attr else G.add_edge(v, u)


def fix_edge_direction_based_hieralchy(G: nx.DiGraph, u: str, v: str) -> None:
    # check whether u is service metric and v is container metric
    if not (u.startswith('s-') and v.startswith('c-')):
        return
    # check whether u and v in the same service
    u_service = u.split('-', maxsplit=1)[1].split('_')[0]
    v_service = v.split('-', maxsplit=1)[1].split('_')[0]
    if u_service != v_service:
        return
    nx_reverse_edge_direction(G, u, v)


def fix_edge_direction_based_network_call(
    G: nx.DiGraph, u: str, v: str,
    service_dep_graph: nx.DiGraph,
    container_dep_graph: nx.DiGraph,
) -> None:
    # From service to service
    if (u.startswith('s-') and v.startswith('s-')):
        u_service = u.split('-', maxsplit=1)[1].split('_')[0]
        v_service = v.split('-', maxsplit=1)[1].split('_')[0]
        if (v_service not in service_dep_graph[u_service]) and \
           (u_service in service_dep_graph[v_service]):
            nx_reverse_edge_direction(G, u, v)

    # From container to container
    if (u.startswith('c-') and v.startswith('c-')):
        u_ctnr = u.split('-', maxsplit=1)[1].split('_')[0]
        v_ctnr = v.split('-', maxsplit=1)[1].split('_')[0]
        if (v_ctnr not in container_dep_graph[u_ctnr]) and \
           (u_ctnr in container_dep_graph[v_ctnr]):
            nx_reverse_edge_direction(G, u, v)


def fix_edge_directions_in_causal_graph(
    G: nx.DiGraph,
) -> nx.DiGraph:
    """Fix the edge directions in the causal graphs.
    1. Fix directions based on the system hieralchy such as a service and a container
    2. Fix directions based on the network call graph.
    """
    service_dep_graph: nx.DiGraph = SERVICE_CALL_DIGRAPH.reverse()
    container_dep_graph: nx.DiGraph = CONTAINER_CALL_DIGRAPH.reverse()
    # Traverse the all edges of G via the neighbors
    for u, nbrsdict in G.adjacency():
        nbrs = list(nbrsdict.keys())  # to avoid 'RuntimeError: dictionary changed size during iteration'
        for v in nbrs:
            # u -> v
            fix_edge_direction_based_hieralchy(G, u, v)
            fix_edge_direction_based_network_call(G, u, v, service_dep_graph, container_dep_graph)
    return G


def build_causal_graph_with_pcalg(
    dm: np.ndarray,
    labels: dict[int, str],
    init_g: nx.Graph,
    pc_citest_alpha: float,
    pc_variant: str = '',
    pc_citest: str = 'fisher-z',
) -> nx.DiGraph:
    """
    Build causal graph with PC algorithm.
    """
    cm = np.corrcoef(dm.T)
    ci_test = ci_test_fisher_z if pc_citest == 'fisher-z' else pc_citest
    (G, sep_set) = pcalg.estimate_skeleton(
        indep_test_func=ci_test,
        data_matrix=dm,
        alpha=pc_citest_alpha,
        corr_matrix=cm,
        init_graph=init_g,
        method=pc_variant,
    )
    DG: nx.DiGraph = pcalg.estimate_cpdag(skel_graph=G, sep_set=sep_set)
    DG = nx.relabel_nodes(DG, labels)
    DG = find_dags(DG)
    return fix_edge_directions_in_causal_graph(DG)


def build_causal_graphs_with_pgmpy(
    df: pd.DataFrame,
    pc_citest_alpha: float,
    pc_variant: str = 'orig',
    pc_citest: str = 'fisher-z',
) -> nx.DiGraph:
    c = estimators.PC(data=df)
    ci_test = fisher_z if pc_citest == 'fisher-z' else pc_citest
    G = c.estimate(
        variant=pc_variant,
        ci_test=ci_test,
        significance_level=pc_citest_alpha,
        return_type='pdag',
    )
    return find_dags(G)


def find_dags(G: nx.DiGraph) -> nx.DiGraph:
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


def run(dataset: pd.DataFrame, mappings: dict[str, Any], **kwargs) -> tuple[nx.DiGraph, dict[str, Any]]:
    dataset = filter_by_target_metrics(dataset)
    if ROOT_METRIC_LABEL not in dataset.columns:
        raise ValueError(f"dataset has no root metric node: {ROOT_METRIC_LABEL}")

    building_graph_start: float = time.time()

    labels: dict[int, str] = {i: v for i, v in enumerate(dataset.columns)}
    no_paths = build_no_paths(labels, mappings)
    init_g = prepare_init_graph(dataset, no_paths)
    if kwargs['pc_library'] == 'pcalg':
        g = build_causal_graph_with_pcalg(
            dataset.to_numpy(), labels, init_g,
            pc_variant=kwargs['pc_variant'],
            pc_citest=kwargs['pc_citest'],
            pc_citest_alpha=kwargs['pc_citest_alpha'],
        )
    elif kwargs['pc_library'] == 'pgmpy':
        g = build_causal_graphs_with_pgmpy(
            dataset,
            pc_variant=kwargs['pc_variant'],
            pc_citest=kwargs['pc_citest'],
            pc_citest_alpha=kwargs['pc_citest_alpha'],
        )
    else:
        raise ValueError('library should be pcalg or pgmpy')

    building_graph_elapsed: float = time.time() - building_graph_start

    stats = {
        'init_graph_nodes_num': init_g.number_of_nodes(),
        'init_graph_edges_num': init_g.number_of_edges(),
        'graph_nodes_num': g.number_of_nodes(),
        'graph_edges_num': g.number_of_edges(),
        'building_graph_elapsed_sec': building_graph_elapsed,
    }
    return g, stats
