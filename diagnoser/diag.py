import json
import os
import time
from itertools import combinations
from typing import Any

import eval.priorknowledge as pk
import networkx as nx
import numpy as np
import pandas as pd
import pcalg
from pgmpy import estimators

import diagnoser.metric_node as mn
from diagnoser import nx_util

from .citest.fisher_z import ci_test_fisher_z
from .citest.fisher_z_pgmpy import fisher_z

SIGNIFICANCE_LEVEL = 0.05


def filter_by_target_metrics(data_df: pd.DataFrame) -> pd.DataFrame:
    """Filter by specified target metrics
    """
    containers_df, services_df, nodes_df, middlewares_df = None, None, None, None
    if 'containers' in pk.DIAGNOSER_TARGET_DATA:
        containers_df = data_df.filter(
            regex=f"^c-.+({'|'.join(pk.DIAGNOSER_TARGET_DATA['containers'])})$")
    if 'services' in pk.DIAGNOSER_TARGET_DATA:
        services_df = data_df.filter(
            regex=f"^s-.+({'|'.join(pk.DIAGNOSER_TARGET_DATA['services'])})$")
    if 'nodes' in pk.DIAGNOSER_TARGET_DATA:
        nodes_df = data_df.filter(
            regex=f"^n-.+({'|'.join(pk.DIAGNOSER_TARGET_DATA['nodes'])})$")
    if 'middlewares' in pk.DIAGNOSER_TARGET_DATA:
        # TODO: middleware
        middlewares_df = data_df.filter(
            regex=f"^m-.+({'|'.join(pk.DIAGNOSER_TARGET_DATA['middlewares'])})$")
    return pd.concat([containers_df, services_df, nodes_df], axis=1)


def read_data_file(tsdr_result_file: os.PathLike
                   ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tsdr_result = json.load(open(tsdr_result_file))
    reduced_df = pd.DataFrame.from_dict(tsdr_result['reduced_metrics_raw_data'])
    df = filter_by_target_metrics(reduced_df)
    return df, tsdr_result['metrics_dimension'], \
        tsdr_result['clustering_info'], tsdr_result['components_mappings'], \
        tsdr_result['metrics_meta']


def build_subgraph_of_removal_edges(labels: dict[int, str], mappings: dict[str, Any]) -> nx.Graph:
    """Build a subgraph consisting of removal edges with prior knowledges.
    """
    ctnr_graph: nx.Graph = pk.CONTAINER_CALL_DIGRAPH.to_undirected()
    service_graph: nx.Graph = pk.SERVICE_CALL_DIGRAPH.to_undirected()
    node_ctnr_graph: nx.Graph = nx.Graph()  # Here, a node means a host running containers.
    if (nodes_ctnrs := mappings.get('nodes-containers')):
        for node, ctnrs in nodes_ctnrs.items():
            # TODO: 'nsenter' container should be removed from original dataset.
            for ctnr in [c for c in ctnrs if c != 'nsenter']:
                node_ctnr_graph.add_edge(node, ctnr)

    nodes: list[mn.MetricNode] = mn.metric_nodes_from_labels(labels)
    G: nx.Graph = nx.Graph()
    for u, v in combinations(nodes, 2):
        if u.is_container() and v.is_container():
            if u.comp == v.comp or ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_container() and v.is_service():
            u_service: str = pk.CONTAINER_TO_SERVICE[u.comp]
            if u_service == v.comp or service_graph.has_edge(u_service, v.comp):
                continue
        elif u.is_service() and v.is_container():
            v_service: str = pk.CONTAINER_TO_SERVICE[v.comp]
            if u.comp == v_service or service_graph.has_edge(u.comp, v_service):
                continue
        elif u.is_service() and v.is_service():
            if u.comp == v.comp or service_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_node() and v.is_node():
            # each node has no connectivity.
            pass
        elif u.is_node() and v.is_container():
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif u.is_container() and v.is_node():
            if node_ctnr_graph.has_edge(u.comp, v.comp):
                continue
        elif (u.is_node() and v.is_service()):
            v_ctnrs: list[str] = pk.SERVICE_CONTAINERS[v.comp]
            has_ctnr_on_node = False
            for v_ctnr in v_ctnrs:
                if node_ctnr_graph.has_edge(u.comp, v_ctnr):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        elif u.is_service() and v.is_node():
            u_ctnrs: list[str] = pk.SERVICE_CONTAINERS[u.comp]
            has_ctnr_on_node = False
            for u_ctnr in u_ctnrs:
                if node_ctnr_graph.has_edge(u_ctnr, v.comp):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        # TODO: node and middleware metrics
        else:
            raise ValueError(f"'{u}' or '{v}' has unexpected format")
        # use node number because 'pgmpy' package handles only graph nodes consisted with numpy array.
        G.add_edge(u, v)
    return G


def prepare_init_graph(labels: dict[int, str], mappings: dict[str, Any]) -> nx.Graph:
    """Prepare initialized causal graph."""
    nodes: list[mn.MetricNode] = mn.metric_nodes_from_labels(labels)
    init_g = nx.Graph()
    for (u, v) in combinations(nodes, 2):
        init_g.add_edge(u, v)
    RG: nx.Graph = build_subgraph_of_removal_edges(labels, mappings)
    init_g.remove_edges_from(RG.edges())
    return init_g


def fix_edge_direction_based_hieralchy(G: nx.DiGraph, u: mn.MetricNode, v: mn.MetricNode) -> None:
    # Force direction from (container -> service) to (service -> container) in same service
    if u.is_service() and v.is_container():
        # check whether u and v in the same service
        v_service = pk.CONTAINER_TO_SERVICE[v.comp]
        if u.comp == v_service:
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_direction_based_network_call(
    G: nx.DiGraph, u: mn.MetricNode, v: mn.MetricNode,
    service_dep_graph: nx.DiGraph,
    container_dep_graph: nx.DiGraph,
) -> None:
    # From service to service
    if u.is_service() and v.is_service():
        # If u and v is in the same service, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        if (v.comp not in service_dep_graph[u.comp]) and \
           (u.comp in service_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to container
    if u.is_container() and v.is_container():
        # If u and v is in the same container, force bi-directed edge.
        if u.comp == v.comp:
            nx_util.set_bidirected_edge(G, u, v)
        elif (v.comp not in container_dep_graph[u.comp]) and \
             (u.comp in container_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)

    # From service to container
    if u.is_service() and v.is_container():
        v_service = pk.CONTAINER_TO_SERVICE[v.comp]
        if (v_service not in service_dep_graph[u.comp]) and \
           (u.comp in service_dep_graph[v_service]):
            nx_util.reverse_edge_direction(G, u, v)

    # From container to service
    if u.is_container() and v.is_service():
        # u_ctnr = u.split('-', maxsplit=1)[1].split('_')[0]
        # v_service = v.split('-', maxsplit=1)[1].split('_')[0]
        u_service = pk.CONTAINER_TO_SERVICE[u.comp]
        if (v.comp not in service_dep_graph[u_service]) and \
           (u_service in service_dep_graph[v.comp]):
            nx_util.reverse_edge_direction(G, u, v)


def fix_edge_directions_in_causal_graph(
    G: nx.DiGraph,
) -> nx.DiGraph:
    """Fix the edge directions in the causal graphs.
    1. Fix directions based on the system hieralchy such as a service and a container
    2. Fix directions based on the network call graph.
    """
    service_dep_graph: nx.DiGraph = pk.SERVICE_CALL_DIGRAPH.reverse()
    container_dep_graph: nx.DiGraph = pk.CONTAINER_CALL_DIGRAPH.reverse()
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
    node_to_ids = {n: n.id for n in init_g.nodes}
    node_ids_to_node = {n.id: n for n in init_g.nodes}
    init_g = nx.relabel_nodes(init_g, mapping=node_to_ids)
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
    DG = nx.relabel_nodes(DG, mapping=node_ids_to_node)
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
    # Exclude nodes that have no path to root node for visualization
    remove_nodes = []
    UG: nx.Graph = G.to_undirected()
    nodes: nx.classes.reportviews.NodeView = G.nodes
    for node in nodes:
        has_paths: list[bool] = []
        for root in pk.ROOT_METRIC_LABELS:
            rmn = mn.MetricNode(root)
            if UG.has_node(rmn) and UG.has_node(node):
                has_paths.append(nx.has_path(UG, rmn, node))
        if not any(has_paths):
            remove_nodes.append(node)
            continue
        if node.is_service():
            color = "red"
        elif node.is_container():
            color = "blue"
        elif node.is_middleware():
            color = "purple"
        else:
            color = "green"
        G.nodes[node]["color"] = color
    G.remove_nodes_from(remove_nodes)
    return G


def run(dataset: pd.DataFrame, mappings: dict[str, Any], **kwargs) -> tuple[nx.DiGraph, dict[str, Any]]:
    dataset = filter_by_target_metrics(dataset)
    if not any(label in dataset.columns for label in pk.ROOT_METRIC_LABELS):
        raise ValueError(f"dataset has no root metric node: {pk.ROOT_METRIC_LABELS}")

    building_graph_start: float = time.time()

    labels: dict[int, str] = {i: v for i, v in enumerate(dataset.columns)}
    init_g: nx.Graph = prepare_init_graph(labels, mappings)
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
        'causal_graph_nodes_num': g.number_of_nodes(),
        'causal_graph_edges_num': g.number_of_edges(),
        'causal_graph_density': nx.density(g),
        'causal_graph_flow_hierarchy': nx.flow_hierarchy(g),
        'building_graph_elapsed_sec': building_graph_elapsed,
    }
    return g, stats
