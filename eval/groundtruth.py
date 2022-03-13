import re
from collections import defaultdict
from typing import Any

import diagnoser.metric_node as mn
import networkx as nx

import eval.priorknowledge as pk

CHAOS_TO_CAUSE_METRIC_PATTERNS: dict[str, list[str]] = {
    'pod-cpu-hog': [
        'cpu_.+', 'threads', 'sockets', 'file_descriptors', 'processes', 'memory_cache', 'memory_mapped_file',
    ],
    'pod-memory-hog': [
        'memory_.+', 'threads', 'sockets', 'file_descriptors',
        'processes', 'fs_inodes_total', 'fs_limit_bytes', 'ulimits_soft',
    ],
    'pod-network-loss': ['network_.+'],
    'pod-network-latency': ['network_.+'],
}


def generate_containers_to_service() -> dict[str, str]:
    ctos: dict[str, str] = {}
    for service, ctnrs in pk.SERVICE_CONTAINERS.items():
        for ctnr in ctnrs:
            ctos[ctnr] = service
    return ctos


def generate_tsdr_ground_truth() -> dict[str, Any]:
    all_gt_routes: dict[str, dict[str, list[list[str]]]] = defaultdict(lambda: defaultdict(list))
    ctos: dict[str, str] = generate_containers_to_service()
    for chaos, metric_patterns in CHAOS_TO_CAUSE_METRIC_PATTERNS.items():
        for ctnr in pk.CONTAINER_CALL_GRAPH.keys():
            if ctnr in pk.SKIP_CONTAINERS:
                continue
            routes: list[list[str]] = all_gt_routes[chaos][ctnr]
            cause_service: str = ctos[ctnr]
            stos_routes: list[tuple[str, ...]] = pk.SERVICE_TO_SERVICE_ROUTES[cause_service]

            # allow to match any of multiple routes
            for stos_route in stos_routes:
                metrics_patterns: list[str] = []
                # add cause metrics pattern
                metrics_patterns.append(f"^c-{ctnr}_({'|'.join(metric_patterns)})$")
                metrics_patterns.append(f"^s-{cause_service}_.+$")
                if stos_route != ():
                    metrics_patterns.append(f"^s-({'|'.join(stos_route)})_.+")
                routes.append(metrics_patterns)
    return all_gt_routes


TSDR_GROUND_TRUTH: dict[str, Any] = generate_tsdr_ground_truth()


def check_tsdr_ground_truth_by_route(metrics: list[str], chaos_type: str, chaos_comp: str
                                     ) -> tuple[bool, list[str]]:
    gt_metrics_routes: list[list[str]] = TSDR_GROUND_TRUTH[chaos_type][chaos_comp]
    routes_ok: list[tuple[bool, list[str]]] = []
    for gt_route in gt_metrics_routes:
        ok, match_metrics = check_route(metrics, gt_route)
        routes_ok.append((ok, match_metrics))
    for ok, match_metrics in routes_ok:
        if ok:
            return True, match_metrics

    # return longest match_metrics in routes_ok
    max_len = 0
    longest_match_metrics: list[str] = []
    for _, match_metrics in routes_ok:
        if max_len < len(match_metrics):
            max_len = len(match_metrics)
            longest_match_metrics = match_metrics
    return False, longest_match_metrics


def check_route(metrics: list[str], gt_route: list[str]) -> tuple[bool, list[str]]:
    match_metrics: list[str] = []
    gt_metrics_ok = {metric: False for metric in gt_route}
    for metric in metrics:
        for metric_pattern in gt_route:
            if re.match(metric_pattern, metric):
                gt_metrics_ok[metric_pattern] = True
                match_metrics.append(metric)
    for ok in gt_metrics_ok.values():
        if not ok:
            # return partially correct metrics
            return False, match_metrics
    return True, match_metrics


def check_cause_metrics(nodes: mn.MetricNodes, chaos_type: str, chaos_comp: str
                        ) -> tuple[bool, mn.MetricNodes]:
    metric_patterns = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type]
    cause_metrics: list[mn.MetricNode] = []
    for node in nodes:
        for pattern in metric_patterns:
            if re.match(f"^c-{chaos_comp}_{pattern}$", node.label):
                cause_metrics.append(node)
    ret = mn.MetricNodes.from_list_of_metric_node(cause_metrics)
    if len(cause_metrics) > 0:
        return True, ret
    return False, ret


def check_causal_graph(
    G: nx.DiGraph, chaos_type: str, chaos_comp: str,
) -> tuple[bool, list[mn.MetricNodes]]:
    """Check that the causal graph (G) has the accurate route.
    """
    call_graph: nx.DiGraph = G.reverse()  # for traverse starting from root node
    cause_metric_exps: list[str] = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type]
    cause_metric_pattern: re.Pattern = re.compile(f"^c-{chaos_comp}_({'|'.join(cause_metric_exps)})$")

    match_routes: list[mn.MetricNodes] = []
    leaves = [n for n in call_graph.nodes if n.label not in pk.ROOT_METRIC_LABELS]
    roots = [mn.MetricNode(r) for r in pk.ROOT_METRIC_LABELS if call_graph.has_node(mn.MetricNode(r))]
    for root in roots:
        for path in nx.all_simple_paths(call_graph, source=root, target=leaves):
            if len(path) <= 1:
                continue
            # compare the path with ground truth paths
            for i, node in enumerate(path[1:], start=1):  # skip ROOT_METRIC
                prev_node: mn.MetricNode = path[i-1]
                if node.is_service():
                    if prev_node.is_container():
                        prev_service = pk.CONTAINER_TO_SERVICE[prev_node.comp]
                    else:
                        prev_service = prev_node.comp
                    if not pk.SERVICE_CALL_DIGRAPH.has_edge(prev_service, node.comp):
                        break
                elif node.is_container():
                    if prev_node.is_service():
                        cur_service = pk.CONTAINER_TO_SERVICE[node.comp]
                        if not (prev_node.comp == cur_service or pk.SERVICE_CALL_DIGRAPH.has_edge(prev_node.comp, cur_service)):
                            break
                    elif prev_node.is_container():
                        if not (prev_node.comp == node.comp or pk.CONTAINER_CALL_DIGRAPH.has_edge(prev_node.comp, node.comp)):
                            break
                    if i == (len(path) - 1):  # is leaf?
                        if cause_metric_pattern.match(node.label):
                            match_routes.append(mn.MetricNodes.from_list_of_metric_node(path))
                            break
                # TODO: middleware
    return len(match_routes) > 0, match_routes
