import re
from collections import defaultdict
from typing import Any

import networkx as nx

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

ROOT_METRIC_LABEL: str = "s-front-end_latency"

SERVICE_CALL_DIGRAPH: nx.DiGraph = nx.DiGraph([
    ('front-end', 'orders'),
    ('front-end', 'catalogue'),
    ('front-end', 'user'),
    ('front-end', 'carts'),
    ('orders', 'shipping'),
    ('orders', 'payment'),
    ('orders', 'user'),
    ('orders', 'carts'),
])

CONTAINER_CALL_DIGRAPH: nx.DiGraph = nx.DiGraph([
    ('front-end', 'orders'),
    ('front-end', 'carts'),
    ('front-end', 'user'),
    ('front-end', 'catalogue'),
    ('front-end', 'session-db'),
    ('orders', 'shipping'),
    ('orders', 'payment'),
    ('orders', 'user'),
    ('orders', 'carts'),
    ('orders', 'orders-db'),
    ('catalogue', 'catalogue-db'),
    ('user', 'user-db'),
    ('carts', 'carts-db'),
    ('shipping', 'rabbitmq'),
    ('rabbitmq', 'queue-master'),
])

CONTAINER_CALL_GRAPH: dict[str, list[str]] = {
    "front-end": ["orders", "carts", "user", "catalogue"],
    "catalogue": ["front-end", "catalogue-db"],
    "catalogue-db": ["catalogue"],
    "orders": ["front-end", "orders-db", "carts", "user", "payement", "shipping"],
    "orders-db": ["orders"],
    "user": ["front-end", "user-db", "orders"],
    "user-db": ["user"],
    "payment": ["orders"],
    "shipping": ["orders", "rabbitmq"],
    "queue-master": ["rabbitmq"],
    "rabbitmq": ["shipping", "queue-master"],
    "carts": ["front-end", "carts-db", "orders"],
    "carts-db": ["carts"],
    "session-db": ["front-end"]
}

# Use list of tuple because of supporting multiple routes
SERVICE_TO_SERVICES: dict[str, list[str]] = {
    'orders': ['front-end'],
    'carts': ['orders', 'front-end'],
    'user': ['orders', 'front-end'],
    'catalogue': ['front-end'],
    'payment': ['orders'],
    'shipping': ['orders'],
    'front-end': [],
}

SERVICE_TO_SERVICE_ROUTES: dict[str, list[tuple[str, ...]]] = {
    'orders': [('front-end',)],
    'carts': [('orders', 'front-end'), ('front-end',)],
    'user': [('orders', 'front-end'), ('front-end',)],
    'catalogue': [('front-end',)],
    'payment': [('orders',)],
    'shipping': [('orders',)],
    'front-end': [()],
}

SERVICE_CONTAINERS: dict[str, list[str]] = {
    "carts": ["carts", "carts-db"],
    "payment": ["payment"],
    "shipping": ["shipping"],
    "front-end": ["front-end"],
    "user": ["user", "user-db"],
    "catalogue": ["catalogue", "catalogue-db"],
    "orders": ["orders", "orders-db"],
}

CONTAINER_TO_SERVICE: dict[str, str] = {c: s for s, ctnrs in SERVICE_CONTAINERS.items() for c in ctnrs}

SKIP_CONTAINERS = ["queue-master", "rabbitmq", "session-db"]


def generate_containers_to_service() -> dict[str, str]:
    ctos: dict[str, str] = {}
    for service, ctnrs in SERVICE_CONTAINERS.items():
        for ctnr in ctnrs:
            ctos[ctnr] = service
    return ctos


def generate_tsdr_ground_truth() -> dict[str, Any]:
    all_gt_routes: dict[str, dict[str, list[list[str]]]] = defaultdict(lambda: defaultdict(list))
    ctos: dict[str, str] = generate_containers_to_service()
    for chaos, metric_patterns in CHAOS_TO_CAUSE_METRIC_PATTERNS.items():
        for ctnr in CONTAINER_CALL_GRAPH.keys():
            if ctnr in SKIP_CONTAINERS:
                continue
            routes: list[list[str]] = all_gt_routes[chaos][ctnr]
            cause_service: str = ctos[ctnr]
            stos_routes: list[tuple[str, ...]] = SERVICE_TO_SERVICE_ROUTES[cause_service]

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


def check_cause_metrics(metrics: list[str], chaos_type: str, chaos_comp: str
                        ) -> tuple[bool, list[Any]]:
    metric_patterns = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type]
    cause_metrics = []
    for metric in metrics:
        for pattern in metric_patterns:
            if re.match(f"^c-{chaos_comp}_{pattern}$", metric):
                cause_metrics.append(metric)
    if len(cause_metrics) > 0:
        return True, cause_metrics
    return False, cause_metrics


def check_causal_graph(
    G: nx.DiGraph, chaos_type: str, chaos_comp: str,
) -> tuple[bool, list[list[str]]]:
    """Check that the causal graph (G) has the accurate route.
    """
    call_graph: nx.DiGraph = G.reverse()  # for traverse starting from root node
    cause_metric_exps: list[str] = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type]
    cause_metric_pattern: re.Pattern = re.compile(f"^c-{chaos_comp}_({'|'.join(cause_metric_exps)})$")

    match_routes: list[list[Any]] = []
    leaves = list(call_graph.nodes)
    leaves.remove(ROOT_METRIC_LABEL)
    for path in nx.all_simple_paths(call_graph, source=ROOT_METRIC_LABEL, target=leaves):
        if len(path) <= 1:
            continue
        # compare the path with ground truth paths
        for i, node in enumerate(path[1:], start=1):  # skip ROOT_METRIC
            comp: str = node.split('-', maxsplit=1)[1].split('_')[0]
            prev_node: str = path[i-1]
            prev_comp: str = prev_node.split('-', maxsplit=1)[1].split('_')[0]
            if node.startswith('s-'):
                if prev_node.startswith('c-'):
                    prev_service = CONTAINER_TO_SERVICE[prev_comp]
                else:
                    prev_service = prev_comp
                if not SERVICE_CALL_DIGRAPH.has_edge(prev_service, comp):
                    break
            elif node.startswith('c-'):
                if prev_node.startswith('s-'):
                    cur_service = CONTAINER_TO_SERVICE[comp]
                    if not (prev_comp == cur_service or SERVICE_CALL_DIGRAPH.has_edge(prev_comp, cur_service)):
                        break
                elif prev_node.startswith('c-'):
                    if not (prev_comp == comp) or CONTAINER_CALL_DIGRAPH.has_edge(prev_comp, comp):
                        break
                if i == (len(path) - 1):  # is leaf?
                    if cause_metric_pattern.match(node):
                        match_routes.append(path)
                        break
            # TODO: middleware
    return len(match_routes) > 0, match_routes
