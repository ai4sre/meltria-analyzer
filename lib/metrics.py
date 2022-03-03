import re
from collections import defaultdict
from typing import Any

import networkx as nx

CHAOS_TO_CAUSE_METRIC_PATTERNS = {
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

ROOT_METRIC_LABEL = "s-front-end_latency"

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
