import re
from collections import defaultdict
from typing import Any

CHAOS_TO_CAUSE_METRIC_PREFIX = {
    'pod-cpu-hog': 'cpu_',
    'pod-memory-hog': 'memory_',
    'pod-network-loss': 'network_',
    'pod-network-latency': 'network_',
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

SERVICE_TO_SERVICES: dict[str, list[str]] = {
    'orders': ['front-end'],
    'carts': ['orders', 'front-end'],
    'user': ['orders', 'front-end'],
    'catalogue': ['front-end'],
    'payment': ['orders'],
    'shipping': ['orders'],
    'front-end': [],
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
    route: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    ctos: dict[str, str] = generate_containers_to_service()
    for chaos, metric_prefix in CHAOS_TO_CAUSE_METRIC_PREFIX.items():
        for ctnr in CONTAINER_CALL_GRAPH.keys():
            if ctnr in SKIP_CONTAINERS:
                continue
            metrics: list[str] = route[chaos][ctnr]
            metrics.append(f"^c-{ctnr}_{metric_prefix}.+")
            service: str = ctos[ctnr]
            metrics.append(f"^s-{service}_.+")
            src_dep_services = SERVICE_TO_SERVICES[service]
            for src in src_dep_services:
                if src == 'front-end':
                    metrics.append(ROOT_METRIC_LABEL)
                    continue
                metrics.append(f"^s-{src}_.+")
    return route


TSDR_GROUND_TRUTH: dict[str, Any] = generate_tsdr_ground_truth()


def check_tsdr_ground_truth_by_route(metrics: list[str], chaos_type: str, chaos_comp: str
                                     ) -> tuple[bool, list[str]]:
    gt_metrics: list[str] = TSDR_GROUND_TRUTH[chaos_type][chaos_comp]
    gt_metrics_ok = {metric: False for metric in gt_metrics}
    match_metrics: list[str] = []
    for metric in metrics:
        for matcher in gt_metrics:
            if re.match(matcher, metric):
                gt_metrics_ok[matcher] = True
                match_metrics.append(metric)
    for ok in gt_metrics_ok.values():
        if not ok:
            # return partially correct metrics
            return False, match_metrics
    return True, match_metrics


def check_cause_metrics(metrics: list[str], chaos_type: str, chaos_comp: str
                        ) -> tuple[bool, list[Any]]:
    prefix = CHAOS_TO_CAUSE_METRIC_PREFIX[chaos_type]
    cause_metrics = []
    for metric in metrics:
        if re.match(f"^c-{chaos_comp}_{prefix}.+", metric):
            cause_metrics.append(metric)
    if len(cause_metrics) > 0:
        return True, cause_metrics
    return False, cause_metrics
