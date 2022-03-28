from collections import defaultdict

import networkx as nx

ROOT_METRIC_LABELS: tuple[str, str, str] = ("s-front-end_latency", "s-front-end_throughput", "s-front-end_errors")

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

DIAGNOSER_TARGET_DATA: dict[str, list[str]] = {
    "containers": [],  # all
    "services": ["throughput", "latency", "errors"],
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


def group_metrics_by_service(metrics: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(lambda: list())
    for metric in metrics:
        # TODO: resolve duplicated code of MetricNode class.
        comp, base_name = metric.split('-', maxsplit=1)[1].split('_', maxsplit=1)
        if metric.startswith('c-'):
            service = CONTAINER_TO_SERVICE[comp]
        elif metric.startswith('s-'):
            service = comp
        elif metric.startswith('m-'):
            service = CONTAINER_TO_SERVICE[comp]
        else:
            raise ValueError(f'{metric} is invalid')
        groups[service].append(metric)
    return groups
