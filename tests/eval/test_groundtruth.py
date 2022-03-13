import networkx as nx
import pytest
from diagnoser import metric_node as mn

from eval import groundtruth


def test_check_tsdr_ground_truth_by_route():
    metrics = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        'c-user-db_network_receive_bytes',
        's-user_latency',
        'c-orders_cpu_usage_seconds_total',
        's-orders_latency',
        's-front-end_latency',
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
    assert ok is True
    expected = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        's-user_latency',
        's-orders_latency',
        's-front-end_latency',
    ]
    assert found_metrics == expected

    # without orders
    metrics = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        'c-user-db_network_receive_bytes',
        's-user_latency',
        's-front-end_latency',
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
    assert ok is True
    expected = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        's-user_latency',
        's-front-end_latency',
    ]
    assert found_metrics == expected

    # not match
    metrics = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        'c-user-db_network_receive_bytes',
        's-user_latency',
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
    assert ok is False
    expected = [
        'c-user-db_cpu_usage_seconds_total',
        'c-user-db_cpu_user_seconds_total',
        's-user_latency',
    ]
    assert found_metrics == expected

    # only front-end
    metrics = [
        'c-front-end_cpu_usage_seconds_total',
        'c-front-end_cpu_user_seconds_total',
        'c-user-db_network_receive_bytes',
        's-front-end_latency',
    ]
    ok, found_metrics = groundtruth.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'front-end')
    assert ok is True
    expected = [
        'c-front-end_cpu_usage_seconds_total',
        'c-front-end_cpu_user_seconds_total',
        's-front-end_latency',
    ]
    assert found_metrics == expected


@pytest.mark.parametrize(
    "desc,chaos_type,chaos_comp,input,expected",
    [
        (
            'correct01', 'pod-cpu-hog', 'user-db',
            [
                # u (cause) -> v
                ("s-user_latency", "s-front-end_latency"),
                ("c-user-db_cpu_usage_seconds_total", "s-user_latency"),
                ("s-orders_latency", "s-front-end_latency"),
                ("c-orders-db_cpu_usage_seconds_total", "s-orders_latency"),
                ("s-user_latency", "s-orders_latency"),
            ],
            [
                ['s-front-end_latency', 's-orders_latency', 's-user_latency', 'c-user-db_cpu_usage_seconds_total'],
                ['s-front-end_latency', 's-user_latency', 'c-user-db_cpu_usage_seconds_total'],
            ]
        ), (
            'correct02: cause container metric -> the other service metric',
            'pod-cpu-hog', 'user-db',
            [
                # u (cause) -> v
                ("c-user-db_cpu_usage_seconds_total", "s-front-end_latency"),
            ],
            [
                ['s-front-end_latency', 'c-user-db_cpu_usage_seconds_total'],
            ]
        ), (
            'correct03: cause container metric -> container metric',
            'pod-cpu-hog', 'user-db',
            [
                # u (cause) -> v
                ("c-user-db_cpu_usage_seconds_total", "s-front-end_latency"),
                ("c-user-db_cpu_usage_seconds_total", "c-user-db_cpu_user_seconds_total"),
                ("s-orders_latency", "s-front-end_latency"),
                ("c-user-db_cpu_user_seconds_total", "s-orders_latency")
            ],
            [
                ['s-front-end_latency', 's-orders_latency',
                 'c-user-db_cpu_user_seconds_total', 'c-user-db_cpu_usage_seconds_total'],
                ['s-front-end_latency', 's-orders_latency',
                 'c-user-db_cpu_user_seconds_total'],
                ['s-front-end_latency', 'c-user-db_cpu_usage_seconds_total'],
            ]
        ), (
            'correct04: bi-directed graph',
            'pod-cpu-hog', 'carts',
            [
                ['c-carts_cpu_user_seconds_total', 's-front-end_latency'],
                ['s-carts_latency', 's-front-end_latency'],
                ['c-carts_threads', 's-carts_latency'],
                ['c-carts_threads', 'c-carts_cpu_user_seconds_total'],
                ['c-carts_cpu_user_seconds_total', 'c-carts_threads'],
            ],
            [
                ['s-front-end_latency', 'c-carts_cpu_user_seconds_total'],
                ['s-front-end_latency', 'c-carts_cpu_user_seconds_total', 'c-carts_threads'],
                ['s-front-end_latency', 's-carts_latency', 'c-carts_threads'],
                ['s-front-end_latency', 's-carts_latency', 'c-carts_threads', 'c-carts_cpu_user_seconds_total'],
            ]
        ),
    ],
    ids=['correct01', 'correct02', 'correct03', 'correct04'],
)
def test_check_causal_graph(desc, chaos_type, chaos_comp, input, expected):
    edges = [(mn.MetricNode(u), mn.MetricNode(v)) for (u, v) in input]
    expected_edges = [[mn.MetricNode(n) for n in path] for path in expected]
    G = nx.DiGraph(edges)
    ok, routes = groundtruth.check_causal_graph(G, chaos_type, chaos_comp)
    assert ok
    assert sorted([r.nodes for r in routes]) == sorted(expected_edges)
