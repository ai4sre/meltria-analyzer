import networkx as nx
import pytest

import lib.metrics as libmetrics


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
    ok, found_metrics = libmetrics.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = libmetrics.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = libmetrics.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = libmetrics.check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'front-end')
    assert ok is True
    expected = [
        'c-front-end_cpu_usage_seconds_total',
        'c-front-end_cpu_user_seconds_total',
        's-front-end_latency',
    ]
    assert found_metrics == expected


@pytest.mark.parametrize(
    "case,input,expected",
    [
        (
            'normal01',
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
        ),
    ],
    ids=['normal01'],
)
def test_check_causal_graph(case, input, expected):
    G = nx.DiGraph(input)
    ok, routes = libmetrics.check_causal_graph(G, 'pod-cpu-hog', 'user-db')
    assert ok
    assert routes.sort() == expected.sort()
