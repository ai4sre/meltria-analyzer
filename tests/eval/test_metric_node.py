import diagnoser.metric_node as mn
import pytest


@pytest.mark.parametrize(
    ('label', 'expected_comp', 'expected_comp_type', 'expected_base_name'),
    [
        ('c-orders-db_cpu_usage_seconds_total', 'orders-db', mn.MetricType.CONTAINER, 'cpu_usage_seconds_total'),
        ('c-orders_cpu_usage_seconds_total', 'orders', mn.MetricType.CONTAINER, 'cpu_usage_seconds_total'),
        ('c-orders-db_sockets', 'orders-db', mn.MetricType.CONTAINER, 'sockets'),
        ('s-orders_latency', 'orders', mn.MetricType.SERVICE, 'latency'),
        ('s-front-end_latency', 'front-end', mn.MetricType.SERVICE, 'latency'),
        ('n-gke-test-default-pool-1dda290g-n10b_cpu_seconds_total', 'gke-test-default-pool-1dda290g-n10b', mn.MetricType.NODE, 'cpu_seconds_total')
    ],
)
def test_metric_node_init(label, expected_comp, expected_comp_type, expected_base_name):
    got = mn.MetricNode(label)
    assert got.label == label
    assert got.comp == expected_comp
    assert got.comp_type == expected_comp_type
    assert got.base_name == expected_base_name
