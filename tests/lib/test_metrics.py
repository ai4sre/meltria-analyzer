from lib.metrics import check_tsdr_ground_truth_by_route


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
    ok, found_metrics = check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'user-db')
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
    ok, found_metrics = check_tsdr_ground_truth_by_route(metrics, 'pod-cpu-hog', 'front-end')
    assert ok is True
    expected = [
        'c-front-end_cpu_usage_seconds_total',
        'c-front-end_cpu_user_seconds_total',
        's-front-end_latency',
    ]
    assert found_metrics == expected
