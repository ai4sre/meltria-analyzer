import pandas as pd

from tsdr import tsdr


def test_get_container_names_of_service():
    df = pd.DataFrame({
        's-front-end_latency': [],
        's-front-end_throughput': [],
        'c-front-end_cpu_usage': [],
        'c-front-end_memory_usage': [],
        's-user_latency': [],
        's-user_throughput': [],
        'c-user_cpu_usage': [],
        'c-user_memory_usage': [],
        'c-user-db_cpu_usage': [],
        'c-user-db_memory_usage': [],
    })
    got = tsdr.get_container_names_of_service(df)
    expected = {
        'front-end': set(['front-end']),
        'user': set(['user', 'user-db']),
    }
    assert got == expected
