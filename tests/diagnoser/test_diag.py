import networkx as nx
import pytest
from diag_cause import diag


@pytest.mark.parametrize(
    "case,input,expected",
    [
        (
            'hieralchy01: reverse single direction edge',
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            'hieralchy02: determine bi-direction edge',
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            'nwcall01: service to service',
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-front-end_latency", "s-user_latency"),  # wrong
                ("s-user_latency", "s-orders_throughput"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("s-user_latency", "s-orders_throughput", {}),
            ],
        ),
        (
            'nwcall02: container to container',
            [
                ("c-user_cpu_usage_seconds_total", "c-user-db_cpu_usage_seconds_total"),  # wrong
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                ("c-user-db_cpu_usage_seconds_total", "c-user_cpu_usage_seconds_total", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ),
        (
            'hybrid01: mixed hieralchy and nwcall',
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
                ("c-user_cpu_usage_seconds_total", "c-user-db_cpu_usage_seconds_total"),  # wrong
                ("c-user_cpu_usage_seconds_total", "s-user_latency"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
                ("c-user_cpu_usage_seconds_total", "c-user-db_cpu_usage_seconds_total", {}),
            ],
        )
    ],
    ids=['hieralchy01', 'hieralchy02', 'nwcall01', 'nwcall02', 'hybrid01'],
)
def test_fix_edge_directions_in_causal_graph(case, input, expected):
    G = nx.DiGraph()
    G.add_edges_from(input)
    got = diag.fix_edge_directions_in_causal_graph(G)
    assert list(nx.to_edgelist(got)).sort() == expected.sort()
