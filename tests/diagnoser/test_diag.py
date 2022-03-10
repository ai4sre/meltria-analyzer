import networkx as nx
import pytest
from diag_cause import diag


def test_build_subgraph_of_removal_edges():
    metrics = [
        's-front-end_latency', 's-orders_latency', 'c-orders_sockets', 'c-orders-db_cpu_usage_seconds_total',
        's-user_latency', 'c-user_sockets', 'c-user_cpu_usage_seconds_total', 'c-user-db_cpu_usage_seconds_total',
    ]
    labels: dict[int, str] = {i: v for i, v in enumerate(metrics)}
    RG: nx.Graph = diag.build_subgraph_of_removal_edges(labels)
    RG = nx.relabel_nodes(RG, labels)
    expected = [
        ('c-orders_sockets', 'c-user-db_cpu_usage_seconds_total'),
        ('c-user-db_cpu_usage_seconds_total', 'c-orders-db_cpu_usage_seconds_total'),
        ('c-orders-db_cpu_usage_seconds_total', 'c-user_sockets'),
        ('c-orders-db_cpu_usage_seconds_total', 'c-user_cpu_usage_seconds_total'),
    ]
    assert sorted(list(RG.edges)) == sorted(expected)


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
        ), (
            'nwcall03: service to container',
            [
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),  # wrong
            ],
            [
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        ), (
            'nwcall04: container to service',
            [
                ("c-orders_cpu_usage_seconds_total", "s-user_latency"),  # wrong
            ],
            [
                ("s-user_latency", "c-orders_cpu_usage_seconds_total", {}),
            ],
        ), (
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
                ("c-user-db_cpu_usage_seconds_total", "c-user_cpu_usage_seconds_total", {}),
            ],
        )
    ],
    ids=['hieralchy01', 'hieralchy02', 'nwcall01', 'nwcall02', 'nwcall03', 'nwcall04', 'hybrid01'],
)
def test_fix_edge_directions_in_causal_graph(case, input, expected):
    G = nx.DiGraph()
    G.add_edges_from(input)
    got = diag.fix_edge_directions_in_causal_graph(G)
    assert sorted(list(nx.to_edgelist(got))) == sorted(expected)
