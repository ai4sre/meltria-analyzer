import networkx as nx
import pytest
from diag_cause import diag


@pytest.mark.parametrize(
    "case,input,expected",
    [
        (
            'reverse single direction edge',
            [
                ("s-user_latency", "s-front-end_latency"),
                ("s-user_latency", "c-user_cpu_usage_seconds_total"),
            ],
            [
                ("s-user_latency", "s-front-end_latency", {}),
                ("c-user_cpu_usage_seconds_total", "s-user_latency", {}),
            ],
        )
    ],
    ids=['reverse_single'],
)
def test_fix_edge_directions_in_causal_graph(case, input, expected):
    G = nx.DiGraph()
    G.add_edges_from(input)
    got = diag.fix_edge_directions_in_causal_graph(G)
    assert list(nx.to_edgelist(got)) == expected
