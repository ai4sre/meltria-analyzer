import networkx as nx
from diag_cause import diag


def test_fix_edge_directions_in_causal_graph():
    G = nx.DiGraph()
    G.add_edge("s-user_latency", "s-front-end_latency")
    G.add_edge("s-user_latency", "c-user_cpu_usage_seconds_total")
    got = diag.fix_edge_directions_in_causal_graph(G)

    assert got.has_edge("c-user_cpu_usage_seconds_total", "s-user_latency")
    assert not got.has_edge("s-user_latency", "c-user_cpu_usage_seconds_total")
    assert got.has_edge("s-user_latency", "s-front-end_latency")
