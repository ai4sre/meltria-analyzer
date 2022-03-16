import networkx as nx


def reverse_edge_direction(G: nx.DiGraph, u, v) -> None:
    attr = G[u][v]
    G.remove_edge(u, v)
    G.add_edge(v, u, attr=attr) if attr else G.add_edge(v, u)


def set_bidirected_edge(G: nx.DiGraph, u, v) -> None:
    G.add_edge(u, v)
    G.add_edge(v, u)
