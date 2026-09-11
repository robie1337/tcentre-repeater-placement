"""The hop-limited k shortest simple paths behind two of the candidate families.

Until 11 September 2026 these families ranked every simple path by weight and
dropped the ones over the hop limit afterwards, from the first 120. The
cheapest paths by loss use many short links, so on the 40 km site grid all
120 were over 20 hops and the loss family contributed nothing.
"""

import random

import networkx as nx
import pytest

from qrp.paths import _k_shortest_within_hops


def _cost(graph, nodes, weight):
    return sum(graph[a][b][weight] for a, b in zip(nodes[:-1], nodes[1:]))


def _random_graph(seed, n=9, p=0.45):
    rng = random.Random(seed)
    graph = nx.gnp_random_graph(n, p, seed=seed)
    for u, v in graph.edges:
        # A few repeated lengths, so equal-cost paths are common.
        length = rng.choice([20.0, 40.0, 60.0, 80.0])
        graph[u][v]["length_km"] = length
        graph[u][v]["loss_weight"] = 10.0 ** (0.35 * length / 10.0)
    return graph


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("weight", ["length_km", "loss_weight"])
@pytest.mark.parametrize("max_hops", [2, 3, 5])
def test_matches_brute_force(seed, weight, max_hops):
    graph = _random_graph(seed)
    k = 6
    for source, target in [(0, graph.number_of_nodes() - 1), (1, 4)]:
        brute = sorted(
            _cost(graph, p, weight)
            for p in nx.all_simple_paths(graph, source, target, cutoff=max_hops)
        )
        found = _k_shortest_within_hops(graph, source, target, weight, k, max_hops)
        assert len(found) == min(k, len(brute))
        assert len(set(found)) == len(found)
        for nodes in found:
            assert nodes[0] == source and nodes[-1] == target
            assert len(set(nodes)) == len(nodes)
            assert len(nodes) - 1 <= max_hops
            assert all(graph.has_edge(a, b) for a, b in zip(nodes[:-1], nodes[1:]))
        assert [_cost(graph, p, weight) for p in found] == pytest.approx(brute[: len(found)])


def test_limit_acts_inside_the_search():
    """The cheapest route by loss is a ladder of short links over the limit."""
    graph = nx.Graph()
    ladder = ["A"] + [f"s{i}" for i in range(1, 12)] + ["B"]
    for a, b in zip(ladder[:-1], ladder[1:]):
        graph.add_edge(a, b, length_km=20.0)
    for mid in ("P", "Q"):
        graph.add_edge("A", f"{mid}1", length_km=90.0)
        graph.add_edge(f"{mid}1", f"{mid}2", length_km=90.0)
        graph.add_edge(f"{mid}2", "B", length_km=80.0)
    for _, _, data in graph.edges(data=True):
        data["loss_weight"] = 10.0 ** (0.35 * data["length_km"] / 10.0)

    assert tuple(nx.shortest_path(graph, "A", "B", weight="loss_weight")) == tuple(ladder)
    found = _k_shortest_within_hops(graph, "A", "B", "loss_weight", 6, max_hops=5)
    assert set(found) == {("A", "P1", "P2", "B"), ("A", "Q1", "Q2", "B")}


def test_ties_do_not_depend_on_edge_order():
    edges = [("A", "X", 40.0), ("X", "B", 40.0), ("A", "Y", 40.0), ("Y", "B", 40.0),
             ("A", "Z", 40.0), ("Z", "B", 40.0)]
    forward, backward = nx.Graph(), nx.Graph()
    for u, v, length in edges:
        forward.add_edge(u, v, length_km=length)
    for u, v, length in reversed(edges):
        backward.add_edge(v, u, length_km=length)
    assert (_k_shortest_within_hops(forward, "A", "B", "length_km", 2, 4)
            == _k_shortest_within_hops(backward, "A", "B", "length_km", 2, 4))
