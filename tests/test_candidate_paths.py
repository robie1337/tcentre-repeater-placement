"""Candidate path generation against exhaustive enumeration.

The network MILP couples user pairs through shared repeater memory and the
repeater budget, so no per-path pruning rule is safe. These tests pin down the
failures of the legacy generator on purpose-built networks and check the
default generator against every simple path on small random networks.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from qrp.hardware import POURYOUSEF_BASELINE
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from qrp.topology import Topology, build_ca9

#: The paper's baseline with a generation rate high enough that serving a
#: pair has positive utility, so the optimiser wants to serve both pairs.
FAST = POURYOUSEF_BASELINE.with_(generation_rate_hz=1e4)


def small_topology(edges, cities, name="test") -> Topology:
    graph = nx.Graph()
    for u, v, length in edges:
        for node in (u, v):
            if node not in graph:
                graph.add_node(node, kind="city" if node in cities else "site")
        graph.add_edge(u, v, length_km=float(length))
    sites = tuple(n for n in graph.nodes if n not in set(cities))
    return Topology(graph=graph, cities=tuple(cities), sites=sites, name=name, spacing_km=0.0)


def repeater_sets(candidates) -> set[frozenset]:
    return {frozenset(p.repeaters) for p in candidates}


class TestSharedRepeaterCounterexample:
    """A locally dominated path is the one the global optimum needs.

    A-B can go through S1 (2 hops) or through S2 and S3 (3 hops, same worst
    link), so on its own the 3-hop route is dominated. C-D can only use S2 and
    S3. With a budget of two repeaters, serving both pairs needs A-B to share
    S2 and S3, which the legacy generator has already thrown away.
    """

    EDGES = [("A", "S1", 40), ("S1", "B", 40), ("A", "S2", 40), ("S2", "S3", 40),
             ("S3", "B", 40), ("C", "S2", 40), ("S3", "D", 40)]
    CONFIG = NetworkConfig(repeater_memories=100, endnode_memories=100, max_repeaters=2,
                           require_all_pairs=False, use_demand_weights=False)

    def solve(self, strategy):
        topo = small_topology(self.EDGES, ["A", "B", "C", "D"])
        paths = enumerate_paths(topo, [("A", "B"), ("C", "D")], max_link_km=45.0,
                                max_hops=8, strategy=strategy)
        return paths, solve_placement(topo, paths, FAST, self.CONFIG, mip_gap=0.0)

    def test_legacy_drops_the_shared_route(self):
        paths, _ = self.solve("legacy")
        assert frozenset({"S2", "S3"}) not in repeater_sets(paths[("A", "B")])

    def test_legacy_loses_a_pair(self):
        _, legacy = self.solve("legacy")
        _, full = self.solve("exhaustive")
        assert legacy.served_pairs == 1
        assert full.served_pairs == 2
        assert legacy.utility < full.utility - 1.0

    def test_default_generator_recovers_the_optimum(self):
        paths, result = self.solve("candidates")
        _, full = self.solve("exhaustive")
        assert frozenset({"S2", "S3"}) in repeater_sets(paths[("A", "B")])
        assert result.served_pairs == 2
        assert result.utility == pytest.approx(full.utility)


class TestTiedAlternatives:
    """Two routes of equal quality through different repeaters must both survive."""

    EDGES = [("A", "S1", 40), ("S1", "B", 40), ("A", "S2", 40), ("S2", "B", 40)]

    def paths(self, strategy):
        topo = small_topology(self.EDGES, ["A", "B"])
        return enumerate_paths(topo, [("A", "B")], max_link_km=45.0, max_hops=4,
                               strategy=strategy)[("A", "B")]

    def test_default_keeps_both(self):
        assert {frozenset({"S1"}), frozenset({"S2"})} <= repeater_sets(self.paths("candidates"))

    def test_legacy_keeps_only_one(self):
        assert len([p for p in self.paths("legacy") if p.hops == 2]) == 1


class TestCyclicWalkTrap:
    """The best 4-hop walk revisits S1, but a simple 4-hop route exists."""

    EDGES = [("A", "S1", 12), ("S1", "B", 12), ("A", "P", 20), ("P", "Q", 20),
             ("Q", "R", 20), ("R", "B", 20)]
    ROUTE = ("A", "P", "Q", "R", "B")

    def paths(self, strategy):
        topo = small_topology(self.EDGES, ["A", "B"])
        return enumerate_paths(topo, [("A", "B")], max_link_km=22.0, max_hops=6,
                               strategy=strategy)[("A", "B")]

    def test_legacy_loses_the_route(self):
        assert self.ROUTE not in {p.nodes for p in self.paths("legacy")}

    def test_default_keeps_it(self):
        assert self.ROUTE in {p.nodes for p in self.paths("candidates")}


def random_topology(seed: int, n_cities: int = 3, n_sites: int = 4) -> Topology:
    rng = np.random.default_rng(seed)
    cities = [f"C{i}" for i in range(n_cities)]
    sites = [f"S{i}" for i in range(n_sites)]
    nodes = cities + sites
    edges = []
    order = list(rng.permutation(nodes))
    for a, b in zip(order[:-1], order[1:]):  # a random spanning path keeps it connected
        edges.append((a, b, float(rng.uniform(20, 60))))
    present = {frozenset((a, b)) for a, b, _ in edges}
    for a, b in itertools.combinations(nodes, 2):
        if frozenset((a, b)) not in present and rng.random() < 0.35:
            edges.append((a, b, float(rng.uniform(20, 60))))
    topo = small_topology(edges, cities, name=f"random-{seed}")
    for node in nodes:  # isolated nodes cannot occur, but keep every node present
        topo.graph.add_node(node, kind="city" if node in cities else "site")
    return topo


@pytest.mark.parametrize("rate_model", ["paper", "coordinated"])
@pytest.mark.parametrize("seed", range(12))
def test_default_matches_exhaustive_on_small_networks(seed, rate_model):
    """On small congested networks the default set must reach the true optimum."""
    rng = np.random.default_rng(1000 + seed)
    topo = random_topology(seed)
    pairs = topo.demand_pairs()
    config = NetworkConfig(
        repeater_memories=8,
        endnode_memories=8,
        max_repeaters=int(rng.integers(1, 4)),
        require_all_pairs=False,
        use_demand_weights=False,
        rate_model=rate_model,
    )
    results = {}
    for strategy in ("exhaustive", "candidates"):
        paths = enumerate_paths(topo, pairs, max_link_km=70.0, max_hops=5, strategy=strategy)
        results[strategy] = solve_placement(topo, paths, FAST, config, mip_gap=0.0)
    full, reduced = results["exhaustive"], results["candidates"]
    assert full.status == "optimal" and reduced.status == "optimal"
    assert reduced.utility >= full.utility - 1e-6 * max(1.0, abs(full.utility))


def test_exhaustive_refuses_large_graphs():
    topo = build_ca9()
    with pytest.raises(ValueError):
        enumerate_paths(topo, [("Kamloops", "Kelowna")], max_link_km=300.0, max_hops=20,
                        strategy="exhaustive", max_exhaustive=1000)
