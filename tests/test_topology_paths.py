"""Topology construction and candidate path enumeration."""

from __future__ import annotations

import networkx as nx
import pytest

from qrp import paths as paths_mod
from qrp import topology as topo_mod


class TestCa9Graph:
    def test_nine_end_nodes(self):
        topo = topo_mod.build_ca9()
        assert len(topo.cities) == 9

    def test_site_count_in_expected_range(self):
        """Vault note 37 expects roughly 50 to 70 sites at 80 km spacing."""
        topo = topo_mod.build_ca9(spacing_km=80.0)
        assert 45 <= topo.n_candidates <= 70

    def test_every_segment_within_spacing(self):
        topo = topo_mod.build_ca9(spacing_km=80.0)
        for _, _, data in topo.graph.edges(data=True):
            assert data["length_km"] <= 80.0 + 1e-9

    def test_corridor_lengths_are_preserved(self):
        """Splitting a corridor must not change its total length."""
        topo = topo_mod.build_ca9(spacing_km=80.0)
        for (u, v), expected in topo_mod.CITY_EDGES.items():
            total = nx.shortest_path_length(topo.graph, u, v, weight="length_km")
            assert total == pytest.approx(expected)

    def test_two_components_by_default(self):
        topo = topo_mod.build_ca9()
        assert nx.number_connected_components(topo.graph) == 2

    def test_bridge_connects_the_halves(self):
        topo = topo_mod.build_ca9(bridge=True)
        assert nx.number_connected_components(topo.graph) == 1

    def test_demand_pairs_stay_within_components(self):
        topo = topo_mod.build_ca9()
        pairs = topo.demand_pairs()
        # 6 western cities -> 15 pairs, 3 eastern -> 3 pairs.
        assert len(pairs) == 15 + 3
        for a, b in pairs:
            assert nx.has_path(topo.graph, a, b)

    def test_finer_spacing_gives_more_sites(self):
        coarse = topo_mod.build_ca9(spacing_km=120.0)
        fine = topo_mod.build_ca9(spacing_km=50.0)
        assert fine.n_candidates > coarse.n_candidates

    def test_demand_weights(self):
        topo = topo_mod.build_ca9()
        assert topo.demand_weight("Vancouver", "Calgary") == 3
        assert topo.demand_weight("Calgary", "Vancouver") == 3
        assert topo.demand_weight("Kamloops", "Saskatoon") == 1


class TestDumbbell:
    def test_structure(self):
        topo = topo_mod.build_dumbbell(220.0, n_candidates=10)
        assert topo.n_candidates == 10
        assert len(topo.cities) == 2
        assert nx.shortest_path_length(
            topo.graph, "A", "B", weight="length_km"
        ) == pytest.approx(220.0)

    def test_segments_are_equal(self):
        topo = topo_mod.build_dumbbell(220.0, n_candidates=10)
        lengths = {round(d["length_km"], 9) for _, _, d in topo.graph.edges(data=True)}
        assert len(lengths) == 1


class TestLogicalGraph:
    def test_respects_max_link(self):
        topo = topo_mod.build_dumbbell(1100.0, n_candidates=10)
        logical = paths_mod.build_logical_graph(topo, max_link_km=300.0)
        for _, _, data in logical.edges(data=True):
            assert data["length_km"] <= 300.0 + 1e-9

    def test_bypassed_nodes_recorded(self):
        topo = topo_mod.build_dumbbell(1100.0, n_candidates=10)
        logical = paths_mod.build_logical_graph(topo, max_link_km=300.0)
        # A to the third site skips the two sites in between.
        assert logical["A"]["R3"]["bypassed"] == ("R1", "R2")


SEGMENT_KM = 880.0 / 11


def _dumbbell(strategy):
    topo = topo_mod.build_dumbbell(880.0, n_candidates=10)
    pairs = topo.demand_pairs()
    return paths_mod.enumerate_paths(
        topo, pairs, max_link_km=300.0, max_hops=11, strategy=strategy
    )


class TestPathEnumeration:
    @staticmethod
    @pytest.fixture(scope="class")
    def dumbbell_paths():
        return _dumbbell("candidates")

    def test_returns_paths_for_every_pair(self, dumbbell_paths):
        assert all(len(v) > 0 for v in dumbbell_paths.values())

    def test_endpoints_are_correct(self, dumbbell_paths):
        for (source, target), candidates in dumbbell_paths.items():
            for path in candidates:
                assert path.nodes[0] == source
                assert path.nodes[-1] == target

    def test_paths_are_simple(self, dumbbell_paths):
        for candidates in dumbbell_paths.values():
            for path in candidates:
                assert len(set(path.nodes)) == len(path.nodes)

    def test_link_count_matches_hops(self, dumbbell_paths):
        for candidates in dumbbell_paths.values():
            for path in candidates:
                assert len(path.link_lengths_km) == path.hops
                assert len(path.nodes) == path.hops + 1
                assert len(path.repeaters) == path.hops - 1

    def test_no_path_is_shorter_than_the_corridor(self, dumbbell_paths):
        """Simple paths may double back, but none can beat the straight run."""
        for candidates in dumbbell_paths.values():
            for path in candidates:
                assert path.total_km >= 880.0 - 1e-6

    def test_frontier_points_are_present(self, dumbbell_paths):
        """Every Pareto point over (hops, worst link) must be a candidate.

        With links capped at 300 km a link spans at most 3 of the 11 segments,
        so the frontier is 11 hops at 80 km, 6 at 160 km and 4 at 240 km.
        """
        kept = {(p.hops, round(p.max_link_km, 6))
                for candidates in dumbbell_paths.values() for p in candidates}
        for hops, segments in [(11, 1), (6, 2), (4, 3)]:
            assert (hops, round(segments * SEGMENT_KM, 6)) in kept

    def test_no_path_beats_the_physical_bound(self, dumbbell_paths):
        """An h-hop path needs a link of at least ceil(11 / h) segments."""
        for candidates in dumbbell_paths.values():
            for path in candidates:
                bound = SEGMENT_KM * -(-11 // path.hops)
                assert path.max_link_km >= bound - 1e-6

    def test_ca9_enumeration_runs(self):
        topo = topo_mod.build_ca9()
        pairs = topo.demand_pairs()
        result = paths_mod.enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)
        assert len(result) == len(pairs)
        assert sum(len(v) for v in result.values()) > 0
        for candidates in result.values():
            for path in candidates:
                assert len(set(path.nodes)) == len(path.nodes)
                assert path.hops <= 20


class TestLegacyStrategy:
    """The pre-September-2026 generator, kept to reproduce committed results."""

    @staticmethod
    @pytest.fixture(scope="class")
    def legacy_paths():
        return _dumbbell("legacy")

    def test_bottleneck_optimality(self, legacy_paths):
        """The kept path for each hop count minimises its longest link."""
        by_hops = {}
        for candidates in legacy_paths.values():
            for path in candidates:
                by_hops.setdefault(path.hops, []).append(path.max_link_km)
        for hops, values in by_hops.items():
            assert min(values) == pytest.approx(SEGMENT_KM * -(-11 // hops))

    def test_dominated_paths_are_dropped(self, legacy_paths):
        """No kept path may be beaten on hops and worst link at once."""
        for candidates in legacy_paths.values():
            for a in candidates:
                for b in candidates:
                    if a is b:
                        continue
                    strictly_better = (
                        b.hops <= a.hops
                        and b.max_link_km <= a.max_link_km + 1e-9
                        and (b.hops < a.hops or b.max_link_km < a.max_link_km - 1e-9)
                    )
                    assert not strictly_better

    def test_total_length_is_the_corridor_length(self, legacy_paths):
        for candidates in legacy_paths.values():
            for path in candidates:
                assert path.total_km == pytest.approx(880.0)

    def test_unknown_strategy_is_rejected(self):
        topo = topo_mod.build_dumbbell(220.0, n_candidates=3)
        with pytest.raises(ValueError):
            paths_mod.enumerate_paths(topo, topo.demand_pairs(), strategy="fastest")
