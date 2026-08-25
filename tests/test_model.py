"""Model assembly, solving, and the published results it must reproduce."""

from __future__ import annotations

import numpy as np
import pytest

from qrp import physics
from qrp.hardware import POURYOUSEF_BASELINE, TCENTRE_PROJECTED
from qrp.model import NetworkConfig, build_candidates, log_width_grid, solve_placement
from qrp.paths import enumerate_paths
from qrp.solver import INFEASIBLE_UTILITY, available_backends
from qrp.topology import build_dumbbell, build_line, build_ca9

PAPER_CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=True,
)


def dumbbell_case(backbone_km, n_candidates=10, max_hops=11):
    topo = build_dumbbell(backbone_km, n_candidates=n_candidates)
    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=backbone_km + 1.0, max_hops=max_hops)
    return topo, paths


class TestWidthGrid:
    def test_endpoints_present(self):
        grid = log_width_grid(100)
        assert grid[0] == 1
        assert grid[-1] == 100

    def test_sorted_and_unique(self):
        grid = log_width_grid(100)
        assert list(grid) == sorted(set(grid))

    def test_degenerate_case(self):
        assert log_width_grid(1) == (1,)


class TestCandidates:
    def test_coherence_prunes_long_paths(self):
        """A short repeater coherence time must remove every long link."""
        topo, paths = dumbbell_case(600.0)
        tight = POURYOUSEF_BASELINE.with_(t_repeater_memory_s=1e-6)
        assert build_candidates(paths, tight, PAPER_CONFIG) == []

    def test_low_fidelity_prunes_multi_hop(self):
        """At F_L = 0.6 a single swap already lands below the 1/2 floor."""
        topo, paths = dumbbell_case(600.0)
        weak = POURYOUSEF_BASELINE.with_(link_fidelity=0.6)
        candidates = build_candidates(paths, weak, PAPER_CONFIG)
        assert candidates
        assert all(c.path.hops == 1 for c in candidates)

    def test_utility_matches_physics(self):
        topo, paths = dumbbell_case(300.0)
        candidates = build_candidates(paths, POURYOUSEF_BASELINE, PAPER_CONFIG)
        for candidate in candidates[:20]:
            expected = physics.utility(candidate.rate, candidate.fidelity)
            assert candidate.utility == pytest.approx(expected)


class TestPublishedResults:
    """The reproductions that week 1 depends on."""

    def test_no_repeater_on_a_short_backbone(self):
        topo, paths = dumbbell_case(20.0)
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, PAPER_CONFIG)
        assert result.status == "optimal"
        assert result.n_repeaters == 0
        assert result.selections[0]["hops"] == 1

    def test_repeaters_appear_on_a_long_backbone(self):
        topo, paths = dumbbell_case(400.0)
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, PAPER_CONFIG)
        assert result.status == "optimal"
        assert result.n_repeaters > 0

    def test_transition_is_near_forty_km(self):
        """The paper puts the transition at roughly 40 km."""
        threshold = None
        for backbone in np.arange(10.0, 100.0, 2.5):
            topo, paths = dumbbell_case(float(backbone))
            result = solve_placement(topo, paths, POURYOUSEF_BASELINE, PAPER_CONFIG)
            if result.n_repeaters > 0:
                threshold = float(backbone)
                break
        assert threshold is not None
        assert 25.0 <= threshold <= 55.0

    @pytest.mark.parametrize("total_km", [200.0, 320.0])
    def test_coherence_cliff_tracks_two_l_over_c(self, total_km):
        """Below 2L/c of end-node coherence nothing is feasible."""
        topo = build_line(total_km, n_candidates=7)
        pairs = topo.demand_pairs()
        paths = enumerate_paths(topo, pairs, max_link_km=total_km + 1.0, max_hops=8)
        predicted = 2.0 * total_km / physics.C_FIBRE_KM_PER_S

        just_below = POURYOUSEF_BASELINE.with_(
            t_endnode_memory_s=predicted * 0.9, t_repeater_memory_s=predicted * 0.9
        )
        just_above = POURYOUSEF_BASELINE.with_(
            t_endnode_memory_s=predicted * 1.1, t_repeater_memory_s=predicted * 1.1
        )
        assert solve_placement(topo, paths, just_below, PAPER_CONFIG).status != "optimal"
        assert solve_placement(topo, paths, just_above, PAPER_CONFIG).status == "optimal"

    def test_infeasible_reports_the_paper_sentinel(self):
        topo, paths = dumbbell_case(600.0)
        tight = POURYOUSEF_BASELINE.with_(t_endnode_memory_s=1e-9)
        result = solve_placement(topo, paths, tight, PAPER_CONFIG)
        assert result.status != "optimal"
        assert result.utility == INFEASIBLE_UTILITY


class TestModelBehaviour:
    def test_repeaters_come_from_selected_paths(self):
        """Reported repeaters must be exactly the interior nodes in use."""
        topo, paths = dumbbell_case(400.0)
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, PAPER_CONFIG)
        expected = {node for sel in result.selections for node in sel["repeaters"]}
        assert set(result.repeaters) == expected
        assert result.n_repeaters == len(expected)

    def test_budget_is_respected(self):
        topo = build_ca9()
        pairs = topo.demand_pairs()
        paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)
        config = NetworkConfig(max_repeaters=6, require_all_pairs=False)
        result = solve_placement(topo, paths, TCENTRE_PROJECTED, config)
        assert result.n_repeaters <= 6

    def test_more_budget_never_hurts(self):
        topo = build_ca9()
        pairs = topo.demand_pairs()
        paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)
        utilities = []
        for budget in (5, 15, 40):
            config = NetworkConfig(max_repeaters=budget, require_all_pairs=False)
            utilities.append(
                solve_placement(topo, paths, TCENTRE_PROJECTED, config).utility
            )
        assert utilities == sorted(utilities)

    def test_better_hardware_never_hurts(self):
        topo, paths = dumbbell_case(400.0)
        base = POURYOUSEF_BASELINE
        better = base.with_(link_fidelity=0.99, swap_success=0.9)
        u_base = solve_placement(topo, paths, base, PAPER_CONFIG).utility
        u_better = solve_placement(topo, paths, better, PAPER_CONFIG).utility
        assert u_better > u_base

    def test_optional_service_never_loses_utility(self):
        topo = build_ca9()
        pairs = topo.demand_pairs()
        paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)
        optional = NetworkConfig(require_all_pairs=False, max_repeaters=10**6)
        result = solve_placement(topo, paths, TCENTRE_PROJECTED, optional)
        assert result.utility >= 0.0


@pytest.mark.skipif(
    len(available_backends()) < 2, reason="needs two solver backends"
)
class TestSolverAgreement:
    @pytest.mark.parametrize("backbone", [60.0, 240.0, 480.0])
    def test_backends_agree(self, backbone):
        topo, paths = dumbbell_case(backbone)
        values = [
            solve_placement(
                topo, paths, POURYOUSEF_BASELINE, PAPER_CONFIG,
                backend=backend, mip_gap=0.0,
            ).utility
            for backend in available_backends()
        ]
        assert max(values) - min(values) < 1e-6
