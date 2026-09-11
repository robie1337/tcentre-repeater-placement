"""Solve outcomes: a time limit must never read as infeasibility."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

import qrp.model as model_module
from qrp.hardware import POURYOUSEF_BASELINE
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from qrp.sensitivity import analyse, saltelli_points
from qrp.solver import (
    ERROR,
    FEASIBLE_AT_LIMIT,
    INFEASIBLE,
    INFEASIBLE_UTILITY,
    NO_INCUMBENT_AT_LIMIT,
    OPTIMAL,
    UNBOUNDED,
    SolveResult,
    _cplex_status,
    _gurobi_status,
    _highs_result,
    available_backends,
)
from qrp.topology import build_dumbbell


def fake_milp(status, x=None, fun=None, gap=None, bound=None):
    return types.SimpleNamespace(status=status, x=x, fun=fun, message="fake",
                                 mip_gap=gap, mip_dual_bound=bound)


class TestHighsMapping:
    def test_optimal(self):
        r = _highs_result(fake_milp(0, x=np.array([1.0]), fun=-3.0, gap=0.0, bound=-3.0), 0.1)
        assert r.status == OPTIMAL
        assert r.optimal and r.has_solution
        assert r.objective == 3.0
        assert r.best_bound == 3.0

    def test_limit_with_incumbent_is_not_optimal(self):
        r = _highs_result(fake_milp(1, x=np.array([1.0]), fun=-2.0, gap=0.2), 1.0)
        assert r.status == FEASIBLE_AT_LIMIT
        assert r.has_solution and not r.optimal
        assert r.objective == 2.0
        assert r.mip_gap == 0.2

    def test_limit_without_incumbent(self):
        r = _highs_result(fake_milp(1), 1.0)
        assert r.status == NO_INCUMBENT_AT_LIMIT
        assert not r.has_solution
        assert np.isnan(r.objective)

    @pytest.mark.parametrize("code, expected", [(2, INFEASIBLE), (3, UNBOUNDED), (4, ERROR)])
    def test_other_codes(self, code, expected):
        r = _highs_result(fake_milp(code), 0.0)
        assert r.status == expected
        assert not r.has_solution


class TestCplexMapping:
    @pytest.mark.parametrize("name, feasible, expected", [
        ("MIP_optimal", True, OPTIMAL),
        ("MIP_optimal_tolerance", True, OPTIMAL),
        ("MIP_time_limit_feasible", True, FEASIBLE_AT_LIMIT),
        ("MIP_time_limit_infeasible", False, NO_INCUMBENT_AT_LIMIT),
        ("MIP_node_limit_feasible", True, FEASIBLE_AT_LIMIT),
        ("MIP_infeasible", False, INFEASIBLE),
        ("MIP_infeasible_or_unbounded", False, INFEASIBLE),
        ("MIP_unbounded", False, UNBOUNDED),
        ("something_unrecognised", False, ERROR),
        ("something_unrecognised", True, FEASIBLE_AT_LIMIT),
        # The pip CPLEX names status 102 without the MIP_ prefix.
        ("optimal_tolerance", True, OPTIMAL),
    ])
    def test_names(self, name, feasible, expected):
        assert _cplex_status(name, feasible) == expected


def test_zero_gap_reaches_highs(monkeypatch):
    """mip_gap=0.0 must be passed on, not dropped in favour of the solver default."""
    import scipy.optimize
    import scipy.sparse as sp

    from qrp.solver import MilpProblem, solve

    seen = {}

    def fake_solver(**kwargs):
        seen.update(kwargs.get("options") or {})
        return fake_milp(0, x=np.zeros(2), fun=0.0, gap=0.0, bound=0.0)

    monkeypatch.setattr(scipy.optimize, "milp", fake_solver)
    problem = MilpProblem(c=np.zeros(2), A=sp.csr_matrix((1, 2)), lb=np.array([-np.inf]),
                          ub=np.array([1.0]), n_vars=2)
    solve(problem, backend="highs", mip_gap=0.0)
    assert seen.get("mip_rel_gap") == 0.0


def test_gurobi_licence_size_limit_is_too_large():
    """The free Gurobi licence raises on a large model; that is not a failed solve."""
    from qrp.solver import TOO_LARGE, _gurobi_error_status

    assert _gurobi_error_status(10010) == TOO_LARGE
    assert _gurobi_error_status(10001) == ERROR


@pytest.mark.skipif("gurobi" not in available_backends(), reason="gurobipy not installed")
class TestGurobiMapping:
    def test_codes(self):
        from gurobipy import GRB

        assert _gurobi_status(GRB.OPTIMAL, 1, GRB) == OPTIMAL
        assert _gurobi_status(GRB.TIME_LIMIT, 1, GRB) == FEASIBLE_AT_LIMIT
        assert _gurobi_status(GRB.TIME_LIMIT, 0, GRB) == NO_INCUMBENT_AT_LIMIT
        assert _gurobi_status(GRB.INFEASIBLE, 0, GRB) == INFEASIBLE
        assert _gurobi_status(GRB.INF_OR_UNBD, 0, GRB) == INFEASIBLE
        assert _gurobi_status(GRB.UNBOUNDED, 0, GRB) == UNBOUNDED


class TestPlacementResult:
    @staticmethod
    def case():
        topo = build_dumbbell(400.0, n_candidates=10)
        paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=401.0, max_hops=11)
        return topo, paths

    def test_feasible_at_limit_keeps_its_solution(self, monkeypatch):
        topo, paths = self.case()
        real = model_module.solve

        def limited(problem, **kwargs):
            return dataclasses.replace(real(problem, **kwargs), status=FEASIBLE_AT_LIMIT,
                                       mip_gap=0.05)

        monkeypatch.setattr(model_module, "solve", limited)
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, NetworkConfig())
        assert result.status == FEASIBLE_AT_LIMIT
        assert np.isfinite(result.utility)
        assert result.selections
        assert result.mip_gap == 0.05

    def test_no_incumbent_is_not_infeasibility(self, monkeypatch):
        topo, paths = self.case()
        monkeypatch.setattr(
            model_module, "solve",
            lambda problem, **kwargs: SolveResult(NO_INCUMBENT_AT_LIMIT, float("nan"), None, "highs"),
        )
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, NetworkConfig())
        assert result.status == NO_INCUMBENT_AT_LIMIT
        assert np.isnan(result.utility)
        assert result.utility != INFEASIBLE_UTILITY

    def test_proven_infeasible_keeps_the_sentinel(self, monkeypatch):
        topo, paths = self.case()
        monkeypatch.setattr(
            model_module, "solve",
            lambda problem, **kwargs: SolveResult(INFEASIBLE, float("nan"), None, "highs"),
        )
        result = solve_placement(topo, paths, POURYOUSEF_BASELINE, NetworkConfig())
        assert result.status == INFEASIBLE
        assert result.utility == INFEASIBLE_UTILITY


class TestSensitivityStatuses:
    @staticmethod
    def frame(overrides):
        points, problem = saltelli_points(8, names=("generation_rate_hz", "t2_s"))
        frame = pd.DataFrame(points)
        frame["utility"] = np.random.default_rng(0).normal(size=len(frame))
        frame["status"] = OPTIMAL
        for index, status in overrides.items():
            frame.loc[index, "status"] = status
            if status not in (OPTIMAL, FEASIBLE_AT_LIMIT):
                frame.loc[index, "utility"] = float("nan")
        return frame, problem

    def test_unresolved_solves_raise(self):
        frame, problem = self.frame({3: NO_INCUMBENT_AT_LIMIT})
        with pytest.raises(ValueError, match="not infeasibility"):
            analyse(frame, problem)

    def test_unresolved_can_be_allowed_and_are_counted(self):
        frame, problem = self.frame({3: NO_INCUMBENT_AT_LIMIT, 5: INFEASIBLE})
        result = analyse(frame, problem, allow_unresolved=True)
        assert result.n_unresolved == 1
        assert result.n_infeasible == 1
        assert "unresolved" in result.summary()

    def test_not_proven_is_counted_separately(self):
        frame, problem = self.frame({2: FEASIBLE_AT_LIMIT})
        result = analyse(frame, problem)
        assert result.n_not_proven == 1
        assert result.n_infeasible == 0
        assert result.n_unresolved == 0
