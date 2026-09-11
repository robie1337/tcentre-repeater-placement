"""Thin MILP backend.

The model is built once as sparse matrices and handed to whichever solver is
available. There is deliberately no algebraic modelling layer: at several
thousand solves the model-construction cost of Pyomo or PuLP would dominate
the solve itself, and the model here is simple enough to assemble directly.

Three backends:

``highs``   scipy.optimize.milp, which wraps HiGHS. MIT licensed, no
            registration, ships with scipy. This is the default so that the
            project runs anywhere.
``gurobi``  gurobipy, if it is installed and licensed.
``cplex``   the CPLEX Community Edition, capped at 1000 variables.

Running the same instance through several backends is the cross-check
described in the week 1 validation: identical objective values mean the
answer comes from the model rather than from the solver.

Solve outcomes
--------------
Every backend reports one of the statuses below, and the layers above keep
them apart. A time limit is a computational outcome, not a statement about
the hardware, so it must never be read as infeasibility.

``optimal``                 proven optimal within the requested relative gap
``feasible_at_limit``       stopped at a time, node or iteration limit holding
                            a feasible solution that is not proven optimal
``no_incumbent_at_limit``   stopped at a limit before finding any solution
``infeasible``              proven to have no feasible solution
``unbounded``               cannot happen for a binary model; kept for
                            completeness
``error``                   the backend failed or reported something else
``too_large``               over the CPLEX Community Edition size limit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

Backend = Literal["highs", "gurobi", "cplex"]

OPTIMAL = "optimal"
FEASIBLE_AT_LIMIT = "feasible_at_limit"
NO_INCUMBENT_AT_LIMIT = "no_incumbent_at_limit"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
ERROR = "error"
TOO_LARGE = "too_large"

#: Statuses that come with a usable solution vector.
HAS_SOLUTION = frozenset({OPTIMAL, FEASIBLE_AT_LIMIT})
#: Statuses that say nothing about the physics: the solve did not finish.
UNRESOLVED = frozenset({NO_INCUMBENT_AT_LIMIT, UNBOUNDED, ERROR, TOO_LARGE})

INFEASIBLE_UTILITY = -50.0
"""Sentinel utility for a *proven* infeasible instance.

Pouryousef uses -50 for the same purpose in the coherence sweep, so results
here are directly comparable with their figures. It is never used for a solve
that simply did not finish; those report NaN.
"""


@dataclass
class MilpProblem:
    """maximize c @ x subject to lb <= A @ x <= ub, x binary."""

    c: np.ndarray
    A: sp.csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    n_vars: int
    var_names: list[str] = field(default_factory=list)


@dataclass
class SolveResult:
    """What one solve produced. ``objective`` is NaN unless a solution exists."""

    status: str
    objective: float
    x: np.ndarray | None
    backend: str
    message: str = ""
    mip_gap: float = float("nan")
    best_bound: float = float("nan")
    runtime_s: float = float("nan")

    @property
    def optimal(self) -> bool:
        return self.status == OPTIMAL

    @property
    def has_solution(self) -> bool:
        return self.status in HAS_SOLUTION and self.x is not None


def _nan_if_none(value) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def solve(
    problem: MilpProblem,
    backend: Backend = "highs",
    time_limit_s: float | None = None,
    mip_gap: float = 1e-4,
) -> SolveResult:
    """Solve one instance.

    ``mip_gap`` is a relative optimality tolerance. At 1e-4 the reported
    utility is within 0.01% of optimal, which is far below the resolution of
    anything the sweep reports, and it cuts solve time substantially by
    letting the solver stop proving optimality once the bound is tight.
    Set it to 0.0 for an exact solve.
    """
    if backend == "gurobi":
        return _solve_gurobi(problem, time_limit_s, mip_gap)
    if backend == "highs":
        return _solve_highs(problem, time_limit_s, mip_gap)
    if backend == "cplex":
        return _solve_cplex(problem, time_limit_s, mip_gap)
    raise ValueError(f"unknown backend {backend!r}")


# --------------------------------------------------------------------------
# HiGHS
# --------------------------------------------------------------------------


def _solve_highs(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    from scipy.optimize import Bounds, LinearConstraint, milp

    constraints = LinearConstraint(problem.A, problem.lb, problem.ub)
    options: dict = {}
    if time_limit_s is not None:
        options["time_limit"] = time_limit_s
    if mip_gap > 0.0:
        options["mip_rel_gap"] = mip_gap

    started = time.perf_counter()
    # scipy.optimize.milp minimises, so negate the objective.
    res = milp(
        c=-problem.c,
        constraints=constraints,
        integrality=np.ones(problem.n_vars),
        bounds=Bounds(np.zeros(problem.n_vars), np.ones(problem.n_vars)),
        options=options or None,
    )
    return _highs_result(res, time.perf_counter() - started)


def _highs_result(res, runtime_s: float) -> SolveResult:
    """Map a scipy.optimize.milp result onto the shared statuses.

    scipy reports 0 optimal, 1 iteration or time limit, 2 infeasible,
    3 unbounded, 4 anything else. Objective and bound are negated back,
    because the problem was handed to milp as a minimisation.
    """
    x = None if getattr(res, "x", None) is None else np.asarray(res.x)
    fun = getattr(res, "fun", None)
    objective = -float(fun) if (x is not None and fun is not None) else float("nan")
    common = dict(
        backend="highs",
        message=str(getattr(res, "message", "")),
        mip_gap=_nan_if_none(getattr(res, "mip_gap", None)),
        best_bound=-_nan_if_none(getattr(res, "mip_dual_bound", None)),
        runtime_s=runtime_s,
    )
    status = getattr(res, "status", None)
    if status == 0 and x is not None:
        return SolveResult(OPTIMAL, objective, x, **common)
    if status == 1:
        if x is not None:
            return SolveResult(FEASIBLE_AT_LIMIT, objective, x, **common)
        return SolveResult(NO_INCUMBENT_AT_LIMIT, float("nan"), None, **common)
    if status == 2:
        return SolveResult(INFEASIBLE, float("nan"), None, **common)
    if status == 3:
        return SolveResult(UNBOUNDED, float("nan"), None, **common)
    return SolveResult(ERROR, float("nan"), None, **common)


# --------------------------------------------------------------------------
# Gurobi
# --------------------------------------------------------------------------


def _gurobi_status(code: int, sol_count: int, grb) -> str:
    """Map a Gurobi status code onto the shared statuses.

    INF_OR_UNBD counts as infeasible: every variable here is binary, so the
    model cannot be unbounded.
    """
    limits = {
        grb.TIME_LIMIT,
        grb.ITERATION_LIMIT,
        grb.NODE_LIMIT,
        grb.SOLUTION_LIMIT,
        grb.INTERRUPTED,
        grb.SUBOPTIMAL,
        getattr(grb, "WORK_LIMIT", -1),
        getattr(grb, "MEM_LIMIT", -1),
    }
    if code == grb.OPTIMAL:
        return OPTIMAL
    if code in limits:
        return FEASIBLE_AT_LIMIT if sol_count > 0 else NO_INCUMBENT_AT_LIMIT
    if code in (grb.INFEASIBLE, grb.INF_OR_UNBD):
        return INFEASIBLE
    if code == grb.UNBOUNDED:
        return UNBOUNDED
    return ERROR


#: GRB_ERROR_SIZE_LIMIT_EXCEEDED: the model is larger than the free
#: size-limited licence allows.
GUROBI_SIZE_LIMIT_ERRNO = 10010


def _gurobi_error_status(errno: int) -> str:
    """Status for a Gurobi exception: too large for the licence, or an error."""
    return TOO_LARGE if errno == GUROBI_SIZE_LIMIT_ERRNO else ERROR


def _solve_gurobi(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    params = {"OutputFlag": 0, "Threads": 1}
    if mip_gap > 0.0:
        params["MIPGap"] = mip_gap
    with gp.Env(params=params) as env, gp.Model(env=env) as model:
        try:
            x = model.addMVar(problem.n_vars, vtype=GRB.BINARY)
            model.setObjective(problem.c @ x, GRB.MAXIMIZE)

            finite_ub = np.isfinite(problem.ub)
            finite_lb = np.isfinite(problem.lb)
            equality = finite_ub & finite_lb & np.isclose(problem.lb, problem.ub)

            if equality.any():
                model.addConstr(problem.A[equality] @ x == problem.ub[equality])
            upper_only = finite_ub & ~equality
            if upper_only.any():
                model.addConstr(problem.A[upper_only] @ x <= problem.ub[upper_only])
            lower_only = finite_lb & ~equality
            if lower_only.any():
                model.addConstr(problem.A[lower_only] @ x >= problem.lb[lower_only])

            if time_limit_s is not None:
                model.setParam("TimeLimit", time_limit_s)
            model.optimize()
        except gp.GurobiError as exc:
            return SolveResult(_gurobi_error_status(exc.errno), float("nan"), None, "gurobi",
                               str(exc))

        status = _gurobi_status(model.Status, model.SolCount, GRB)
        has = status in HAS_SOLUTION
        try:
            bound = float(model.ObjBound)
        except (gp.GurobiError, AttributeError):
            bound = float("nan")
        return SolveResult(
            status,
            float(model.ObjVal) if has else float("nan"),
            np.array(x.X) if has else None,
            "gurobi",
            f"status {model.Status}",
            mip_gap=float(model.MIPGap) if has else float("nan"),
            best_bound=bound,
            runtime_s=float(model.Runtime),
        )


# --------------------------------------------------------------------------
# CPLEX
# --------------------------------------------------------------------------

#: Variable ceiling of the free CPLEX Community Edition, measured on this
#: machine by adding binaries until it refuses. Error 1016 fires above this.
CPLEX_COMMUNITY_VAR_LIMIT = 1000

_CPLEX_OPTIMAL = {"MIP_optimal", "MIP_optimal_tolerance", "optimal"}
_CPLEX_INFEASIBLE = {"MIP_infeasible", "infeasible", "MIP_infeasible_or_unbounded",
                     "infeasible_or_unbounded"}
_CPLEX_UNBOUNDED = {"MIP_unbounded", "unbounded"}


def _cplex_status(name: str, primal_feasible: bool) -> str:
    """Map a CPLEX status name onto the shared statuses.

    Limit and abort statuses end in ``_feasible`` or ``_infeasible`` according
    to whether an incumbent exists. Anything unrecognised that still holds a
    feasible solution is reported as not proven optimal rather than optimal.
    """
    if name in _CPLEX_OPTIMAL:
        return OPTIMAL
    if name in _CPLEX_INFEASIBLE:
        return INFEASIBLE
    if name in _CPLEX_UNBOUNDED:
        return UNBOUNDED
    if "limit" in name or "abort" in name or name.startswith("MIP_fail"):
        return FEASIBLE_AT_LIMIT if primal_feasible else NO_INCUMBENT_AT_LIMIT
    return FEASIBLE_AT_LIMIT if primal_feasible else ERROR


def _solve_cplex(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    """CPLEX backend, the solver the source paper used.

    The pip package ships the Community Edition, which is free and needs no
    registration but caps the problem at 1000 variables. Instances above that
    return a "too_large" status rather than raising, so the cross-check can
    skip them and carry on.
    """
    import cplex

    if problem.n_vars > CPLEX_COMMUNITY_VAR_LIMIT:
        return SolveResult(
            TOO_LARGE, float("nan"), None, "cplex",
            f"{problem.n_vars} variables exceeds the Community Edition limit "
            f"of {CPLEX_COMMUNITY_VAR_LIMIT}",
        )

    model = cplex.Cplex()
    for stream in ("log", "results", "warning", "error"):
        getattr(model, f"set_{stream}_stream")(None)

    model.variables.add(
        obj=[float(v) for v in problem.c],
        lb=[0.0] * problem.n_vars,
        ub=[1.0] * problem.n_vars,
        types=["B"] * problem.n_vars,
    )
    model.objective.set_sense(model.objective.sense.maximize)

    csr = problem.A.tocsr()
    rows, senses, rhs = [], [], []
    for i in range(csr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        indices = [int(j) for j in csr.indices[start:end]]
        values = [float(v) for v in csr.data[start:end]]
        lower, upper = problem.lb[i], problem.ub[i]
        if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
            senses.append("E")
            rhs.append(float(upper))
        elif np.isfinite(upper):
            senses.append("L")
            rhs.append(float(upper))
        else:
            senses.append("G")
            rhs.append(float(lower))
        rows.append(cplex.SparsePair(ind=indices, val=values))

    model.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)

    if time_limit_s is not None:
        model.parameters.timelimit.set(time_limit_s)
    if mip_gap > 0.0:
        model.parameters.mip.tolerances.mipgap.set(mip_gap)

    started = time.perf_counter()
    try:
        model.solve()
    except cplex.exceptions.CplexError as exc:
        return SolveResult(ERROR, float("nan"), None, "cplex", str(exc),
                           runtime_s=time.perf_counter() - started)
    runtime = time.perf_counter() - started

    code = model.solution.get_status()
    try:
        name = model.solution.status[code]
    except Exception:  # pragma: no cover - older API without the name table
        name = model.solution.get_status_string()
    feasible = bool(model.solution.is_primal_feasible())
    status = _cplex_status(name, feasible)
    has = status in HAS_SOLUTION

    gap = bound = float("nan")
    if has:
        try:
            gap = float(model.solution.MIP.get_mip_relative_gap())
        except cplex.exceptions.CplexError:
            pass
    try:
        bound = float(model.solution.MIP.get_best_objective())
    except cplex.exceptions.CplexError:
        pass

    return SolveResult(
        status,
        float(model.solution.get_objective_value()) if has else float("nan"),
        np.array(model.solution.get_values()) if has else None,
        "cplex",
        name,
        mip_gap=gap,
        best_bound=bound,
        runtime_s=runtime,
    )


def available_backends() -> list[str]:
    """Which backends can actually run here. Printed by the scripts."""
    found = []
    try:
        import scipy.optimize  # noqa: F401

        found.append("highs")
    except ImportError:
        pass
    try:
        import gurobipy  # noqa: F401

        found.append("gurobi")
    except ImportError:
        pass
    try:
        import cplex  # noqa: F401

        found.append("cplex")
    except ImportError:
        pass
    return found
