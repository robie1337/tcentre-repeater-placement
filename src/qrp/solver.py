"""Thin MILP backend.

The model is built once as sparse matrices and handed to whichever solver is
available. There is deliberately no algebraic modelling layer: at several
thousand solves the model-construction cost of Pyomo or PuLP would dominate
the solve itself, and the model here is simple enough to assemble directly.

Two backends:

``highs``   scipy.optimize.milp, which wraps HiGHS. MIT licensed, no
            registration, ships with scipy. This is the default so that the
            project runs anywhere.
``gurobi``  gurobipy, if it is installed and licensed. Faster, and the one
            named in the plan. Selected with backend="gurobi".

Running the same instance through both is the cross-check described in the
week 1 validation: identical objective values mean the answer comes from the
model rather than from the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

Backend = Literal["highs", "gurobi", "cplex"]

INFEASIBLE_UTILITY = -50.0
"""Sentinel reported when an instance has no feasible solution.

Pouryousef uses -50 for the same purpose in the coherence sweep, so results
here are directly comparable with their figures.
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
    status: str
    objective: float
    x: np.ndarray | None
    backend: str
    message: str = ""

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


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


def _solve_highs(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    from scipy.optimize import LinearConstraint, milp, Bounds

    # scipy.optimize.milp minimises, so negate the objective.
    constraints = LinearConstraint(problem.A, problem.lb, problem.ub)
    options: dict = {}
    if time_limit_s is not None:
        options["time_limit"] = time_limit_s
    if mip_gap > 0.0:
        options["mip_rel_gap"] = mip_gap

    res = milp(
        c=-problem.c,
        constraints=constraints,
        integrality=np.ones(problem.n_vars),
        bounds=Bounds(np.zeros(problem.n_vars), np.ones(problem.n_vars)),
        options=options or None,
    )

    if res.status == 0 and res.x is not None:
        return SolveResult("optimal", float(-res.fun), np.asarray(res.x), "highs", res.message)
    if res.status == 2:
        return SolveResult("infeasible", INFEASIBLE_UTILITY, None, "highs", res.message)
    return SolveResult("error", INFEASIBLE_UTILITY, None, "highs", res.message)


def _solve_gurobi(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    params = {"OutputFlag": 0, "Threads": 1}
    if mip_gap > 0.0:
        params["MIPGap"] = mip_gap
    with gp.Env(params=params) as env, gp.Model(env=env) as model:
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

        if model.Status == GRB.OPTIMAL:
            return SolveResult("optimal", float(model.ObjVal), np.array(x.X), "gurobi")
        if model.Status == GRB.INFEASIBLE:
            return SolveResult("infeasible", INFEASIBLE_UTILITY, None, "gurobi")
        return SolveResult("error", INFEASIBLE_UTILITY, None, "gurobi", f"status {model.Status}")


#: Variable ceiling of the free CPLEX Community Edition, measured on this
#: machine by adding binaries until it refuses. Error 1016 fires above this.
CPLEX_COMMUNITY_VAR_LIMIT = 1000


def _solve_cplex(
    problem: MilpProblem, time_limit_s: float | None, mip_gap: float = 1e-4
) -> SolveResult:
    """CPLEX backend, the solver the source paper used.

    The pip package ships the Community Edition, which is free and needs no
    registration but caps the problem at 1000 variables. Instances above that
    return a "too large" status rather than raising, so the cross-check can
    skip them and carry on.
    """
    import cplex

    if problem.n_vars > CPLEX_COMMUNITY_VAR_LIMIT:
        return SolveResult(
            "too_large", INFEASIBLE_UTILITY, None, "cplex",
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

    try:
        model.solve()
    except cplex.exceptions.CplexError as exc:
        return SolveResult("error", INFEASIBLE_UTILITY, None, "cplex", str(exc))

    status = model.solution.get_status_string()
    if model.solution.is_primal_feasible():
        return SolveResult(
            "optimal",
            float(model.solution.get_objective_value()),
            np.array(model.solution.get_values()),
            "cplex",
            status,
        )
    if "infeasible" in status.lower():
        return SolveResult("infeasible", INFEASIBLE_UTILITY, None, "cplex", status)
    return SolveResult("error", INFEASIBLE_UTILITY, None, "cplex", status)


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
