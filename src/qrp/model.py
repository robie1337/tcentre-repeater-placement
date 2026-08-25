"""The repeater placement MILP.

This is Pouryousef's path-based formulation, Problem 1, with the hardware
inputs swapped for T centre values. The objective and the constraint
structure are unchanged; what changes is the attenuation constant, the link
fidelity, the swap success probability and the coherence budgets.

How the non-linear objective stays linear
-----------------------------------------
Utility is log2(R * (F - 1/2)), which is not linear in the decision
variables. The paper handles this by enumerating each path together with each
path width and evaluating the utility offline, so that each (path, width)
combination carries a constant objective coefficient. We do the same. The
solver therefore never sees a logarithm.

Widths are sampled on a log-spaced grid rather than enumerated exhaustively.
Utility grows as log2(W), so a log-spaced grid samples the objective
uniformly. The paper enumerates every width up to min(D, W_E), which for
D = 100 means a hundred variables per path where eight capture the same range
to within a fraction of a bit. This is an approximation and is recorded as
one; set ``width_grid=range(1, D+1)`` to reproduce the paper exactly.

Decision variables
------------------
y[q, p, w]  binary, user pair q is served by path p at width w
r[u]        binary, a repeater is installed at candidate location u

Constraints, with the paper's equation numbers
----------------------------------------------
(9)   sum of w * y over paths through u  <=  D * r[u]      per candidate u
(10)  sum of y over (p, w) for pair q    ==  1             per user pair
(11)  sum of w * y over paths ending at v <= W_E           per end node v
(12)  sum of r                            <= N_max
(13)  2 * tau_l(longest link) <= T_RM        enforced by pruning candidates
(14)  tau_e2e(path)           <= T_EM        enforced by pruning candidates

Constraints (13) and (14) do not depend on the decision variables, only on
the path, so a path violating either can never be used and is dropped before
the model is built. That is equivalent to including them and cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import physics
from .hardware import Hardware
from .paths import CandidatePath
from .solver import INFEASIBLE_UTILITY, MilpProblem, SolveResult, solve
from .topology import Topology


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def log_width_grid(max_width: int, n_points: int = 8) -> tuple[int, ...]:
    """Log-spaced integer widths from 1 to ``max_width``."""
    if max_width < 1:
        raise ValueError("max_width must be at least 1")
    if max_width == 1:
        return (1,)
    raw = np.geomspace(1, max_width, num=n_points)
    return tuple(sorted({int(round(v)) for v in raw}))


@dataclass(frozen=True)
class NetworkConfig:
    """Everything that is not hardware and not topology.

    Defaults follow the paper's dumbbell experiment: D = W_E = 100, one path
    per user pair, and a repeater budget that does not bind.
    """

    repeater_memories: int = 100          # D_u
    endnode_memories: int = 100           # W_E
    max_repeaters: int = 10**6            # N_max
    paths_per_pair: int = 1               # K
    width_grid: tuple[int, ...] | None = None
    require_all_pairs: bool = True
    use_demand_weights: bool = False
    cities_can_host_repeaters: bool = True
    #: Which end-to-end rate expression feeds the utility coefficients.
    #: "paper" is Eq. (2), R = q^(h-1) W p_min, the source paper's optimistic
    #: pipelined model, kept as the default so every validation reproduces.
    #: "ext" is Q-CAST's exact slotted throughput (Shi & Qian, SIGCOMM 2020),
    #: the literature-anchored low-W*p replacement. "coordinated" is the
    #: buffered waiting-time model (Bernardes et al. 2011 structure). The
    #: truth lies inside the bracket the three of them span.
    rate_model: str = "paper"

    def widths(self) -> tuple[int, ...]:
        if self.width_grid is not None:
            return self.width_grid
        return log_width_grid(min(self.repeater_memories, self.endnode_memories))


# --------------------------------------------------------------------------
# Candidate assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One (user pair, path, width) combination with its utility."""

    pair: tuple[str, str]
    path: CandidatePath
    width: int
    utility: float
    rate: float
    fidelity: float
    p_min: float
    tau_e2e_s: float
    wp_min: float


def build_candidates(
    paths: dict[tuple[str, str], list[CandidatePath]],
    hardware: Hardware,
    config: NetworkConfig,
    demand_weight=None,
) -> list[Candidate]:
    """Evaluate every (path, width) pair and drop the unusable ones.

    A combination is dropped when the round-trip time on its longest link
    exceeds the repeater coherence time, when its end-to-end time exceeds the
    end-node coherence time, or when its utility is not finite (fidelity at or
    below the classical floor of 1/2).
    """
    widths = config.widths()
    out: list[Candidate] = []

    for pair, pair_paths in paths.items():
        weight = 1.0
        if config.use_demand_weights and demand_weight is not None:
            weight = float(demand_weight(*pair))

        for path in pair_paths:
            # Eq. (13): the longest link's round trip must fit repeater memory.
            if physics.tau_repeater_deadline(path.max_link_km) > hardware.t_repeater_memory_s:
                continue
            # Eq. (14): end-to-end time must fit end-node memory.
            tau = physics.tau_e2e(path.link_lengths_km)
            if tau > hardware.t_endnode_memory_s:
                continue

            link_probs = [
                physics.link_success(
                    length,
                    hardware.alpha_db_per_km,
                    hardware.eta_emission,
                    hardware.eta_detection,
                )
                for length in path.link_lengths_km
            ]
            p_min = min(link_probs)
            fidelity = physics.e2e_fidelity(
                hardware.link_fidelity,
                path.hops,
                hardware.gate_fidelity,
                hardware.measurement_fidelity,
                swap_werner=hardware.swap_werner,
            )
            if fidelity <= 0.5:
                continue

            for width in widths:
                if config.rate_model == "coordinated":
                    rate = physics.e2e_rate_coordinated(
                        hardware.swap_success,
                        path.hops,
                        width,
                        link_probs,
                        hardware.generation_rate_hz,
                    )
                elif config.rate_model == "ext":
                    rate = physics.e2e_rate_ext(
                        hardware.swap_success,
                        path.hops,
                        width,
                        link_probs,
                        hardware.generation_rate_hz,
                    )
                elif config.rate_model == "paper":
                    rate = physics.e2e_rate(
                        hardware.swap_success,
                        path.hops,
                        width,
                        p_min,
                        hardware.generation_rate_hz,
                    )
                else:
                    raise ValueError(f"unknown rate_model {config.rate_model!r}")
                u = physics.utility(rate, fidelity)
                if not np.isfinite(u):
                    continue
                out.append(
                    Candidate(
                        pair=pair,
                        path=path,
                        width=width,
                        utility=weight * u,
                        rate=rate,
                        fidelity=fidelity,
                        p_min=p_min,
                        tau_e2e_s=tau,
                        wp_min=physics.rate_approximation_ratio(width, p_min),
                    )
                )

    return out


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


@dataclass
class PlacementModel:
    problem: MilpProblem
    candidates: list[Candidate]
    repeater_nodes: list[str]
    pairs: list[tuple[str, str]]
    n_candidates: int


def build_model(
    topology: Topology,
    paths: dict[tuple[str, str], list[CandidatePath]],
    hardware: Hardware,
    config: NetworkConfig,
) -> PlacementModel | None:
    """Assemble the MILP.

    Returns None when the instance is infeasible before the solver is even
    called. That happens when no candidate survives pruning at all, or when
    ``require_all_pairs`` is set and some user pair has no usable path. The
    second case is what produces the coherence cliff: below a certain T_EM
    every path for some pair violates Eq. (14), so the network cannot serve
    its demand at any repeater budget.
    """
    candidates = build_candidates(
        paths, hardware, config, demand_weight=topology.demand_weight
    )
    if not candidates:
        return None

    pairs = sorted({c.pair for c in candidates})
    if config.require_all_pairs and len(pairs) < len(paths):
        return None

    pair_index = {p: i for i, p in enumerate(pairs)}

    repeater_nodes = sorted(topology.sites)
    if config.cities_can_host_repeaters:
        interior_cities = {
            node
            for c in candidates
            for node in c.path.repeaters
            if node in set(topology.cities)
        }
        repeater_nodes = sorted(set(repeater_nodes) | interior_cities)
    node_index = {n: i for i, n in enumerate(repeater_nodes)}

    end_nodes = sorted({n for p in pairs for n in p})
    end_index = {n: i for i, n in enumerate(end_nodes)}

    n_y = len(candidates)
    n_r = len(repeater_nodes)
    n_vars = n_y + n_r

    c = np.zeros(n_vars)
    for j, cand in enumerate(candidates):
        c[j] = cand.utility

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    row = 0

    # Eq. (10): one path per user pair.
    for pair in pairs:
        for j, cand in enumerate(candidates):
            if cand.pair == pair:
                rows.append(row)
                cols.append(j)
                vals.append(1.0)
        lb.append(1.0 if config.require_all_pairs else 0.0)
        ub.append(float(config.paths_per_pair))
        row += 1

    # Eq. (9): repeater memory, and the coupling that forces r[u] = 1.
    for node, u_idx in node_index.items():
        touched = False
        for j, cand in enumerate(candidates):
            if node in cand.path.repeaters:
                rows.append(row)
                cols.append(j)
                vals.append(float(cand.width))
                touched = True
        if touched:
            rows.append(row)
            cols.append(n_y + u_idx)
            vals.append(-float(config.repeater_memories))
            lb.append(-np.inf)
            ub.append(0.0)
            row += 1
        else:
            # No candidate uses this node; drop the empty row.
            pass

    # Eq. (11): end-node memory.
    for node, _ in end_index.items():
        entries = [
            (j, float(cand.width))
            for j, cand in enumerate(candidates)
            if node in (cand.pair[0], cand.pair[1])
        ]
        if not entries:
            continue
        for j, w in entries:
            rows.append(row)
            cols.append(j)
            vals.append(w)
        lb.append(-np.inf)
        ub.append(float(config.endnode_memories))
        row += 1

    # Eq. (12): repeater budget.
    if config.max_repeaters < n_r:
        for u_idx in range(n_r):
            rows.append(row)
            cols.append(n_y + u_idx)
            vals.append(1.0)
        lb.append(-np.inf)
        ub.append(float(config.max_repeaters))
        row += 1

    A = sp.csr_matrix((vals, (rows, cols)), shape=(row, n_vars))

    problem = MilpProblem(
        c=c,
        A=A,
        lb=np.array(lb),
        ub=np.array(ub),
        n_vars=n_vars,
    )
    return PlacementModel(
        problem=problem,
        candidates=candidates,
        repeater_nodes=repeater_nodes,
        pairs=pairs,
        n_candidates=n_y,
    )


# --------------------------------------------------------------------------
# Solving and result extraction
# --------------------------------------------------------------------------


@dataclass
class PlacementResult:
    """What one solve produced."""

    status: str
    utility: float
    n_repeaters: int
    repeaters: list[str]
    served_pairs: int
    total_pairs: int
    selections: list[dict] = field(default_factory=list)
    backend: str = ""
    min_wp_min: float = float("nan")
    n_model_vars: int = 0

    def summary(self) -> str:
        if self.status != "optimal":
            return f"{self.status}: utility set to sentinel {self.utility:g}"
        return (
            f"utility {self.utility:.3f} | {self.n_repeaters} repeaters | "
            f"{self.served_pairs}/{self.total_pairs} pairs served"
        )


def solve_placement(
    topology: Topology,
    paths: dict[tuple[str, str], list[CandidatePath]],
    hardware: Hardware,
    config: NetworkConfig,
    backend: str = "highs",
    time_limit_s: float | None = None,
    mip_gap: float = 1e-4,
) -> PlacementResult:
    """Build and solve one instance. This is the unit the sweep repeats."""
    total_pairs = len(paths)
    model = build_model(topology, paths, hardware, config)
    if model is None:
        return PlacementResult(
            status="infeasible",
            utility=INFEASIBLE_UTILITY,
            n_repeaters=0,
            repeaters=[],
            served_pairs=0,
            total_pairs=total_pairs,
            backend=backend,
        )

    result: SolveResult = solve(
        model.problem, backend=backend, time_limit_s=time_limit_s, mip_gap=mip_gap
    )
    if not result.feasible or result.x is None:
        return PlacementResult(
            status=result.status,
            utility=INFEASIBLE_UTILITY,
            n_repeaters=0,
            repeaters=[],
            served_pairs=0,
            total_pairs=total_pairs,
            backend=result.backend,
            n_model_vars=model.problem.n_vars,
        )

    x = result.x
    n_y = model.n_candidates
    chosen = [j for j in range(n_y) if x[j] > 0.5]

    # Read the placement off the selected paths, not off the r variables.
    #
    # Constraint (9) forces r_u to 1 whenever a selected path passes through
    # u, but nothing forces r_u to 0 when no path uses it. When the budget
    # constraint (12) does not bind there is no pressure either way, so the
    # solver is free to leave unused r variables at 1. Those are not
    # repeaters anyone would build. The set that matters is the union of the
    # interior nodes of the paths actually chosen.
    repeaters = sorted(
        {node for j in chosen for node in model.candidates[j].path.repeaters}
    )

    selections = []
    wp_values = []
    for j in chosen:
        cand = model.candidates[j]
        wp_values.append(cand.wp_min)
        selections.append(
            {
                "pair": f"{cand.pair[0]}-{cand.pair[1]}",
                "hops": cand.path.hops,
                "width": cand.width,
                "utility": cand.utility,
                "rate_hz": cand.rate,
                "fidelity": cand.fidelity,
                "max_link_km": cand.path.max_link_km,
                "total_km": cand.path.total_km,
                "repeaters": list(cand.path.repeaters),
                "w_pmin": cand.wp_min,
            }
        )

    return PlacementResult(
        status="optimal",
        utility=float(result.objective),
        n_repeaters=len(repeaters),
        repeaters=repeaters,
        served_pairs=len(chosen),
        total_pairs=total_pairs,
        selections=selections,
        backend=result.backend,
        min_wp_min=min(wp_values) if wp_values else float("nan"),
        n_model_vars=model.problem.n_vars,
    )
