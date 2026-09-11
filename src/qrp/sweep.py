"""Running the placement model over many hardware points.

The expensive object is the candidate path set, and it does not depend on
hardware at all: paths are geometry. So it is built once and reused across
every point of the sweep, which is what makes several thousand solves cheap.

Work is split into one chunk per worker rather than one task per point, so
the topology and path set are pickled once per worker instead of once per
solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .hardware import (
    LOG_SCALED,
    SWEEP_BOUNDS,
    SWEEP_ORDER,
    Hardware,
    hardware_from_sweep,
)
from .model import NetworkConfig, solve_placement
from .paths import CandidatePath, enumerate_paths
from .topology import Topology, build_ca9


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


@dataclass
class SweepContext:
    """Everything held fixed while hardware varies."""

    topology: Topology
    paths: dict[tuple[str, str], list[CandidatePath]]
    config: NetworkConfig
    base_hardware: Hardware
    backend: str = "highs"
    mip_gap: float = 1e-4
    time_limit_s: float | None = 60.0
    path_strategy: str = "candidates"

    @classmethod
    def build(
        cls,
        spacing_km: float = 80.0,
        max_link_km: float = 300.0,
        max_hops: int = 20,
        config: NetworkConfig | None = None,
        base_hardware: Hardware | None = None,
        bridge: bool = False,
        path_strategy: str = "candidates",
        **kwargs,
    ) -> "SweepContext":
        from .hardware import TCENTRE_MIDRANGE

        topology = build_ca9(spacing_km=spacing_km, bridge=bridge)
        pairs = topology.demand_pairs()
        paths = enumerate_paths(topology, pairs, max_link_km=max_link_km, max_hops=max_hops,
                                strategy=path_strategy)
        return cls(
            topology=topology,
            paths=paths,
            config=config or NetworkConfig(),
            base_hardware=base_hardware or TCENTRE_MIDRANGE,
            path_strategy=path_strategy,
            **kwargs,
        )


# --------------------------------------------------------------------------
# One evaluation
# --------------------------------------------------------------------------


def evaluate(context: SweepContext, values: dict[str, float]) -> dict:
    """Solve one hardware point and flatten the result into a record."""
    hardware = hardware_from_sweep(values, base=context.base_hardware)
    started = time.perf_counter()
    result = solve_placement(
        context.topology,
        context.paths,
        hardware,
        context.config,
        backend=context.backend,
        time_limit_s=context.time_limit_s,
        mip_gap=context.mip_gap,
    )
    elapsed = time.perf_counter() - started

    record = dict(values)
    record.update(
        status=result.status,
        utility=result.utility,
        n_repeaters=result.n_repeaters,
        served_pairs=result.served_pairs,
        total_pairs=result.total_pairs,
        all_pairs_served=int(result.served_pairs == result.total_pairs),
        min_w_pmin=result.min_wp_min,
        n_model_vars=result.n_model_vars,
        solve_seconds=elapsed,
        repeaters=";".join(result.repeaters),
        mip_gap=result.mip_gap,
        backend=result.backend,
        rate_model=context.config.rate_model,
        path_strategy=context.path_strategy,
    )
    if result.selections:
        record["mean_hops"] = float(np.mean([s["hops"] for s in result.selections]))
        record["mean_fidelity"] = float(np.mean([s["fidelity"] for s in result.selections]))
    else:
        record["mean_hops"] = float("nan")
        record["mean_fidelity"] = float("nan")
    return record


def _evaluate_chunk(context: SweepContext, chunk: list[dict]) -> list[dict]:
    return [evaluate(context, values) for values in chunk]


# --------------------------------------------------------------------------
# Parallel driver
# --------------------------------------------------------------------------


def run_points(
    context: SweepContext,
    points: list[dict],
    n_jobs: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate a list of hardware points, in parallel where worthwhile."""
    if not points:
        return pd.DataFrame()

    if n_jobs == 1 or len(points) < 8:
        records = [evaluate(context, values) for values in points]
        return pd.DataFrame.from_records(records)

    from joblib import Parallel, delayed

    n_chunks = min(n_jobs * 4, len(points))
    chunks = [list(c) for c in np.array_split(np.array(points, dtype=object), n_chunks)]
    chunks = [c for c in chunks if c]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(_evaluate_chunk)(context, chunk) for chunk in chunks
    )
    records = [record for batch in results for record in batch]
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Point generators
# --------------------------------------------------------------------------


def _axis(name: str, n: int) -> np.ndarray:
    low, high = SWEEP_BOUNDS[name]
    if name in LOG_SCALED:
        return np.geomspace(low, high, n)
    return np.linspace(low, high, n)


def grid_points(
    x_name: str,
    y_name: str,
    n_x: int = 24,
    n_y: int = 24,
    fixed: dict[str, float] | None = None,
) -> list[dict]:
    """Two-dimensional grid, with the other parameters held at midpoints.

    This is the phase diagram tier of the sampling design.
    """
    for name in (x_name, y_name):
        if name not in SWEEP_BOUNDS:
            raise KeyError(f"unknown sweep parameter {name!r}")

    held = dict(fixed or {})
    for name in SWEEP_ORDER:
        if name in (x_name, y_name) or name in held:
            continue
        low, high = SWEEP_BOUNDS[name]
        held[name] = float(np.sqrt(low * high)) if name in LOG_SCALED else (low + high) / 2

    points = []
    for x, y in product(_axis(x_name, n_x), _axis(y_name, n_y)):
        values = dict(held)
        values[x_name] = float(x)
        values[y_name] = float(y)
        points.append(values)
    return points


def line_points(name: str, n: int = 40, fixed: dict[str, float] | None = None) -> list[dict]:
    """One-dimensional scan, used for the coherence cliff and threshold checks."""
    held = dict(fixed or {})
    for other in SWEEP_ORDER:
        if other == name or other in held:
            continue
        low, high = SWEEP_BOUNDS[other]
        held[other] = float(np.sqrt(low * high)) if other in LOG_SCALED else (low + high) / 2

    points = []
    for value in _axis(name, n):
        values = dict(held)
        values[name] = float(value)
        points.append(values)
    return points


def latin_hypercube_points(
    n: int = 500,
    seed: int = 20260803,
    names: tuple[str, ...] = SWEEP_ORDER,
) -> list[dict]:
    """Latin hypercube sample. The boundary-exploration tier.

    Spread across the whole box, used to sharpen the regime borders that the
    grid only samples coarsely.
    """
    from SALib.sample import latin

    problem = sweep_problem(names)
    raw = latin.sample(problem, n, seed=seed)
    return _decode(raw, names)


def sweep_problem(names: tuple[str, ...] = SWEEP_ORDER) -> dict:
    """SALib problem definition.

    Log-scaled parameters are sampled in log10 space and converted back, so
    that a uniform sample in SALib's unit box is uniform in decades.
    """
    bounds = []
    for name in names:
        low, high = SWEEP_BOUNDS[name]
        if name in LOG_SCALED:
            bounds.append([np.log10(low), np.log10(high)])
        else:
            bounds.append([low, high])
    return {"num_vars": len(names), "names": list(names), "bounds": bounds}


def _decode(raw: np.ndarray, names: tuple[str, ...]) -> list[dict]:
    points = []
    for row in raw:
        values = {}
        for name, value in zip(names, row):
            values[name] = float(10.0**value) if name in LOG_SCALED else float(value)
        points.append(values)
    return points
