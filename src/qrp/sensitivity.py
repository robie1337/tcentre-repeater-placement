"""Variance-based global sensitivity analysis.

Sobol indices answer the hardware roadmap question directly: of generation
rate, coherence time, link fidelity and swap success, which one explains most
of the variation in network utility. First-order index S_i is the share of
variance from parameter i alone; total-effect index S_Ti adds everything it
contributes through interactions with the others.

This is not one-factor-at-a-time. OFAT holds every other parameter at a
nominal value and moves one, which measures only a single line through the
parameter box and cannot see interactions at all. Sobol sampling covers the
whole box, which matters here because coherence time and hop count interact
strongly: how much a coherence improvement buys depends on how long the paths
already are.

Sample cost is N * (D + 2) model solves for D parameters with second-order
indices switched off. With D = 4 and N = 1024 that is 6,144 solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hardware import LABELS, SWEEP_ORDER
from .solver import INFEASIBLE_UTILITY
from .sweep import SweepContext, _decode, run_points, sweep_problem


@dataclass
class SobolResult:
    names: tuple[str, ...]
    first_order: np.ndarray
    first_order_conf: np.ndarray
    total_effect: np.ndarray
    total_effect_conf: np.ndarray
    n_samples: int
    output_name: str
    n_infeasible: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "parameter": [LABELS.get(n, n) for n in self.names],
                "key": self.names,
                "S1": self.first_order,
                "S1_conf": self.first_order_conf,
                "ST": self.total_effect,
                "ST_conf": self.total_effect_conf,
            }
        ).sort_values("ST", ascending=False, ignore_index=True)

    def summary(self) -> str:
        frame = self.to_frame()
        lines = [
            f"Sobol indices on {self.output_name}: "
            f"{self.n_samples} model solves, {self.n_infeasible} infeasible"
        ]
        for row in frame.itertuples():
            lines.append(
                f"  {row.parameter:<26} S1 = {row.S1: .3f} +/- {row.S1_conf:.3f}"
                f"   ST = {row.ST: .3f} +/- {row.ST_conf:.3f}"
            )
        return "\n".join(lines)


def saltelli_points(
    n_base: int = 1024,
    names: tuple[str, ...] = SWEEP_ORDER,
    calc_second_order: bool = False,
    seed: int = 20260812,
) -> tuple[list[dict], dict]:
    """Generate the Saltelli sample.

    ``n_base`` should be a power of two: the scrambled Sobol sequence has its
    convergence properties on powers of two, and SALib warns otherwise.
    """
    if n_base & (n_base - 1) != 0:
        raise ValueError(f"n_base should be a power of two, got {n_base}")

    from SALib.sample import sobol as sobol_sample

    problem = sweep_problem(names)
    raw = sobol_sample.sample(
        problem, n_base, calc_second_order=calc_second_order, seed=seed
    )
    return _decode(raw, names), problem


def analyse(
    frame: pd.DataFrame,
    problem: dict,
    output: str = "utility",
    calc_second_order: bool = False,
) -> SobolResult:
    """Decompose the variance of one output column.

    Infeasible points carry the sentinel utility rather than being dropped.
    Removing them would break the Saltelli estimator, which needs the sample
    matrix intact and in order. Treating infeasibility as a very bad outcome
    is also the honest reading: a hardware setting that cannot serve the
    network is worse than one that serves it badly.
    """
    from SALib.analyze import sobol as sobol_analyse

    values = frame[output].to_numpy(dtype=float)
    n_infeasible = int((frame["status"] != "optimal").sum())

    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=INFEASIBLE_UTILITY, neginf=INFEASIBLE_UTILITY)

    if np.allclose(values, values[0]):
        zeros = np.zeros(problem["num_vars"])
        return SobolResult(
            names=tuple(problem["names"]),
            first_order=zeros,
            first_order_conf=zeros.copy(),
            total_effect=zeros.copy(),
            total_effect_conf=zeros.copy(),
            n_samples=len(values),
            output_name=output,
            n_infeasible=n_infeasible,
        )

    indices = sobol_analyse.analyze(
        problem, values, calc_second_order=calc_second_order, print_to_console=False
    )
    return SobolResult(
        names=tuple(problem["names"]),
        first_order=np.asarray(indices["S1"]),
        first_order_conf=np.asarray(indices["S1_conf"]),
        total_effect=np.asarray(indices["ST"]),
        total_effect_conf=np.asarray(indices["ST_conf"]),
        n_samples=len(values),
        output_name=output,
        n_infeasible=n_infeasible,
    )


def run_sobol(
    context: SweepContext,
    n_base: int = 1024,
    names: tuple[str, ...] = SWEEP_ORDER,
    outputs: tuple[str, ...] = ("utility", "served_pairs", "n_repeaters"),
    n_jobs: int = 6,
    calc_second_order: bool = False,
    seed: int = 20260812,
) -> tuple[pd.DataFrame, dict[str, SobolResult]]:
    """Sample, solve, and decompose. Returns the raw frame and the indices."""
    points, problem = saltelli_points(n_base, names, calc_second_order, seed)
    frame = run_points(context, points, n_jobs=n_jobs)
    results = {
        output: analyse(frame, problem, output=output, calc_second_order=calc_second_order)
        for output in outputs
        if output in frame.columns
    }
    return frame, results
