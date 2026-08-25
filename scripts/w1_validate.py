"""Week 1, Tuesday: reproduce the source paper's published results.

Nothing in this project is trustworthy until the model reproduces results
that someone else published. This script runs the paper's own settings, on
the paper's own test topologies, and checks three things:

  1. Below roughly 40 km of backbone the optimiser places no repeater.
     Paper, Section IV: "no repeaters are used and there will be a direct
     link between the end nodes".

  2. There is an end-node coherence time below which no feasible solution
     exists. The paper reports 3.2 ms for SURFnet, whose user pairs sit at
     200 to 250 km.

  3. Two independent solvers return the same objective value, so the answer
     comes from the model rather than from the solver.

Run:  python scripts/w1_validate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import banner, save_frame, step
from qrp import figures, physics
from qrp.hardware import POURYOUSEF_BASELINE
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from qrp.solver import available_backends
from qrp.topology import build_dumbbell, build_line

# The paper's own defaults, Section IV.
CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    paths_per_pair=1,
    require_all_pairs=True,
    use_demand_weights=False,
)


def backbone_crossover_km(
    hardware,
    n_candidates: int = 10,
) -> float:
    """Closed-form one-hop / two-hop crossover for the dumbbell.

    A second hop multiplies the rate by q_s and costs fidelity, and buys a
    shorter worst link. With ``n_candidates`` equally spaced sites the best
    two-hop split leaves a worst link of ceil(n/2)/n of the backbone, where
    n = n_candidates + 1, so the crossover solves

        log2(q_s) + log2(F_2 - 1/2) - log2(F_1 - 1/2)
            + (alpha/10)(1 - worst) L log2(10) = 0

    Everything here is read off ``hardware``, so the same expression covers
    the C band baseline and the O band T centre. Returns infinity when a
    single swap already drops the path to the classical floor, which is the
    honest answer for measured T centre fidelity: no repeater ever helps.
    """
    f1 = physics.e2e_fidelity(hardware.link_fidelity, 1) - 0.5
    f2 = physics.e2e_fidelity(hardware.link_fidelity, 2) - 0.5
    if f2 <= 0.0:
        return float("inf")
    n_segments = n_candidates + 1
    worst = -(-n_segments // 2) / n_segments
    gain_per_km = (hardware.alpha_db_per_km / 10.0) * (1.0 - worst) * np.log2(10.0)
    return float(-(np.log2(hardware.swap_success) + np.log2(f2) - np.log2(f1)) / gain_per_km)


def check_backbone_threshold(
    hardware=POURYOUSEF_BASELINE,
    label: str = "paper baseline",
    step_km: float = 0.5,
    check_paper_40: bool = True,
) -> pd.DataFrame:
    """Sweep the dumbbell backbone length and see when repeaters appear.

    The scan step is 0.5 km, not the 2.5 km used until 2026-08-24. At
    2.5 km from a 5.0 km start the samples are 37.5 then 40.0, which made
    the reported threshold land on exactly 40.0 km and look like an exact
    reproduction of the paper's "roughly 40 km". It was a grid artifact.
    The crossover is 38.3 km in closed form and 38.5 km at this resolution.
    """
    step(f"Check 1: the backbone length below which no repeater is placed [{label}]")

    rows = []
    for backbone in np.arange(step_km, 205.0, step_km):
        topology = build_dumbbell(float(backbone), n_candidates=10)
        pairs = topology.demand_pairs()
        paths = enumerate_paths(topology, pairs, max_link_km=backbone + 1.0, max_hops=11)
        result = solve_placement(topology, paths, hardware, CONFIG)
        hops = result.selections[0]["hops"] if result.selections else 0
        rows.append(
            {
                "backbone_km": float(backbone),
                "n_repeaters": result.n_repeaters,
                "hops": hops,
                "utility": result.utility,
                "status": result.status,
            }
        )

    frame = pd.DataFrame(rows)
    positive = frame[frame["n_repeaters"] > 0]
    threshold = positive["backbone_km"].min() if not positive.empty else float("nan")

    if threshold == threshold:  # not NaN
        print(f"   first repeater appears at {threshold:.1f} km")
    else:
        print("   no repeater is placed at any length in the scan")
    if check_paper_40:
        print("   paper reports the transition at roughly 40 km")

    analytic = backbone_crossover_km(hardware)
    print(f"   analytic crossover for this discretisation: {analytic:.1f} km")

    if analytic == float("inf"):
        print("   NOTE  one swap already drops this path to the classical floor,")
        print("         so no repeater is ever worth placing at any distance.")
    elif abs(threshold - analytic) <= max(5.0, 2.0 * step_km):
        print("   PASS  model matches its own analytic crossover")
    else:
        print("   FAIL  model disagrees with the analytic crossover")
    if check_paper_40:
        if 25.0 <= threshold <= 55.0:
            print("   PASS  transition is in the paper's neighbourhood of 40 km")
        else:
            print("   FAIL  transition is far from the paper's 40 km")

    return frame


def check_coherence_cliff() -> pd.DataFrame:
    """Find the end-node coherence time below which nothing is feasible."""
    step("Check 2: the end-node coherence cliff")

    rows = []
    for total_km in (200.0, 250.0, 320.0, 400.0):
        topology = build_line(total_km, n_candidates=7)
        pairs = topology.demand_pairs()
        paths = enumerate_paths(topology, pairs, max_link_km=total_km + 1.0, max_hops=8)

        cliff = float("nan")
        for t_em in np.geomspace(0.2e-3, 40e-3, 240):
            hardware = POURYOUSEF_BASELINE.with_(
                t_endnode_memory_s=float(t_em), t_repeater_memory_s=float(t_em)
            )
            result = solve_placement(topology, paths, hardware, CONFIG)
            if result.status == "optimal":
                cliff = float(t_em)
                break

        predicted = 2.0 * total_km / physics.C_FIBRE_KM_PER_S
        rows.append(
            {
                "total_km": total_km,
                "cliff_s": cliff,
                "predicted_s": predicted,
                "ratio": cliff / predicted if predicted else float("nan"),
            }
        )
        print(
            f"   {total_km:5.0f} km path: cliff at {cliff * 1e3:6.3f} ms, "
            f"2L/c = {predicted * 1e3:6.3f} ms"
        )

    frame = pd.DataFrame(rows)
    print()
    print("   This shows the cliff obeys 2L/c on a straight line of fibre. It")
    print("   is an internal consistency check of the timing model, not a")
    print("   reproduction of anything published: picking a length that gives")
    print("   the paper's 3.2 ms would be circular.")
    print("   The real comparison, on the authors' own SURFnet topology with")
    print("   their measured fibre lengths, is scripts/w1_surfnet_cliff.py.")

    ratios = frame["ratio"].dropna()
    if not ratios.empty and np.allclose(ratios, 1.0, atol=0.06):
        print("   PASS  cliff tracks 2L/c across every path length tested")
    else:
        print("   FAIL  cliff does not track 2L/c")
    return frame


def check_solver_agreement() -> pd.DataFrame:
    """Same instances, two solvers, same objective."""
    step("Check 3: cross-solver agreement")

    backends = available_backends()
    print(f"   backends available: {backends}")
    if len(backends) < 2:
        print("   only one backend present, so this check is skipped.")
        print("   install gurobipy to run it (the free pip licence is large")
        print("   enough for this model).")
        return pd.DataFrame()

    print("   cplex is the solver the source paper used. The pip package is")
    print("   the Community Edition, free and unregistered, capped at 1000")
    print("   variables, which covers these validation instances.")

    rows = []
    for backbone in (60.0, 120.0, 240.0, 480.0):
        topology = build_dumbbell(backbone, n_candidates=10)
        pairs = topology.demand_pairs()
        paths = enumerate_paths(topology, pairs, max_link_km=backbone + 1.0, max_hops=11)
        values = {}
        for backend in backends:
            result = solve_placement(
                topology, paths, POURYOUSEF_BASELINE, CONFIG, backend=backend, mip_gap=0.0
            )
            if result.status == "optimal":
                values[backend] = result.utility
        spread = max(values.values()) - min(values.values()) if len(values) > 1 else 0.0
        rows.append({"backbone_km": backbone, **values, "abs_diff": spread})
        print(f"   {backbone:5.0f} km: " + ", ".join(f"{k}={v:.6f}" for k, v in values.items()))

    frame = pd.DataFrame(rows)
    if (frame["abs_diff"] < 1e-6).all():
        print(f"   PASS  all {len(backends)} solvers agree to 1e-6")
    else:
        print("   FAIL  solvers disagree")
    return frame


def check_rate_regime() -> None:
    """Report how far the runs sit from the W * p_min >> 1 regime."""
    step("Check 4: validity of the rate approximation")

    topology = build_dumbbell(240.0, n_candidates=10)
    pairs = topology.demand_pairs()
    paths = enumerate_paths(topology, pairs, max_link_km=241.0, max_hops=11)
    result = solve_placement(topology, paths, POURYOUSEF_BASELINE, CONFIG)

    print(f"   240 km dumbbell, min W * p_min = {result.min_wp_min:.3g}")
    print("   Eq. (2) is stated as valid where W * p_min >> 1. The paper")
    print("   inherits this and so do we. Every sweep record carries the")
    print("   quantity so that results outside the regime can be flagged")
    print("   rather than quietly reported.")
    if result.min_wp_min < 1.0:
        print("   NOTE  this instance sits below 1, so its rate is optimistic.")


def main() -> None:
    banner("WEEK 1 VALIDATION: reproducing Pouryousef et al., IEEE TQE 2024")
    print("Hardware: " + ", ".join(f"{k}={v}" for k, v in POURYOUSEF_BASELINE.as_dict().items()))

    backbone = check_backbone_threshold()
    save_frame(backbone, "w1_backbone_threshold.csv")
    figures.plot_validation(backbone, out_dir=None)

    cliff = check_coherence_cliff()
    save_frame(cliff, "w1_coherence_cliff.csv")

    agreement = check_solver_agreement()
    if not agreement.empty:
        save_frame(agreement, "w1_solver_agreement.csv")

    check_rate_regime()

    banner("VALIDATION COMPLETE")
    print("Figures written to figures/. Tables written to results/.")


if __name__ == "__main__":
    main()
