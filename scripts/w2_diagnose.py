"""Explain the two things the sweep turned up that need a mechanism.

First: no *sampled* point serves all eighteen user pairs, yet the best corner
of the box serves all eighteen. That is a statement about the sample, not
about the hardware, and the difference matters. This finds how good each
parameter has to be, with the others at their best, before the whole network
can be served.

Second: every solve sits below the W * p_min >> 1 regime in which the paper
states its rate equation is valid. This quantifies how far below, and what
memory count would be needed to get back inside it.

Run:  python scripts/w2_diagnose.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import RESULTS, banner, save_frame, step
from qrp import physics, topology
from qrp.hardware import LABELS, LOG_SCALED, SWEEP_BOUNDS, SWEEP_ORDER, TCENTRE_PROJECTED
from qrp.model import NetworkConfig, build_candidates, solve_placement
from qrp.paths import enumerate_paths

CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=False,
)

BEST_CORNER = {name: SWEEP_BOUNDS[name][1] for name in SWEEP_ORDER}


def hardware_at(values):
    from qrp.hardware import hardware_from_sweep

    return hardware_from_sweep(values, base=TCENTRE_PROJECTED)


def main() -> None:
    banner("DIAGNOSING THE SWEEP")

    topo = topology.build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)

    # ---------------------------------------------------------------- part 1
    step("The best corner of the sweep box")
    result = solve_placement(topo, paths, hardware_at(BEST_CORNER), CONFIG)
    print(f"   {result.summary()}")
    print("   So the ceiling seen in the sweep is a property of the sample,")
    print("   not of the hardware. Saltelli and Latin hypercube samples put")
    print("   very few points near a corner of a four-dimensional box, and")
    print("   the two-dimensional grids hold the other two parameters at")
    print("   their midpoints by construction.")

    step("How good does each parameter have to be, with the others at their best")
    rows = []
    for name in SWEEP_ORDER:
        low, high = SWEEP_BOUNDS[name]
        axis = np.geomspace(low, high, 60) if name in LOG_SCALED else np.linspace(low, high, 60)
        threshold = None
        for value in axis:
            values = dict(BEST_CORNER)
            values[name] = float(value)
            r = solve_placement(topo, paths, hardware_at(values), CONFIG)
            if r.served_pairs == r.total_pairs:
                threshold = float(value)
                break
        rows.append(
            {
                "parameter": LABELS[name],
                "key": name,
                "box_low": low,
                "box_high": high,
                "needed_for_all_18": threshold,
                "fraction_of_range": (
                    None if threshold is None
                    else (np.log10(threshold / low) / np.log10(high / low)
                          if name in LOG_SCALED
                          else (threshold - low) / (high - low))
                ),
            }
        )
        shown = f"{threshold:.4g}" if threshold is not None else "not reachable in box"
        print(f"   {LABELS[name]:<26} needs at least {shown}")

    save_frame(pd.DataFrame(rows), "w2_requirements.csv")
    print()
    print("   Read this as a hardware requirement: with everything else at the")
    print("   top of its published range, this is the level each parameter has")
    print("   to reach before the nine city network can serve all of its")
    print("   traffic.")

    # ---------------------------------------------------------------- part 2
    step("How far outside the rate approximation is this network")

    result = solve_placement(topo, paths, hardware_at(BEST_CORNER), CONFIG)
    selections = pd.DataFrame(result.selections)
    if not selections.empty:
        p_min = selections["w_pmin"] / selections["width"]
        needed = 1.0 / p_min
        print(f"   worst-link success probability on the chosen paths:")
        print(f"     median {p_min.median():.3g}, worst {p_min.min():.3g}")
        print(f"   memories per node needed for W * p_min = 1:")
        print(f"     median {needed.median():,.0f}, worst {needed.max():,.0f}")
        print(f"   memories per node actually modelled: {CONFIG.endnode_memories}")

    path = RESULTS / "w2_sobol_samples.csv"
    if path.exists():
        frame = pd.read_csv(path)
        valid = frame["min_w_pmin"].dropna()
        below = valid[valid < 1.0]
        print()
        print(f"   across the sweep: {len(below)} of {len(valid)} solves have W * p_min < 1")
        if len(below):
            print(f"     median {below.median():.3g}, worst {below.min():.3g}")

    step("What that means, stated plainly")
    print("   Eq. (2) of the paper is R = q_s^(h-1) W p_min, and the paper says")
    print("   it holds where W p_min >> 1. On this network it never does.")
    print()
    print("   The reason is the O band. At 0.35 dB/km an 80 km link transmits")
    print(f"   {physics.link_success(80.0, 0.35):.2e} of the photons sent, so reaching W p_min = 1")
    print(f"   needs about {1 / physics.link_success(80.0, 0.35):,.0f} memories per node. The paper's own")
    print("   experiments use 100. At the C band value of 0.2 dB/km the same")
    print(f"   link transmits {physics.link_success(80.0, 0.2):.2e}, needing about {1 / physics.link_success(80.0, 0.2):,.0f}, which is")
    print("   why the approximation is reasonable in their setting and not in")
    print("   ours.")
    print()
    print("   This is a finding, not a bug. Moving a placement model from the")
    print("   C band to the O band breaks an approximation the published model")
    print("   relies on, and the fix is either far more multiplexing or a rate")
    print("   equation that stays valid at low p. It belongs in the report as")
    print("   a limitation and in the next stage as a question.")

    banner("DIAGNOSIS COMPLETE")


if __name__ == "__main__":
    main()
