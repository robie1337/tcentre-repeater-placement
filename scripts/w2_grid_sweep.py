"""Week 2, Tuesday: the grid tier. This produces the phase diagram.

Sweeps generation rate against link fidelity at fixed coherence and swap
success, and draws the result as a heat map with the boundary where the
network stops serving all eighteen user pairs.

Also runs one-dimensional scans over each parameter, which is what the report
uses to describe where the boundaries are and why.

Run:  python scripts/w2_grid_sweep.py [--n 24] [--jobs 6]
"""

from __future__ import annotations

import argparse
import time

from _common import banner, save_frame, step
from qrp import figures
from qrp.hardware import LABELS, SWEEP_ORDER
from qrp.model import NetworkConfig
from qrp.sweep import SweepContext, grid_points, line_points, run_points

# The repeater budget is left unbinding for the hardware sweep.
#
# With a tight budget the budget is what limits the network, and the sweep
# measures the budget rather than the hardware. Every candidate site is made
# available so that what the diagram shows is the hardware boundary. The
# effect of a tight budget is a separate question and is answered by
# w1_single_case.py, which runs three budgets at fixed hardware.
CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=False,
    use_demand_weights=False,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24, help="grid points per axis")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--budget", type=int, default=10**6)
    args = parser.parse_args()

    banner("WEEK 2 GRID SWEEP: the phase diagram")

    config = NetworkConfig(
        repeater_memories=100,
        endnode_memories=100,
        max_repeaters=args.budget,
        require_all_pairs=False,
        use_demand_weights=False,
    )
    context = SweepContext.build(
        spacing_km=80.0, max_link_km=300.0, max_hops=20, config=config
    )
    print(f"repeater budget: {context.config.max_repeaters}")
    print(f"candidate paths: {sum(len(v) for v in context.paths.values())}")

    # --- the flagship two-dimensional sweep ---------------------------------
    step(f"Grid: generation rate against link fidelity ({args.n} x {args.n})")
    points = grid_points("generation_rate_hz", "link_fidelity", args.n, args.n)
    started = time.time()
    frame = run_points(context, points, n_jobs=args.jobs)
    print(f"   {len(frame)} solves in {time.time() - started:.1f}s")
    print(f"   feasible: {(frame['status'] == 'optimal').sum()} / {len(frame)}")
    print(f"   all pairs served: {int(frame['all_pairs_served'].sum())} points")
    save_frame(frame, "w2_grid_rate_fidelity.csv")

    for value in ("utility", "served_pairs", "n_repeaters"):
        figures.plot_phase_diagram(
            frame, "generation_rate_hz", "link_fidelity", value=value,
            stem=f"fig_phase_rate_fidelity_{value}",
        )
    print("   wrote figures/fig_phase_rate_fidelity_*.pdf")

    # --- a second slice: coherence against swap success ---------------------
    step(f"Grid: coherence time against swap success ({args.n} x {args.n})")
    points = grid_points("t2_s", "swap_success", args.n, args.n)
    frame2 = run_points(context, points, n_jobs=args.jobs)
    print(f"   feasible: {(frame2['status'] == 'optimal').sum()} / {len(frame2)}")
    save_frame(frame2, "w2_grid_t2_swap.csv")
    figures.plot_phase_diagram(
        frame2, "t2_s", "swap_success", value="served_pairs",
        stem="fig_phase_t2_swap_served",
    )
    figures.plot_phase_diagram(
        frame2, "t2_s", "swap_success", value="utility",
        stem="fig_phase_t2_swap_utility",
    )
    print("   wrote figures/fig_phase_t2_swap_*.pdf")

    # --- a third slice: coherence against generation rate -------------------
    # The line scans show coherence has by far the widest effect on how many
    # pairs can be served, so this is the slice where the boundary lives.
    step(f"Grid: coherence time against generation rate ({args.n} x {args.n})")
    points = grid_points("t2_s", "generation_rate_hz", args.n, args.n)
    frame3 = run_points(context, points, n_jobs=args.jobs)
    print(f"   feasible: {(frame3['status'] == 'optimal').sum()} / {len(frame3)}")
    print(f"   all pairs served: {int(frame3['all_pairs_served'].sum())} points")
    save_frame(frame3, "w2_grid_t2_rate.csv")
    for value in ("served_pairs", "utility", "n_repeaters"):
        figures.plot_phase_diagram(
            frame3, "t2_s", "generation_rate_hz", value=value,
            stem=f"fig_phase_t2_rate_{value}",
        )
    print("   wrote figures/fig_phase_t2_rate_*.pdf")

    # --- one-dimensional scans ---------------------------------------------
    for name in SWEEP_ORDER:
        step(f"Line scan over {LABELS[name]}")
        points = line_points(name, n=40)
        scan = run_points(context, points, n_jobs=args.jobs, verbose=False)
        save_frame(scan, f"w2_scan_{name}.csv")
        figures.plot_line_scan(
            scan, name,
            values=("utility", "served_pairs", "n_repeaters"),
            stem=f"fig_scan_{name}",
            title=f"Scan over {LABELS[name]}, other parameters at box midpoint",
        )
        served = scan["served_pairs"]
        print(f"   pairs served ranges {served.min()} to {served.max()}")

    banner("GRID SWEEP COMPLETE")


if __name__ == "__main__":
    main()
