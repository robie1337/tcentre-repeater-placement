"""Final bracket and corrected swap-quality sweep, after the literature research.

Three things, informed by the two research reports:

1. THREE-MODEL BRACKET. Eq. (2) (the optimistic ceiling), Q-CAST's EXT
   (Shi & Qian, SIGCOMM 2020 -- the literature-anchored exact slotted
   model), and the buffered coordinated model. The truth for a
   long-coherence platform sits between EXT and the ceiling.

2. SWAP QUALITY, CORRECTED BRACKET. The five-parameter Sobol re-run with
   s_q in [0.71, 0.997] -- measured-today components to projected
   cavity-assisted readout -- replacing the earlier [0.848, 1.0] which
   rested on an unverifiable 0.946 readout figure.

3. HOP BUDGETS at the bracket anchors, for the report's requirements table.
   Benchmark: da Silva et al. (QST 2024) report required swap quality
   0.996-0.997 for a 900 km chain.

Run:  python scripts/w3b_final.py [--jobs 6] [--n5 512]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import banner, save_frame, step
from qrp import figures, physics
from qrp.hardware import SWEEP_ORDER_5, TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from qrp.sensitivity import analyse, saltelli_points
from qrp.sweep import SweepContext, run_points
from qrp.topology import build_ca9


def open_config(rate_model: str) -> NetworkConfig:
    return NetworkConfig(
        repeater_memories=100,
        endnode_memories=100,
        max_repeaters=10**6,
        require_all_pairs=False,
        use_demand_weights=False,
        rate_model=rate_model,
    )


def part1_three_model_bracket(topo, paths) -> None:
    step("1. The three-model rate bracket")
    rows = []
    for hw_name, hardware in [("midrange", TCENTRE_MIDRANGE), ("projected", TCENTRE_PROJECTED)]:
        for model_name in ("paper", "ext", "coordinated"):
            result = solve_placement(topo, paths, hardware, open_config(model_name))
            rows.append(
                {
                    "hardware": hw_name,
                    "rate_model": model_name,
                    "utility": round(result.utility, 3),
                    "served_pairs": result.served_pairs,
                    "n_repeaters": result.n_repeaters,
                }
            )
            print(f"   {hw_name:<10} {model_name:<12} {result.summary()}")
    save_frame(pd.DataFrame(rows), "w3b_three_model_bracket.csv")
    print()
    print("   paper = infinite-buffer flow ceiling (Pouryousef Eq. 2 / Razavi 2009);")
    print("   ext = exact slotted throughput (Shi & Qian, SIGCOMM 2020);")
    print("   coordinated = buffered waiting-time model (Bernardes 2011 structure).")


def part2_sobol_corrected(args) -> None:
    step("2. Five-parameter Sobol, swap quality in [0.71, 0.997], EXT rate model")
    context = SweepContext.build(
        spacing_km=80.0, max_link_km=300.0, max_hops=20,
        config=open_config("ext"),
    )
    points, problem = saltelli_points(args.n5, names=SWEEP_ORDER_5)
    started = time.time()
    frame = run_points(context, points, n_jobs=args.jobs)
    print(f"   {len(frame)} solves in {(time.time() - started) / 60:.1f} min")
    save_frame(frame, "w3b_sobol_5param_ext.csv")

    tables = []
    for output in ("served_pairs", "utility"):
        result = analyse(frame, problem, output=output)
        print()
        print(result.summary())
        figures.plot_sobol(result, stem=f"fig_sobol_final_{output}")
        table = result.to_frame()
        table.insert(0, "output", output)
        tables.append(table)
    save_frame(pd.concat(tables, ignore_index=True), "w3b_sobol_final_indices.csv")


def part3_hop_budgets() -> None:
    step("3. Hop budgets at the swap-quality anchors")
    print("   (longest path whose end-to-end fidelity stays above 1/2)")
    print()
    header = f"   {'s_q':>8} | " + " | ".join(f"F_L={f}" for f in (0.92, 0.96, 0.998))
    print(header)
    print("   " + "-" * (len(header) - 3))
    rows = []
    for s_q, label in [
        (0.71, "measured today (SPAM floor)"),
        (0.848, "earlier premise (unverified)"),
        (0.997, "projected ceiling"),
        (1.0, "paper assumption"),
    ]:
        budgets = []
        for f_l in (0.92, 0.96, 0.998):
            best = 0
            for hops in range(1, 300):
                if physics.e2e_fidelity(f_l, hops, swap_werner=s_q) > 0.5:
                    best = hops
                else:
                    break
            budgets.append(best)
        rows.append({"s_q": s_q, "label": label,
                     "hops_at_0.92": budgets[0], "hops_at_0.96": budgets[1],
                     "hops_at_0.998": budgets[2]})
        print(f"   {s_q:>8} | " + " | ".join(f"{b:>6}" for b in budgets) + f"   {label}")
    save_frame(pd.DataFrame(rows), "w3b_hop_budgets.csv")
    print()
    print("   Benchmark: da Silva et al. (QST 9, 045041 (2024)) require swap")
    print("   quality 0.996-0.997 for their 900 km NV chain. The T centre's")
    print("   projected ceiling only just reaches that bar; today's measured")
    print("   components sit far below it.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--n5", type=int, default=512)
    args = parser.parse_args()

    banner("W3B: FINAL BRACKET AND CORRECTED SWAP-QUALITY SWEEP")

    topo = build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)

    part1_three_model_bracket(topo, paths)
    part2_sobol_corrected(args)
    part3_hop_budgets()

    banner("W3B COMPLETE")


if __name__ == "__main__":
    main()
