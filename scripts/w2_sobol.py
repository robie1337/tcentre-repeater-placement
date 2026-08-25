"""Week 2, Wednesday: the Sobol tier. Which hardware parameter moves the result.

Variance-based global sensitivity analysis over the four swept parameters.
Cost is N * (D + 2) solves, so N = 1024 with four parameters is 6,144 solves.

This is the number a hardware programme would act on: it says whether rate,
coherence time, link fidelity or swap success buys the most network
performance per unit of engineering effort.

Run:  python scripts/w2_sobol.py [--n 1024] [--jobs 6]
      python scripts/w2_sobol.py --n 128        (quick check, ~13 min)
"""

from __future__ import annotations

import argparse
import time

from _common import banner, save_frame, step
from qrp import figures
from qrp.model import NetworkConfig
from qrp.sensitivity import analyse, saltelli_points
from qrp.sweep import SweepContext, run_points

CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=False,
    use_demand_weights=False,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024,
                        help="Saltelli base sample size, must be a power of two")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--second-order", action="store_true",
                        help="also estimate pairwise interaction indices")
    args = parser.parse_args()

    banner("WEEK 2 SOBOL: ranking the hardware parameters")

    context = SweepContext.build(
        spacing_km=80.0, max_link_km=300.0, max_hops=20, config=CONFIG
    )

    points, problem = saltelli_points(
        args.n, calc_second_order=args.second_order
    )
    print(f"parameters: {problem['names']}")
    print(f"sample: N = {args.n}, {len(points)} model solves"
          f"{' (with second order)' if args.second_order else ''}")

    step("Solving")
    started = time.time()
    frame = run_points(context, points, n_jobs=args.jobs)
    elapsed = time.time() - started
    print(f"   {len(frame)} solves in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(frame), 1):.3f}s each)")
    print(f"   feasible: {(frame['status'] == 'optimal').sum()} / {len(frame)}")
    save_frame(frame, "w2_sobol_samples.csv")

    below_regime = (frame["min_w_pmin"] < 1.0).sum()
    if below_regime:
        print(f"   NOTE {below_regime} solves have W * p_min < 1, outside the")
        print("        regime where the paper's rate approximation is stated")
        print("        to hold. Reported in the limitations section.")

    step("Sobol indices")
    rows = []
    for output in ("utility", "served_pairs", "n_repeaters"):
        result = analyse(frame, problem, output=output,
                         calc_second_order=args.second_order)
        print()
        print(result.summary())
        figures.plot_sobol(result, stem=f"fig_sobol_{output}")
        table = result.to_frame()
        table.insert(0, "output", output)
        rows.append(table)

    import pandas as pd

    save_frame(pd.concat(rows, ignore_index=True), "w2_sobol_indices.csv")
    print("\n   wrote figures/fig_sobol_*.pdf")

    banner("SOBOL COMPLETE")


if __name__ == "__main__":
    main()
