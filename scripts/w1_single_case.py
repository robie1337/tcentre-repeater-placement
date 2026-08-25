"""Week 1, Friday: one complete case on the CA9 graph.

One hardware setting in, one repeater placement out, drawn on the map. This
is the first real figure and it freezes the model interface before any
sweeping starts.

Also runs the placement at three repeater budgets, which is what the report
needs for the deployment discussion.

Run:  python scripts/w1_single_case.py
"""

from __future__ import annotations

import pandas as pd

from _common import banner, save_frame, step
from qrp import figures, topology
from qrp.hardware import TCENTRE_MEASURED, TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths, path_statistics

# On the CA9 graph we do not require every pair to be served.
#
# With the requirement on, a single unservable pair makes the whole instance
# infeasible and the result carries no information about the other
# seventeen. With it off, the number of pairs served becomes the interesting
# output and the boundary where it falls below eighteen is exactly the
# feasibility boundary the requirement would have reported. That gives a
# readable phase diagram instead of a binary one.
BUDGETS = (10, 25, 53)


def run_budget(topo, paths, hardware, budget, label):
    config = NetworkConfig(
        repeater_memories=100,
        endnode_memories=100,
        max_repeaters=budget,
        require_all_pairs=False,
        use_demand_weights=False,
    )
    result = solve_placement(topo, paths, hardware, config)
    print(
        f"   budget {budget:>3}: {result.summary():<58} "
        f"[{result.n_model_vars} vars]"
    )
    return {
        "hardware": label,
        "budget": budget,
        "status": result.status,
        "utility": result.utility,
        "n_repeaters": result.n_repeaters,
        "served_pairs": result.served_pairs,
        "total_pairs": result.total_pairs,
        "repeaters": ";".join(result.repeaters),
    }, result


def main() -> None:
    banner("WEEK 1 SINGLE CASE: T centre hardware on CA9")

    topo = topology.build_ca9(spacing_km=80.0)
    print(topology.describe(topo))

    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)
    print(path_statistics(paths))

    figures.plot_topology(topo, stem="fig_topology")
    print("   wrote figures/fig_topology.pdf")

    rows = []
    best_for_map = None

    for label, hardware in [
        ("measured", TCENTRE_MEASURED),
        ("midrange", TCENTRE_MIDRANGE),
        ("projected", TCENTRE_PROJECTED),
    ]:
        step(f"{label}: " + ", ".join(
            f"{k}={v:g}" for k, v in hardware.as_dict().items()
            if k in ("alpha_db_per_km", "link_fidelity", "swap_success",
                     "generation_rate_hz", "t_endnode_memory_s")
        ))
        for budget in BUDGETS:
            row, result = run_budget(topo, paths, hardware, budget, label)
            rows.append(row)
            if label == "projected" and budget == 25:
                best_for_map = result

    frame = pd.DataFrame(rows)
    save_frame(frame, "w1_single_case.csv")

    if best_for_map is not None and best_for_map.repeaters:
        figures.plot_topology(
            topo,
            chosen_repeaters=best_for_map.repeaters,
            stem="fig_placement_projected_b25",
            title="Optimal placement, projected T centre hardware, budget 25",
        )
        print("   wrote figures/fig_placement_projected_b25.pdf")

        step("Paths chosen at the projected setting, budget 25")
        for sel in sorted(best_for_map.selections, key=lambda s: -s["utility"])[:8]:
            print(
                f"   {sel['pair']:<24} h={sel['hops']:<3} w={sel['width']:<4} "
                f"F={sel['fidelity']:.4f}  R={sel['rate_hz']:.3g} Hz  "
                f"U={sel['utility']:.2f}"
            )

    banner("SINGLE CASE COMPLETE")


if __name__ == "__main__":
    main()
