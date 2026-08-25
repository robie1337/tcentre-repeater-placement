"""Do the two model fixes change the conclusions?

Four experiments, one per question.

1. RATE BRACKET. Solve the network under the paper's rate model and the
   conservative coordinated model. The truth lies between them; if the
   qualitative story agrees at both ends, it is robust to the rate-model
   uncertainty.

2. SENSITIVITY UNDER A VALID FORMULA. Re-run the four-parameter Sobol
   decomposition under the coordinated model, which is built for the low
   W*p regime. If coherence still dominates, the ranking was not an
   artifact of running Eq. (2) outside its stated domain.

3. SWAP NOISE AS A FIFTH PARAMETER. Sweep the per-swap Werner factor
   across [measured-today, perfect] alongside the original four, instead of
   holding it at the paper's perfect-operations value. This answers whether
   the unresolved gate-fidelity question can overturn the ranking.

4. BUYING OUT OF THE INVALID REGIME. Raise the memory ceiling from 100 to
   2048 and let the optimiser choose widths. Finding 4 says the O-band
   needs ~631 memories per node for Eq. (2)'s condition; this asks whether
   the model, allowed to buy memory, actually does.

Run:  python scripts/w3_fixes.py [--jobs 6] [--n 1024] [--n5 512]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import banner, save_frame, step
from qrp import figures
from qrp.hardware import (
    SWEEP_ORDER,
    SWEEP_ORDER_5,
    TCENTRE_MIDRANGE,
    TCENTRE_PROJECTED,
)
from qrp.model import NetworkConfig, log_width_grid, solve_placement
from qrp.paths import enumerate_paths
from qrp.sensitivity import analyse, saltelli_points
from qrp.sweep import SweepContext, grid_points, run_points
from qrp.topology import build_ca9


def open_config(rate_model: str, **overrides) -> NetworkConfig:
    defaults = dict(
        repeater_memories=100,
        endnode_memories=100,
        max_repeaters=10**6,
        require_all_pairs=False,
        use_demand_weights=False,
        rate_model=rate_model,
    )
    defaults.update(overrides)
    return NetworkConfig(**defaults)


def part1_bracket(topo, paths) -> None:
    step("1. The rate bracket on the full network")
    rows = []
    for hw_name, hardware in [("midrange", TCENTRE_MIDRANGE), ("projected", TCENTRE_PROJECTED)]:
        for model_name in ("paper", "coordinated"):
            result = solve_placement(topo, paths, hardware, open_config(model_name))
            rows.append(
                {
                    "hardware": hw_name,
                    "rate_model": model_name,
                    "status": result.status,
                    "utility": round(result.utility, 3),
                    "served_pairs": result.served_pairs,
                    "n_repeaters": result.n_repeaters,
                }
            )
            print(f"   {hw_name:<10} {model_name:<12} {result.summary()}")
    frame = pd.DataFrame(rows)
    save_frame(frame, "w3_rate_bracket.csv")

    print()
    print("   Reading: 'paper' is the optimistic pipelined end of the bracket,")
    print("   'coordinated' the conservative end. Compare served_pairs and the")
    print("   placement story across the two, not the absolute utilities —")
    print("   the utility scales differ because the models count differently.")


def part2_sobol_valid(context_coord, args) -> None:
    step("2. Four-parameter Sobol under the coordinated (valid at low W*p) model")
    points, problem = saltelli_points(args.n, names=SWEEP_ORDER)
    started = time.time()
    frame = run_points(context_coord, points, n_jobs=args.jobs)
    print(f"   {len(frame)} solves in {(time.time() - started) / 60:.1f} min")
    save_frame(frame, "w3_sobol_coordinated.csv")

    for output in ("served_pairs", "utility"):
        result = analyse(frame, problem, output=output)
        print()
        print(result.summary())
        figures.plot_sobol(result, stem=f"fig_sobol_coordinated_{output}")

    print()
    print("   Compare with results/w2_sobol_indices.csv (paper model). If the")
    print("   ordering matches, the ranking is not an artifact of Eq. (2).")


def part2b_phase(context_coord, args) -> None:
    step("2b. Phase diagram (t2 x rate) under the coordinated model")
    points = grid_points("t2_s", "generation_rate_hz", 24, 24)
    frame = run_points(context_coord, points, n_jobs=args.jobs, verbose=False)
    save_frame(frame, "w3_grid_t2_rate_coordinated.csv")
    figures.plot_phase_diagram(
        frame, "t2_s", "generation_rate_hz", value="served_pairs",
        stem="fig_phase_t2_rate_served_coordinated",
        title="Pairs served, coordinated rate model",
    )
    print("   wrote figures/fig_phase_t2_rate_served_coordinated.pdf")


def part3_swap_noise(context_coord, args) -> None:
    step("3. Five-parameter Sobol including the per-swap noise factor")
    print("   bracket [0.848, 1.0]: measured-today worst case to the paper's")
    print("   perfect-operations assumption")
    points, problem = saltelli_points(args.n5, names=SWEEP_ORDER_5)
    started = time.time()
    frame = run_points(context_coord, points, n_jobs=args.jobs)
    print(f"   {len(frame)} solves in {(time.time() - started) / 60:.1f} min")
    save_frame(frame, "w3_sobol_5param.csv")

    tables = []
    for output in ("served_pairs", "utility"):
        result = analyse(frame, problem, output=output)
        print()
        print(result.summary())
        figures.plot_sobol(result, stem=f"fig_sobol_5param_{output}")
        table = result.to_frame()
        table.insert(0, "output", output)
        tables.append(table)
    save_frame(pd.concat(tables, ignore_index=True), "w3_sobol_5param_indices.csv")


def part4_buy_memory(topo, paths) -> None:
    step("4. Raising the memory ceiling to 2048: does the model buy its way out?")
    wide = log_width_grid(2048, n_points=12)
    print(f"   width grid: {wide}")

    rows = []
    for model_name in ("paper", "coordinated"):
        config = open_config(
            model_name,
            repeater_memories=2048,
            endnode_memories=2048,
            width_grid=wide,
        )
        result = solve_placement(topo, paths, TCENTRE_PROJECTED, config)
        chosen = sorted({s["width"] for s in result.selections})
        over_631 = sum(1 for s in result.selections if s["width"] >= 631)
        print(
            f"   {model_name:<12} {result.summary()}\n"
            f"                widths chosen: {chosen}; "
            f"{over_631}/{len(result.selections)} paths at W >= 631"
        )
        for s in result.selections:
            rows.append(
                {
                    "rate_model": model_name,
                    "pair": s["pair"],
                    "hops": s["hops"],
                    "width": s["width"],
                    "w_pmin": s["w_pmin"],
                }
            )
    frame = pd.DataFrame(rows)
    save_frame(frame, "w3_memory_ceiling.csv")

    print()
    print("   Under the paper model W*p_min >= 1 becomes reachable at high W;")
    print("   whether the optimiser pays for it is the memory-pricing result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--n", type=int, default=1024, help="4-param Saltelli base")
    parser.add_argument("--n5", type=int, default=512, help="5-param Saltelli base")
    args = parser.parse_args()

    banner("W3: DO THE MODEL FIXES CHANGE THE CONCLUSIONS?")

    topo = build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)

    part1_bracket(topo, paths)

    context_coord = SweepContext.build(
        spacing_km=80.0, max_link_km=300.0, max_hops=20,
        config=open_config("coordinated"),
    )
    part2_sobol_valid(context_coord, args)
    part2b_phase(context_coord, args)
    part3_swap_noise(context_coord, args)
    part4_buy_memory(topo, paths)

    banner("W3 COMPLETE")


if __name__ == "__main__":
    main()
