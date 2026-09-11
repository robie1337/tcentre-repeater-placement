"""Do the hop limit, the link-length cap and the width grid change the CA9 answer?

Three approximations cut the problem down before the solver sees it:

  max_hops      paths of more than 20 logical links are never generated
  max_link_km   links longer than 300 km are not in the logical graph
  width grid    each path is offered at 8 log-spaced widths, not at every
                width from 1 to 100

None of them is physics. This script re-solves the preset CA9 instances with
each one loosened and reports whether utility, pairs served, repeaters built
or the chosen sites move.

Instances: midrange and projected presets, unlimited budget and budgets of
20 and 10, serving optional, under Eq. (2) and the buffered rate model. The
width check adds the memoryless model, whose dependence on width is the
least like log2(W), which is what the eight-point grid was chosen for.

Run:  python scripts/w8_truncation_checks.py [--jobs 6] [--backend gurobi]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import RESULTS, save_frame
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import NetworkConfig, log_width_grid, solve_placement
from qrp.paths import enumerate_paths, path_statistics
from qrp.topology import build_ca9

UNLIMITED = 10**6
TIME_LIMIT_S = 900.0
PRESETS = {"midrange": TCENTRE_MIDRANGE, "projected": TCENTRE_PROJECTED}
BUDGETS = [UNLIMITED, 20, 10]
HOPS = [12, 16, 20, 24, 30]
LINKS_KM = [200.0, 300.0, 400.0, 500.0, 600.0]
GRIDS = {"log8": None, "log16": log_width_grid(100, 16), "log32": log_width_grid(100, 32),
         "all": tuple(range(1, 101))}
DEFAULT = {"max_hops": 20, "max_link_km": 300.0, "width_grid": "log8"}
LOOSEST = {"max_hops": 30, "max_link_km": 600.0, "width_grid": "all"}
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def budget_label(b: int) -> str:
    return "unlimited" if b >= UNLIMITED else str(b)


def solve_instance(topo, paths, check, setting, hardware_key, budget, rate_model, grid_name,
                   backend="highs"):
    config = NetworkConfig(max_repeaters=budget, require_all_pairs=False, rate_model=rate_model,
                           width_grid=GRIDS[grid_name])
    started = time.perf_counter()
    result = solve_placement(topo, paths, PRESETS[hardware_key], config, backend=backend,
                             time_limit_s=TIME_LIMIT_S, mip_gap=1e-6)
    return {
        "backend": backend,
        "check": check, "setting": setting, "hardware": hardware_key, "budget": budget,
        "rate_model": rate_model, "width_grid": grid_name, "status": result.status,
        "utility": result.utility, "served_pairs": result.served_pairs,
        "n_repeaters": result.n_repeaters, "sites": ";".join(sorted(result.repeaters)),
        "n_model_vars": result.n_model_vars, "mip_gap": result.mip_gap,
        "solve_s": time.perf_counter() - started,
    }


def report(frame: pd.DataFrame, check: str, reference) -> None:
    rows = frame[frame["check"] == check]
    key = ["hardware", "budget", "rate_model"]
    ref = rows[rows["setting"] == reference].set_index(key)
    say(f"   reference: {check} = {reference}")
    say(f"   {'setting':>8} | {'solved':>6} | {'same pairs':>10} {'same repeaters':>14}"
        f" {'same sites':>10} | {'max |dU| bits':>13} | {'median vars':>11} {'median solve s':>14}")
    for setting, g in rows.groupby("setting", sort=False):
        g = g.set_index(key)
        ok = (g["status"] == "optimal") & (ref.loc[g.index, "status"] == "optimal")
        g_ok, r_ok = g[ok], ref.loc[g.index][ok]
        same_pairs = int((g_ok["served_pairs"] == r_ok["served_pairs"]).sum())
        same_reps = int((g_ok["n_repeaters"] == r_ok["n_repeaters"]).sum())
        same_sites = int((g_ok["sites"] == r_ok["sites"]).sum())
        du = (g_ok["utility"] - r_ok["utility"]).abs().max() if len(g_ok) else float("nan")
        say(f"   {str(setting):>8} | {int(ok.sum()):>2}/{len(g):<3} | {same_pairs:>10} {same_reps:>14}"
            f" {same_sites:>10} | {du:>13.4f} | {g['n_model_vars'].median():>11.0f}"
            f" {g['solve_s'].median():>14.1f}")
    moved = []
    for setting, g in rows.groupby("setting", sort=False):
        if setting == reference:
            continue
        for idx, r in g.set_index(key).iterrows():
            base = ref.loc[idx]
            if r["status"] == "optimal" and base["status"] == "optimal" and (
                    r["served_pairs"] != base["served_pairs"] or r["n_repeaters"] != base["n_repeaters"]):
                moved.append(f"{setting}: {idx[0]} budget {budget_label(idx[1])} {idx[2]}"
                             f" pairs {r['served_pairs']} vs {base['served_pairs']},"
                             f" repeaters {r['n_repeaters']} vs {base['n_repeaters']},"
                             f" utility {r['utility']:.3f} vs {base['utility']:.3f}")
    for line in moved:
        say(f"     changed: {line}")
    if not moved:
        say("     no setting changes pairs served or repeaters built")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", default="highs", choices=["highs", "gurobi", "cplex"])
    args = parser.parse_args()

    from joblib import Parallel, delayed

    say("=" * 78)
    say("W8: DO THE HOP LIMIT, THE LINK CAP AND THE WIDTH GRID CHANGE THE ANSWER?")
    say("=" * 78)
    topo = build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()

    tasks = []
    path_sets = {}
    for hops in HOPS:
        path_sets[("max_hops", hops)] = (300.0, hops)
    for link in LINKS_KM:
        path_sets[("max_link_km", link)] = (link, 20)
    built = {}
    for (check, setting), (link, hops) in path_sets.items():
        if (link, hops) not in built:
            started = time.perf_counter()
            built[(link, hops)] = enumerate_paths(topo, pairs, max_link_km=link, max_hops=hops)
            say(f"   paths at max_link {link:.0f} km, max_hops {hops}: "
                f"{path_statistics(built[(link, hops)])} ({time.perf_counter() - started:.1f} s)")
        for hw in PRESETS:
            for budget in BUDGETS:
                for rate in ("paper", "coordinated"):
                    tasks.append((built[(link, hops)], check, setting, hw, budget, rate, "log8"))
    for grid in GRIDS:
        for hw in PRESETS:
            for budget in BUDGETS:
                for rate in ("paper", "coordinated", "ext"):
                    tasks.append((built[(300.0, 20)], "width_grid", grid, hw, budget, rate, grid))

    say(f"\n   {len(tasks)} solves on {args.backend}, time limit {TIME_LIMIT_S:.0f} s each")
    started = time.time()
    rows = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(solve_instance)(topo, paths, check, setting, hw, budget, rate, grid, args.backend)
        for paths, check, setting, hw, budget, rate, grid in tasks
    )
    frame = pd.DataFrame.from_records(rows)
    say(f"   done in {(time.time() - started) / 60:.1f} min;"
        f" {int((frame['status'] != 'optimal').sum())} solves not optimal")
    save_frame(frame, "w8_truncation.csv")

    for check in ("max_hops", "max_link_km", "width_grid"):
        say(f"\n-- {check} (default {DEFAULT[check]})")
        report(frame, check, LOOSEST[check])

    say("\n" + "=" * 78)
    (RESULTS / "w8_truncation.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("   wrote results/w8_truncation.log")


if __name__ == "__main__":
    main()
