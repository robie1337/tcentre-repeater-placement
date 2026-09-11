"""What will rerunning the committed sweeps on the default candidate set cost?

The Sobol, grid and Latin hypercube sweeps were solved on the legacy path set.
The default set makes each CA9 model about five times larger. This script
takes a sample of hardware points from each committed sweep, solves every one
on both path sets side by side with the settings the sweeps used (serving
optional, unlimited budget, mip_gap 1e-4, a 60 s limit), and scales the
measured times to the number of solves each sweep contains.

Two estimates are given, because this machine may be busy with other jobs:
the times measured here directly, and the committed legacy timings scaled by
the default-to-legacy ratio measured here, which cancels most of the load.
Solves on the default set that do not finish inside 60 s are solved again
with a 900 s limit, to see how long they actually need.

Run:  python scripts/w10_rerun_cost.py [--points 40] [--jobs 6]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from io import StringIO

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp.hardware import SWEEP_BOUNDS, TCENTRE_MIDRANGE
from qrp.model import NetworkConfig
from qrp.paths import enumerate_paths, path_statistics
from qrp.sweep import SweepContext, evaluate
from qrp.topology import build_ca9

#: Committed sweep files and the rate model each was solved with, read off the
#: scripts that wrote them (w2_*: Eq. (2); w3_fixes: buffered; w3b: memoryless).
SWEEPS = {
    "w2_sobol_samples": "paper",
    "w2_grid_rate_fidelity": "paper",
    "w2_grid_t2_swap": "paper",
    "w2_grid_t2_rate": "paper",
    "w2_lhs_samples": "paper",
    "w2_scan_generation_rate_hz": "paper",
    "w2_scan_link_fidelity": "paper",
    "w2_scan_swap_success": "paper",
    "w2_scan_t2_s": "paper",
    "w3_sobol_coordinated": "coordinated",
    "w3_grid_t2_rate_coordinated": "coordinated",
    "w3_sobol_5param": "coordinated",
    "w3b_sobol_5param_ext": "ext",
}
SWEEP_LIMIT_S = 60.0
LONG_LIMIT_S = 900.0
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def committed(name: str) -> pd.DataFrame:
    """The committed version, so reruns in the working tree cannot change the baseline."""
    blob = subprocess.run(["git", "show", f"HEAD:results/{name}.csv"], capture_output=True,
                          text=True, check=True).stdout
    return pd.read_csv(StringIO(blob), low_memory=False)


def solve_point(topo, paths, strategy, rate_model, values, limit_s):
    context = SweepContext(topology=topo, paths=paths,
                           config=NetworkConfig(require_all_pairs=False, rate_model=rate_model),
                           base_hardware=TCENTRE_MIDRANGE, time_limit_s=limit_s, mip_gap=1e-4,
                           path_strategy=strategy, spacing_km=80.0, max_link_km=300.0, max_hops=20)
    record = evaluate(context, values)
    record["limit_s"] = limit_s
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=40, help="sampled points per rate model")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    from joblib import Parallel, delayed

    say("=" * 78)
    say("W10: WHAT RERUNNING THE SWEEPS ON THE DEFAULT CANDIDATE SET WILL COST")
    say("=" * 78)

    topo = build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    path_sets = {s: enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20, strategy=s)
                 for s in ("legacy", "candidates")}
    for s, p in path_sets.items():
        say(f"   {s} paths: {path_statistics(p)}")

    sweeps = {name: committed(name) for name in SWEEPS}
    rng = np.random.default_rng(20260911)
    tasks = []
    for rate_model in ("paper", "coordinated", "ext"):
        pool = pd.concat([df for name, df in sweeps.items() if SWEEPS[name] == rate_model],
                         ignore_index=True)
        picked = pool.iloc[rng.choice(len(pool), size=min(args.points, len(pool)), replace=False)]
        for i, (_, row) in enumerate(picked.iterrows()):
            values = {k: float(row[k]) for k in SWEEP_BOUNDS if k in row and pd.notna(row[k])}
            for strategy in ("legacy", "candidates"):
                tasks.append((strategy, rate_model, i, values))

    say(f"\n   {len(tasks)} solves at the sweeps' {SWEEP_LIMIT_S:.0f} s limit")
    started = time.time()
    records = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(solve_point)(topo, path_sets[s], s, rm, v, SWEEP_LIMIT_S) for s, rm, _, v in tasks
    )
    for rec, (s, rm, i, _) in zip(records, tasks):
        rec["sample"] = i
    frame = pd.DataFrame.from_records(records)
    say(f"   done in {(time.time() - started) / 60:.1f} min")

    slow = [(t, r) for t, r in zip(tasks, records)
            if t[0] == "candidates" and r["status"] != "optimal"]
    if slow:
        say(f"   {len(slow)} default-set solves not optimal at {SWEEP_LIMIT_S:.0f} s;"
            f" solving them again at {LONG_LIMIT_S:.0f} s")
        again = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(solve_point)(topo, path_sets["candidates"], "candidates", t[1], t[3], LONG_LIMIT_S)
            for t, _ in slow
        )
        for (t, _), rec in zip(slow, again):
            rec["sample"] = t[2]
        frame = pd.concat([frame, pd.DataFrame.from_records(again)], ignore_index=True)
    save_frame(frame, "w10_rerun_cost.csv")

    base = frame[frame["limit_s"] == SWEEP_LIMIT_S]
    say(f"\n   {'rate model':<11} {'path set':<10} | {'median vars':>11} {'median s':>9} {'mean s':>8}"
        f" {'p90 s':>7} {'max s':>7} | not optimal at 60 s")
    ratio = {}
    for (rm, s), g in base.groupby(["rate_model", "path_strategy"], sort=False):
        say(f"   {rm:<11} {s:<10} | {g['n_model_vars'].median():>11.0f} {g['solve_seconds'].median():>9.2f}"
            f" {g['solve_seconds'].mean():>8.2f} {g['solve_seconds'].quantile(0.9):>7.1f}"
            f" {g['solve_seconds'].max():>7.1f} | {int((g['status'] != 'optimal').sum())} of {len(g)}")
    for rm, g in base.groupby("rate_model", sort=False):
        means = g.groupby("path_strategy")["solve_seconds"].mean()
        ratio[rm] = means["candidates"] / means["legacy"]
        say(f"   {rm}: default set takes {ratio[rm]:.1f}x the legacy time on the same points")

    long = frame[frame["limit_s"] == LONG_LIMIT_S]
    if len(long):
        say(f"   at {LONG_LIMIT_S:.0f} s: {int((long['status'] == 'optimal').sum())} of {len(long)}"
            f" finish, taking {long['solve_seconds'].median():.0f} s median and"
            f" {long['solve_seconds'].max():.0f} s at most")

    say(f"\n   {'sweep':<28} {'rate':<11} {'solves':>6} | {'legacy CPU h':>12} | default CPU h:"
        f" {'measured':>8} {'scaled':>7} | wall h at 6 jobs")
    totals = {"legacy": 0.0, "measured": 0.0, "scaled": 0.0}
    for name, rm in SWEEPS.items():
        df = sweeps[name]
        legacy_h = df["solve_seconds"].sum() / 3600
        cand_mean = base[(base["rate_model"] == rm) & (base["path_strategy"] == "candidates")]["solve_seconds"].mean()
        measured_h = len(df) * cand_mean / 3600
        scaled_h = legacy_h * ratio[rm]
        totals["legacy"] += legacy_h
        totals["measured"] += measured_h
        totals["scaled"] += scaled_h
        say(f"   {name:<28} {rm:<11} {len(df):>6} | {legacy_h:>12.2f} |               {measured_h:>8.1f}"
            f" {scaled_h:>7.1f} | {min(measured_h, scaled_h) / 6:>4.1f} to {max(measured_h, scaled_h) / 6:.1f}")
    say(f"   {'total':<28} {'':<11} {sum(len(d) for d in sweeps.values()):>6} | {totals['legacy']:>12.2f} |"
        f"               {totals['measured']:>8.1f} {totals['scaled']:>7.1f} |"
        f" {min(totals['measured'], totals['scaled']) / 6:>4.1f} to {max(totals['measured'], totals['scaled']) / 6:.1f}")
    say("   The limit matters as much as the time: a solve cut off at 60 s is not a result.")

    say("\n   Do the sampled points change answer between path sets? (both optimal)")
    both = base.pivot_table(index=["rate_model", "sample"], columns="path_strategy",
                            values=["status", "utility", "served_pairs"], aggfunc="first")
    ok = (both[("status", "legacy")] == "optimal") & (both[("status", "candidates")] == "optimal")
    b = both[ok]
    for rm in ("paper", "coordinated", "ext"):
        sub = b[b.index.get_level_values("rate_model") == rm]
        if not len(sub):
            continue
        du = sub[("utility", "candidates")] - sub[("utility", "legacy")]
        same = int((sub[("served_pairs", "candidates")] == sub[("served_pairs", "legacy")]).sum())
        say(f"   {rm:<11} {len(sub):>3} points: same pairs served on {same}; utility gain"
            f" median {du.median():.3f}, max {du.max():.3f} bits")

    say("\n" + "=" * 78)
    (RESULTS / "w10_rerun_cost.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("   wrote results/w10_rerun_cost.log")


if __name__ == "__main__":
    main()
