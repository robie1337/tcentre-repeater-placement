"""Do HiGHS, Gurobi and CPLEX agree on models built from the default candidate set?

Under the default path generator the full CA9 model has about 4,454
variables, more than the size-limited Gurobi licence (2,000) or CPLEX
Community (1,000) accept, so the agreement check that used to run on CA9
now needs a paid licence. This script checks agreement on instances that fit
the free licences and still use the default generator:

  ca9_w1       CA9, default paths, one width (14)        all three solvers
  ca9_w3       CA9, default paths, widths 1, 14 and 100  HiGHS and Gurobi
  ca9_legacy   CA9, legacy paths, the usual eight widths all three solvers

across the presets and a Latin hypercube sample of hardware, three budgets,
three rate models and two coherence models. Every solve is exact (mip_gap
0). Two backends agree when they return the same status and, when both are
optimal, objectives equal to 1e-6 relative. A backend that reports
too_large is counted, not compared.

Run:  python scripts/w9_solver_agreement.py [--points 10] [--jobs 4]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED, hardware_from_sweep
from qrp.model import NetworkConfig, build_model
from qrp.paths import enumerate_paths, path_statistics
from qrp.solver import OPTIMAL, TOO_LARGE, available_backends, solve
from qrp.sweep import latin_hypercube_points
from qrp.topology import build_ca9

UNLIMITED = 10**6
TIME_LIMIT_S = 600.0
BUDGETS = [UNLIMITED, 20, 10]
RATE_MODELS = ("paper", "coordinated", "ext")
COHERENCE_MODELS = ("paper", "decay_optimistic")
SETTINGS = {
    "ca9_w1": ("candidates", (14,)),
    "ca9_w3": ("candidates", (1, 14, 100)),
    "ca9_legacy": ("legacy", None),
}
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def run_point(topo, path_sets, point, backends) -> list[dict]:
    rows = []
    for setting, (strategy, grid) in SETTINGS.items():
        paths = path_sets[strategy]
        for budget, rate, coherence in itertools.product(BUDGETS, RATE_MODELS, COHERENCE_MODELS):
            config = NetworkConfig(max_repeaters=budget, require_all_pairs=False, rate_model=rate,
                                   coherence_model=coherence, width_grid=grid)
            built = build_model(topo, paths, point["hardware"], config)
            base = {"point": point["id"], "setting": setting, "budget": budget,
                    "rate_model": rate, "coherence_model": coherence,
                    "n_vars": built.problem.n_vars if built else 0,
                    "n_rows": built.problem.A.shape[0] if built else 0}
            if built is None:
                rows.append({**base, "backend": "none", "status": "no_candidates",
                             "objective": 0.0, "solve_s": 0.0})
                continue
            for backend in backends:
                started = time.perf_counter()
                result = solve(built.problem, backend=backend, time_limit_s=TIME_LIMIT_S, mip_gap=0.0)
                rows.append({**base, "backend": backend, "status": result.status,
                             "objective": result.objective,
                             "solve_s": time.perf_counter() - started})
    return rows


def agreement(frame: pd.DataFrame, backends) -> None:
    key = ["point", "setting", "budget", "rate_model", "coherence_model"]
    wide = frame[frame["backend"].isin(backends)].pivot_table(
        index=key, columns="backend", values=["status", "objective"], aggfunc="first")
    say(f"   {'setting':<11} {'pair':<15} | {'compared':>8} {'status differs':>14}"
        f" {'objective differs':>17} {'max rel diff':>12} | too large (a / b)")
    for setting in SETTINGS:
        sub = wide[wide.index.get_level_values("setting") == setting]
        for a, b in itertools.combinations(backends, 2):
            sa, sb = sub[("status", a)], sub[("status", b)]
            too_large = (int((sa == TOO_LARGE).sum()), int((sb == TOO_LARGE).sum()))
            both_ran = (sa != TOO_LARGE) & (sb != TOO_LARGE)
            status_diff = int((sa[both_ran] != sb[both_ran]).sum())
            both_opt = both_ran & (sa == OPTIMAL) & (sb == OPTIMAL)
            oa, ob = sub[("objective", a)][both_opt].astype(float), sub[("objective", b)][both_opt].astype(float)
            rel = (oa - ob).abs() / np.maximum(1.0, np.maximum(oa.abs(), ob.abs()))
            obj_diff = int((rel > 1e-6).sum())
            max_rel = float(rel.max()) if len(rel) else float("nan")
            say(f"   {setting:<11} {a + '/' + b:<15} | {int(both_ran.sum()):>8} {status_diff:>14}"
                f" {obj_diff:>17} {max_rel:>12.2e} | {too_large[0]} / {too_large[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=10, help="Latin hypercube points besides the presets")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    from joblib import Parallel, delayed

    say("=" * 78)
    say("W9: DO HIGHS, GUROBI AND CPLEX AGREE ON THE DEFAULT CANDIDATE SET?")
    say("=" * 78)
    backends = [b for b in ("highs", "gurobi", "cplex") if b in available_backends()]
    say(f"   backends available: {backends}")
    if len(backends) < 2:
        say("   fewer than two backends, nothing to compare")
        return

    topo = build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    path_sets = {s: enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20, strategy=s)
                 for s in ("candidates", "legacy")}
    for strategy, paths in path_sets.items():
        say(f"   {strategy} paths: {path_statistics(paths)}")

    points = [{"id": "midrange", "hardware": TCENTRE_MIDRANGE},
              {"id": "projected", "hardware": TCENTRE_PROJECTED}]
    points += [{"id": f"lhs{i:03d}", "hardware": hardware_from_sweep(v)}
               for i, v in enumerate(latin_hypercube_points(n=args.points, seed=20260911))]

    started = time.time()
    out = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(run_point)(topo, path_sets, p, backends) for p in points
    )
    frame = pd.DataFrame.from_records([r for rows in out for r in rows])
    say(f"\n   {len(points)} hardware points, {frame[['point', 'setting', 'budget', 'rate_model', 'coherence_model']].drop_duplicates().shape[0]}"
        f" models, {len(frame)} solves in {(time.time() - started) / 60:.1f} min")
    sizes = frame.groupby("setting")["n_vars"].agg(["min", "max"])
    for setting, r in sizes.iterrows():
        say(f"   {setting:<11} variables {int(r['min'])} to {int(r['max'])}")
    statuses = frame.groupby(["backend", "status"]).size()
    say("   statuses: " + "; ".join(f"{b} {s} {n}" for (b, s), n in statuses.items()))
    save_frame(frame, "w9_solver_agreement.csv")

    say("")
    agreement(frame, backends)

    say("\n" + "=" * 78)
    (RESULTS / "w9_solver_agreement.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("   wrote results/w9_solver_agreement.log")


if __name__ == "__main__":
    main()
