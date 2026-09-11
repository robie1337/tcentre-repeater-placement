"""Does correcting the rate equation change the placement decision?

Finding 4 says Eq. (2) of the source paper sits outside its stated regime at
1326 nm in every solve. The question that decides whether that is a paper or
a footnote is whether it changes what a planner would build. A rate bias that
scaled every candidate by the same factor would leave the optimal placement
untouched, because the objective is a sum of logarithms. A bias that varies
from path to path could move it.

The experiment
--------------
For each hardware point and repeater budget, solve the placement three times:
under the paper's Eq. (2), and under the two low-W*p models already in
physics.py (``coordinated``, the buffered model, and ``ext``, Q-CAST's
memoryless slotted model). Only the objective coefficients differ between the
three; the constraints are identical. So every model's solution is feasible
in every other model and can be scored under that model's utility.

Each instance is classed against a corrected model C:

  identical   the same (pair, path, width) selections
  tie         different selections that score no worse under C
  changed     the paper's selections score worse under C than C's own

Changed instances are further split by what moved: only the set of pairs
served, the routing of pairs both plans serve, or only memory widths.

Regret = U_C(C's choice) - U_C(paper's choice), in bits of log2 utility.
Divided by the pairs served it is the loss in log2 of R * (F - 1/2) per pair
when a planner trusts Eq. (2) and the corrected model is right. The reverse
regret, scored under Eq. (2), is also recorded.

Two objectives
--------------
``published``    the paper's objective with serving optional. A pair is served
                 only when log2(R (F - 1/2)) > 0, i.e. R (F - 1/2) above one
                 per second. That threshold depends on the time unit, so
                 changes in which pairs are served are partly a unit artefact.
``serve_first``  a large constant is added to every candidate's coefficient,
                 so the solver maximises pairs served first and utility
                 second. Serving feasibility has no rate term, so the served
                 set cannot depend on the rate model here, and every
                 remaining difference is routing, sites or memory.

A second experiment raises the memory ceiling to 2048 on a subset, under
serve_first, to see whether the models disagree on memory allocation.

Run:  python scripts/w4_rate_model_decisions.py [--jobs 20] [--n 300] [--quick]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED, hardware_from_sweep
from qrp.model import NetworkConfig, build_model, log_width_grid
from qrp.paths import enumerate_paths
from qrp.solver import solve
from qrp.sweep import latin_hypercube_points
from qrp.topology import build_ca9

UNLIMITED = 10**6
SERVE_BONUS = 1000.0
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


# --------------------------------------------------------------------------
# One solve, keeping enough to rescore the answer under another model
# --------------------------------------------------------------------------


def solve_instance(topo, paths, hardware, model_name, budget, memories, width_grid,
                   time_limit_s, mip_gap, serve_bonus):
    config = NetworkConfig(
        repeater_memories=memories,
        endnode_memories=memories,
        max_repeaters=budget,
        require_all_pairs=False,
        use_demand_weights=False,
        rate_model=model_name,
        width_grid=width_grid,
    )
    built = build_model(topo, paths, hardware, config)
    if built is None:
        # No usable candidate at all. Serving is optional, so the empty
        # selection is the optimum with utility zero.
        return {"status": "optimal", "chosen": frozenset(), "utils": {}, "cands": {}}

    utils, cands = {}, {}
    for cand in built.candidates:
        key = (cand.pair, cand.path.nodes, cand.width)
        utils[key] = cand.utility
        cands[key] = cand

    if serve_bonus:
        built.problem.c[: built.n_candidates] += serve_bonus

    result = solve(built.problem, backend="highs", time_limit_s=time_limit_s, mip_gap=mip_gap)
    if not result.feasible:
        return {"status": result.status, "chosen": frozenset(), "utils": utils, "cands": cands}

    chosen = frozenset(
        (built.candidates[j].pair, built.candidates[j].path.nodes, built.candidates[j].width)
        for j in range(built.n_candidates)
        if result.x[j] > 0.5
    )
    return {"status": "optimal", "chosen": chosen, "utils": utils, "cands": cands}


def score(chosen, utils) -> float:
    """Utility of a selection under one model; -inf if any path is unusable there."""
    total = 0.0
    for key in chosen:
        value = utils.get(key)
        if value is None:
            return -math.inf
        total += value
    return total


def repeaters_of(chosen) -> set[str]:
    return {node for (_, nodes, _) in chosen for node in nodes[1:-1]}


def mean_or_nan(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------
# Comparing the paper's plan with a corrected model's plan
# --------------------------------------------------------------------------


def compare(paper, other, tag: str) -> dict:
    kp, ko = paper["chosen"], other["chosen"]
    u_other_own = score(ko, other["utils"])
    u_other_paper = score(kp, other["utils"])
    u_paper_own = score(kp, paper["utils"])
    u_paper_other = score(ko, paper["utils"])

    regret = u_other_own - u_other_paper
    reverse = u_paper_own - u_paper_other
    tol = 1e-3 + 1e-5 * abs(u_other_own)

    reps_p, reps_o = repeaters_of(kp), repeaters_of(ko)
    union = reps_p | reps_o
    pairs_p = {k[0]: k for k in kp}
    pairs_o = {k[0]: k for k in ko}
    common = pairs_p.keys() & pairs_o.keys()
    path_diff = sum(pairs_p[p][1] != pairs_o[p][1] for p in common)
    width_diff = sum(pairs_p[p][2] != pairs_o[p][2] for p in common)
    served_diff = set(pairs_p) != set(pairs_o)

    if kp == ko:
        outcome, kind = "identical", ""
    elif regret <= tol:
        outcome, kind = "tie", ""
    else:
        outcome = "changed"
        if path_diff:
            kind = "routing"
        elif served_diff and not width_diff:
            kind = "served_set_only"
        elif width_diff and not served_diff:
            kind = "memory_only"
        else:
            kind = "served_and_memory"

    # Rate the corrected model assigns to each plan's path, on pairs both serve.
    log_ratios = []
    for pair in common:
        c_own = other["cands"].get(pairs_o[pair])
        c_paper = other["cands"].get(pairs_p[pair])
        if c_own is not None and c_paper is not None and c_paper.rate > 0 and c_own.rate > 0:
            log_ratios.append(math.log2(c_own.rate / c_paper.rate))

    served = len(ko)
    return {
        f"{tag}_outcome": outcome,
        f"{tag}_change_kind": kind,
        f"{tag}_regret_bits": regret,
        f"{tag}_reverse_regret_bits": reverse,
        f"{tag}_regret_bits_per_pair": regret / served if served else 0.0,
        f"{tag}_site_jaccard": (len(reps_p & reps_o) / len(union)) if union else 1.0,
        f"{tag}_sites_only_paper": len(reps_p - reps_o),
        f"{tag}_sites_only_other": len(reps_o - reps_p),
        f"{tag}_d_repeaters": len(reps_o) - len(reps_p),
        f"{tag}_d_served": len(ko) - len(kp),
        f"{tag}_pairs_path_differs": path_diff,
        f"{tag}_pairs_width_differs": width_diff,
        f"{tag}_d_mean_hops": mean_or_nan(len(pairs_o[p][1]) - len(pairs_p[p][1]) for p in common),
        f"{tag}_d_mean_width": mean_or_nan(pairs_o[p][2] - pairs_p[p][2] for p in common),
        f"{tag}_rate_gain_bits_common": mean_or_nan(log_ratios),
    }


# --------------------------------------------------------------------------
# One hardware point, every budget
# --------------------------------------------------------------------------


def run_point(topo, paths, point, budgets, memories, width_grid, models, time_limit_s,
              mip_gap, serve_bonus, detail):
    hardware = point["hardware"]
    records, details = [], []

    for budget in budgets:
        sols = {
            m: solve_instance(topo, paths, hardware, m, budget, memories, width_grid,
                              time_limit_s, mip_gap, serve_bonus)
            for m in models
        }
        paper = sols["paper"]
        record = {
            "point": point["id"],
            "source": point["source"],
            **point["values"],
            "budget": budget,
            "memories": memories,
        }
        for m in models:
            s = sols[m]
            record[f"{m}_status"] = s["status"]
            record[f"{m}_utility"] = score(s["chosen"], s["utils"]) if s["status"] == "optimal" else float("nan")
            record[f"{m}_repeaters"] = len(repeaters_of(s["chosen"]))
            record[f"{m}_served"] = len(s["chosen"])
            record[f"{m}_mean_hops"] = mean_or_nan(len(k[1]) - 1 for k in s["chosen"])

        wp = [paper["cands"][k].wp_min for k in paper["chosen"] if k in paper["cands"]]
        record["paper_min_wp"] = min(wp) if wp else float("nan")
        record["paper_median_wp"] = float(np.median(wp)) if wp else float("nan")

        for m in models:
            if m == "paper":
                continue
            if paper["status"] == "optimal" and sols[m]["status"] == "optimal":
                record.update(compare(paper, sols[m], m))
            else:
                record[f"{m}_outcome"] = "solver_problem"
        records.append(record)

        if detail:
            for m in models:
                for key in sorted(sols[m]["chosen"]):
                    cand = sols[m]["cands"][key]
                    row = {
                        "point": point["id"],
                        "budget": budget,
                        "memories": memories,
                        "plan": m,
                        "pair": f"{key[0][0]}-{key[0][1]}",
                        "hops": cand.path.hops,
                        "width": cand.width,
                        "max_link_km": round(cand.path.max_link_km, 1),
                        "repeaters": ";".join(cand.path.repeaters),
                        "wp_min": cand.wp_min,
                    }
                    for scorer in models:
                        scored = sols[scorer]["utils"].get(key)
                        row[f"utility_under_{scorer}"] = scored if scored is not None else -math.inf
                    details.append(row)

    return records, details


def run_all(topo, paths, points, budgets, memories, width_grid, models, jobs, time_limit_s,
            mip_gap, serve_bonus):
    from joblib import Parallel, delayed

    tasks = (
        delayed(run_point)(topo, paths, p, budgets, memories, width_grid, models, time_limit_s,
                           mip_gap, serve_bonus, p["source"] != "lhs")
        for p in points
    )
    out = Parallel(n_jobs=jobs, backend="loky", verbose=0)(tasks)
    records = [r for recs, _ in out for r in recs]
    details = [d for _, dets in out for d in dets]
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(details)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def budget_label(b: int) -> str:
    return "unlimited" if b >= UNLIMITED else str(b)


def summarise(frame: pd.DataFrame, tag: str) -> None:
    usable = frame[frame[f"{tag}_outcome"] != "solver_problem"]
    dropped = len(frame) - len(usable)
    if dropped:
        say(f"   ({dropped} instances dropped for a solver status other than optimal)")

    say(f"   {'budget':>9} | {'n':>4} | {'ident':>5} | {'tie':>4} | {'changed':>11} |"
        f" routing  served-set  memory  both | regret bits/pair: median   p90    max")
    for budget, g in usable.groupby("budget", sort=False):
        counts = g[f"{tag}_outcome"].value_counts()
        changed = g[g[f"{tag}_outcome"] == "changed"]
        kinds = changed[f"{tag}_change_kind"].value_counts()
        per_pair = changed[f"{tag}_regret_bits_per_pair"]
        finite = per_pair[np.isfinite(per_pair)]
        n_inf = int((~np.isfinite(per_pair)).sum())
        n_ch = int(counts.get("changed", 0))
        line = (
            f"   {budget_label(budget):>9} | {len(g):>4} | {int(counts.get('identical', 0)):>5} |"
            f" {int(counts.get('tie', 0)):>4} | {n_ch:>4} ({100 * n_ch / len(g):>3.0f}%) |"
            f" {int(kinds.get('routing', 0)):>7} {int(kinds.get('served_set_only', 0)):>11}"
            f" {int(kinds.get('memory_only', 0)):>7} {int(kinds.get('served_and_memory', 0)):>5} |"
        )
        if len(finite):
            line += f" {finite.median():>22.3f} {finite.quantile(0.9):>6.3f} {finite.max():>6.3f}"
        if n_inf:
            line += f"  +{n_inf} with a zero-rate path under {tag}"
        say(line)


def direction(frame: pd.DataFrame, tag: str, kind: str | None = None) -> None:
    changed = frame[frame[f"{tag}_outcome"] == "changed"]
    if kind:
        changed = changed[changed[f"{tag}_change_kind"] == kind]
    if changed.empty:
        say("     none")
        return
    say(f"     over {len(changed)} instances ({tag} minus paper):")
    for col, label in [
        (f"{tag}_d_repeaters", "repeaters built"),
        (f"{tag}_d_served", "pairs served"),
        (f"{tag}_d_mean_hops", "mean hops, pairs both serve"),
        (f"{tag}_d_mean_width", "mean width, pairs both serve"),
    ]:
        v = changed[col].dropna()
        say(f"       {label:<30} mean {v.mean():+7.2f} | fewer {int((v < 0).sum()):>4} |"
            f" same {int((v == 0).sum()):>4} | more {int((v > 0).sum()):>4}")
    j = changed[f"{tag}_site_jaccard"]
    p = changed[f"{tag}_pairs_path_differs"]
    g = changed[f"{tag}_rate_gain_bits_common"].dropna()
    say(f"       site-set overlap (Jaccard)     median {j.median():.2f} | min {j.min():.2f}")
    say(f"       pairs routed differently       median {p.median():.0f} | max {p.max():.0f}")
    if len(g):
        say(f"       corrected rate gained on shared pairs, bits  median {g.median():.3f} | max {g.max():.3f}")


def where(frame: pd.DataFrame, tag: str) -> None:
    lhs = frame[(frame["source"] == "lhs") & (frame[f"{tag}_outcome"] != "solver_problem")]
    if lhs.empty:
        return
    lhs = lhs.assign(changed=(lhs[f"{tag}_outcome"] == "changed").astype(int))
    bins = {
        "t2_s": ([0, 5e-3, 20e-3, 1.0], ["T2 < 5 ms", "5-20 ms", "> 20 ms"]),
        "generation_rate_hz": ([0, 1e4, 5e4, 1e9], ["rate < 10 kHz", "10-50 kHz", "> 50 kHz"]),
        "link_fidelity": ([0, 0.95, 0.975, 1.0], ["F_L < 0.95", "0.95-0.975", "> 0.975"]),
        "swap_success": ([0, 0.65, 0.8, 1.0], ["q_s < 0.65", "0.65-0.8", "> 0.8"]),
    }
    for col, (edges, labels) in bins.items():
        cut = pd.cut(lhs[col], bins=edges, labels=labels)
        shares = lhs.groupby(cut, observed=False)["changed"].agg(["mean", "size"])
        text = " | ".join(f"{lab}: {100 * row['mean']:.0f}% of {int(row['size'])}"
                          for lab, row in shares.iterrows())
        say(f"     {text}")


def presets_table(frame: pd.DataFrame, models) -> None:
    pre = frame[frame["source"] != "lhs"]
    for _, r in pre.iterrows():
        parts = [f"   {r['source']:<9} budget {budget_label(r['budget']):>9} |"]
        for m in models:
            parts.append(f" {m[:5]} {int(r[f'{m}_repeaters']):>2}r {int(r[f'{m}_served']):>2}p |")
        for m in models:
            if m == "paper":
                continue
            out = r.get(f"{m}_outcome", "")
            kind = r.get(f"{m}_change_kind", "")
            reg = r.get(f"{m}_regret_bits_per_pair", float("nan"))
            parts.append(f" vs {m[:5]}: {out:<9} {kind:<16} {reg:>7.3f} bits/pair |")
        say("".join(parts))


def report(frame: pd.DataFrame, models, heading: str) -> None:
    say(f"\n   Presets (r = repeaters built, p = pairs served):")
    presets_table(frame, models)
    for tag, label in [("coordinated", "buffered model (valid at low W*p)"),
                       ("ext", "memoryless model (Q-CAST EXT)")]:
        if tag not in models:
            continue
        say(f"\n   {heading}: paper's plan against the {label}")
        summarise(frame, tag)
        say("     Direction, all changed instances:")
        direction(frame, tag)
        say("     Direction, routing changes only:")
        direction(frame, tag, kind="routing")
        say("     Where in the sweep box the plan changes (share of Latin hypercube points):")
        where(frame, tag)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--n", type=int, default=300, help="Latin hypercube points")
    parser.add_argument("--memory-points", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--gap", type=float, default=1e-6)
    args = parser.parse_args()

    budgets = [UNLIMITED, 40, 30, 20, 15, 10, 6]
    n = args.n
    memory_points = args.memory_points
    suffix = ""
    if args.quick:
        budgets = [UNLIMITED, 20, 8]
        n = 8
        memory_points = 2
        suffix = "_quick"

    say("=" * 78)
    say("W4: DOES CORRECTING THE RATE EQUATION CHANGE THE PLACEMENT DECISION?")
    say("=" * 78)

    topo = build_ca9(spacing_km=80.0)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=20)

    def preset(name, hw):
        return {"id": name, "source": name, "hardware": hw,
                "values": {"generation_rate_hz": hw.generation_rate_hz,
                           "t2_s": hw.t_repeater_memory_s,
                           "link_fidelity": hw.link_fidelity,
                           "swap_success": hw.swap_success}}

    presets = [preset("midrange", TCENTRE_MIDRANGE), preset("projected", TCENTRE_PROJECTED)]
    lhs = [
        {"id": f"lhs{i:03d}", "source": "lhs", "hardware": hardware_from_sweep(v), "values": v}
        for i, v in enumerate(latin_hypercube_points(n=n, seed=20260910))
    ]
    points = presets + lhs
    models = ("paper", "coordinated", "ext")

    for number, (objective, bonus) in enumerate(
        [("published", 0.0), ("serve_first", SERVE_BONUS)], start=1
    ):
        say(f"\n-- {number}. Objective '{objective}': {len(points)} hardware points x"
            f" {len(budgets)} budgets x {len(models)} rate models, memory ceiling 100")
        started = time.time()
        frame, details = run_all(topo, paths, points, budgets, 100, None, models, args.jobs,
                                 args.time_limit, args.gap, bonus)
        say(f"   {len(frame) * len(models)} solves in {(time.time() - started) / 60:.1f} min")
        for m in models:
            bad = int((frame[f"{m}_status"] != "optimal").sum())
            if bad:
                say(f"   {bad} {m} solves did not return optimal")
        frame.insert(0, "objective", objective)
        save_frame(frame, f"w4_decisions_{objective}{suffix}.csv")
        save_frame(details, f"w4_preset_paths_{objective}{suffix}.csv")
        report(frame, models, objective)

    say(f"\n-- 3. Memory ceiling 2048, objective 'serve_first', paper vs buffered,"
        f" {len(presets) + memory_points} points")
    wide = log_width_grid(2048, n_points=12)
    started = time.time()
    mem_frame, mem_details = run_all(topo, paths, presets + lhs[:memory_points],
                                     [UNLIMITED, 20, 10], 2048, wide, ("paper", "coordinated"),
                                     args.jobs, args.time_limit, args.gap, SERVE_BONUS)
    say(f"   {len(mem_frame) * 2} solves in {(time.time() - started) / 60:.1f} min")
    save_frame(mem_frame, f"w4_memory2048{suffix}.csv")
    save_frame(mem_details, f"w4_memory2048_preset_paths{suffix}.csv")
    report(mem_frame, ("paper", "coordinated"), "memory 2048")

    say("\n" + "=" * 78)
    say("W4 COMPLETE")
    say("=" * 78)

    log_name = f"w4_decisions{suffix}.log"
    (RESULTS / log_name).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"   wrote results/{log_name}")


if __name__ == "__main__":
    main()
