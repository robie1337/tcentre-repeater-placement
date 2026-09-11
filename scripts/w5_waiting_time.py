"""Does the coherence constraint need to bound waiting time, not propagation time?

The source paper gates each path on memory coherence with two hard cut-offs:
Eq. (13), 2 tau_l(longest link) <= T_RM, and Eq. (14), tau_e2e <= T_EM. Its
tau_e2e "includes the classical messages exchange between consecutive
repeaters", which is propagation. Neither constraint counts the time a
memory spends holding one link's pair while the other links on the path are
still failing and retrying.

In the regime Eq. (2) assumes, W * p_min >> 1, that wait is a round or two
and propagation is the right thing to bound. At 1326 nm W * p_min is around
0.03, a link needs tens of attempt rounds, and the wait can exceed T2 while
tau_e2e sits well inside it. This script measures how much of the published
plan that affects and whether a waiting-aware gate changes the plan.

The waiting model
-----------------
Each link e runs W multiplexed attempts per round and succeeds in a round
with P_e = 1 - (1 - p_e)^W. A round lasts

  heralded   max(1 / generation rate, 2 l_e / c)   no retry before the herald
             returns (primary)
  source     1 / generation rate                   the paper's implicit clock

so link e is ready after an exponential time with rate
lambda_e = -ln(1 - P_e) / tau_e. Two gates, each a hard cut-off in the
paper's own style:

  storage    E[max_e T_e] - E[min_e T_e] + tau_e2e <= T2
             expected time the first ready pair waits for the last, plus the
             paper's own classical delay (primary)
  slowest    max_e E[T_e] <= T2
             mean wait of the slowest link on its own

Plans compared, all under the source paper's own objective (serving is
optional, as in the paper). ``--serve-first`` switches to maximising pairs
served first; at tight budgets that forces long, near-zero-rate links into
the published plan and inflates how much of it looks unreachable, so it is
not the default:

  base_paper     Eq. (2) rates, published gates        the published plan
  base_coord     buffered rates, published gates
  wait_paper_*   Eq. (2) rates plus a waiting gate      isolates the gate
  wait_coord_*   buffered rates plus a waiting gate     both corrections

Run:  python scripts/w5_waiting_time.py [--jobs 20] [--n 300] [--quick]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp import physics, waiting
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED, hardware_from_sweep
from qrp.model import NetworkConfig, build_model, log_width_grid
from qrp.paths import enumerate_paths, path_statistics
from qrp.solver import solve
from qrp.sweep import latin_hypercube_points
from qrp.topology import build_ca9

UNLIMITED = 10**6
SERVE_BONUS = 1000.0
#: Gate and clock per variant. Storage gates run inside the model as
#: coherence_model="waiting_gate"; the slowest-link gate has no model option
#: and goes through build_model's candidate_filter.
VARIANTS = {
    "storage_heralded": ("storage", "heralded"),
    "storage_source": ("storage", "source"),
    "slowest_heralded": ("slowest", "heralded"),
}
PRIMARY = "storage_heralded"
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


# --------------------------------------------------------------------------
# Waiting times
# --------------------------------------------------------------------------


def waiting_table(paths, hardware, widths) -> dict:
    """(path nodes, width, clock) -> (storage seconds, slowest-link seconds).

    The formulas are qrp.waiting's, the same ones the model applies under
    ``coherence_model="waiting_gate"``.
    """
    table = {}
    for pair_paths in paths.values():
        for path in pair_paths:
            probs = [physics.link_success(L, hardware.alpha_db_per_km, hardware.eta_emission,
                                          hardware.eta_detection)
                     for L in path.link_lengths_km]
            tau_e2e = physics.tau_e2e(path.link_lengths_km)
            for w in widths:
                for clock in waiting.WAITING_CLOCKS:
                    lams = waiting.link_ready_rates(path.link_lengths_km, probs, w,
                                                    hardware.generation_rate_hz, clock)
                    storage = waiting.expected_storage_time(lams, tau_e2e)
                    slowest = float(np.max(1.0 / lams))
                    table[(path.nodes, w, clock)] = (storage, slowest)
    return table


def allowed_set(table, t2, variant) -> set:
    gate, clock = VARIANTS[variant]
    index = 0 if gate == "storage" else 1
    return {(nodes, w) for (nodes, w, c), values in table.items()
            if c == clock and values[index] <= t2}


# --------------------------------------------------------------------------
# Solving with an optional extra gate
# --------------------------------------------------------------------------


def solve_plan(topo, paths, hardware, rate_model, budget, memories, width_grid, variant,
               allowed, time_limit_s, mip_gap, serve_bonus):
    """Solve one plan. ``variant`` is None for the paper's gates, or a VARIANTS key.

    Storage gates run inside the model as coherence_model="waiting_gate" with
    the variant's clock. The slowest-link gate has no model option, so it
    passes its precomputed ``allowed`` set through build_model's
    candidate_filter.
    """
    coherence_model, clock, candidate_filter = "paper", "heralded", None
    if variant is not None:
        gate, clock = VARIANTS[variant]
        if gate == "storage":
            coherence_model = "waiting_gate"
        else:
            def candidate_filter(c):
                return (c.path.nodes, c.width) in allowed
    config = NetworkConfig(
        repeater_memories=memories,
        endnode_memories=memories,
        max_repeaters=budget,
        require_all_pairs=False,
        use_demand_weights=False,
        rate_model=rate_model,
        width_grid=width_grid,
        coherence_model=coherence_model,
        waiting_clock=clock,
    )
    built = build_model(topo, paths, hardware, config, candidate_filter=candidate_filter)

    if built is None:
        return {"status": "optimal", "chosen": frozenset(), "cands": {}}
    cands = {(c.pair, c.path.nodes, c.width): c for c in built.candidates}
    if serve_bonus:
        built.problem.c[: built.n_candidates] += serve_bonus
    result = solve(built.problem, backend="highs", time_limit_s=time_limit_s, mip_gap=mip_gap)
    if not result.optimal:
        return {"status": result.status, "chosen": frozenset(), "cands": cands}
    chosen = frozenset(
        (built.candidates[j].pair, built.candidates[j].path.nodes, built.candidates[j].width)
        for j in range(built.n_candidates)
        if result.x[j] > 0.5
    )
    return {"status": "optimal", "chosen": chosen, "cands": cands}


def repeaters_of(chosen) -> set[str]:
    return {node for (_, nodes, _) in chosen for node in nodes[1:-1]}


def mean_or_nan(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def compare(base, other, tag) -> dict:
    kb, ko = base["chosen"], other["chosen"]
    reps_b, reps_o = repeaters_of(kb), repeaters_of(ko)
    union = reps_b | reps_o
    pairs_b = {k[0]: k for k in kb}
    pairs_o = {k[0]: k for k in ko}
    common = pairs_b.keys() & pairs_o.keys()
    path_diff = sum(pairs_b[p][1] != pairs_o[p][1] for p in common)
    width_diff = sum(pairs_b[p][2] != pairs_o[p][2] for p in common)
    served_diff = set(pairs_b) != set(pairs_o)
    if kb == ko:
        kind = "identical"
    elif path_diff:
        kind = "routing"
    elif served_diff and width_diff:
        kind = "served_and_memory"
    elif served_diff:
        kind = "served_set_only"
    else:
        kind = "memory_only"
    return {
        f"{tag}_kind": kind,
        f"{tag}_served": len(ko),
        f"{tag}_repeaters": len(reps_o),
        f"{tag}_d_served": len(ko) - len(kb),
        f"{tag}_d_repeaters": len(reps_o) - len(reps_b),
        f"{tag}_site_jaccard": (len(reps_b & reps_o) / len(union)) if union else 1.0,
        f"{tag}_pairs_path_differs": path_diff,
        f"{tag}_pairs_width_differs": width_diff,
        f"{tag}_d_mean_hops": mean_or_nan(len(pairs_o[p][1]) - len(pairs_b[p][1]) for p in common),
        f"{tag}_d_mean_width": mean_or_nan(pairs_o[p][2] - pairs_b[p][2] for p in common),
        f"{tag}_d_mean_worst_link_km": mean_or_nan(
            other["cands"][pairs_o[p]].path.max_link_km - base["cands"][pairs_b[p]].path.max_link_km
            for p in common
        ),
    }


# --------------------------------------------------------------------------
# One hardware point
# --------------------------------------------------------------------------


def run_point(topo, paths, point, budgets, memories, width_grid, variants, time_limit_s,
              mip_gap, serve_bonus, detail):
    hardware = point["hardware"]
    t2 = min(hardware.t_repeater_memory_s, hardware.t_endnode_memory_s)
    config_widths = NetworkConfig(repeater_memories=memories, endnode_memories=memories,
                                  width_grid=width_grid).widths()
    table = waiting_table(paths, hardware, config_widths)
    allowed = {v: allowed_set(table, t2, v) for v in variants}

    def solve_for(rate_model, budget, variant=None):
        return solve_plan(topo, paths, hardware, rate_model, budget, memories, width_grid,
                          variant, allowed.get(variant), time_limit_s, mip_gap, serve_bonus)

    records, details = [], []
    for budget in budgets:
        plans = {"base_paper": solve_for("paper", budget),
                 "base_coord": solve_for("coordinated", budget)}
        for v in variants:
            plans[f"wait_paper_{v}"] = solve_for("paper", budget, v)
            plans[f"wait_coord_{v}"] = solve_for("coordinated", budget, v)

        base = plans["base_paper"]
        record = {"point": point["id"], "source": point["source"], **point["values"],
                  "budget": budget, "memories": memories,
                  "statuses_ok": int(all(p["status"] == "optimal" for p in plans.values())),
                  "base_paper_served": len(base["chosen"]),
                  "base_paper_repeaters": len(repeaters_of(base["chosen"])),
                  "base_coord_served": len(plans["base_coord"]["chosen"]),
                  "base_coord_repeaters": len(repeaters_of(plans["base_coord"]["chosen"]))}

        # How much of the published plan fails each waiting gate.
        for v in variants:
            gate, clock = VARIANTS[v]
            idx = 0 if gate == "storage" else 1
            fails = [k for k in base["chosen"] if (k[1], k[2]) not in allowed[v]]
            record[f"unreachable_{v}"] = len(fails)
            ratios = [table[(k[1], k[2], clock)][idx] / t2 for k in base["chosen"]]
            record[f"median_wait_over_t2_{v}"] = float(np.median(ratios)) if ratios else float("nan")

        for name, plan in plans.items():
            if name == "base_paper":
                continue
            record.update(compare(base, plan, name))
        records.append(record)

        if detail:
            for name, plan in plans.items():
                for key in sorted(plan["chosen"]):
                    cand = plan["cands"][key]
                    storage, slowest = table[(key[1], key[2], "heralded")]
                    details.append({
                        "point": point["id"], "budget": budget, "memories": memories,
                        "plan": name, "pair": f"{key[0][0]}-{key[0][1]}", "hops": cand.path.hops,
                        "width": cand.width, "max_link_km": round(cand.path.max_link_km, 1),
                        "wp_min": cand.wp_min, "tau_e2e_ms": 1e3 * cand.tau_e2e_s,
                        "storage_ms_heralded": 1e3 * storage, "slowest_ms_heralded": 1e3 * slowest,
                        "t2_ms": 1e3 * t2, "repeaters": ";".join(cand.path.repeaters),
                    })
    return records, details


def run_all(topo, paths, points, budgets, memories, width_grid, variants, jobs, time_limit_s,
            mip_gap, serve_bonus):
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_point)(topo, paths, p, budgets, memories, width_grid, variants, time_limit_s,
                           mip_gap, serve_bonus, p["source"] != "lhs")
        for p in points
    )
    return (pd.DataFrame.from_records([r for recs, _ in out for r in recs]),
            pd.DataFrame.from_records([d for _, dets in out for d in dets]))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def budget_label(b: int) -> str:
    return "unlimited" if b >= UNLIMITED else str(b)


def presets_table(frame, variants) -> None:
    for _, r in frame[frame["source"] != "lhs"].iterrows():
        unfinished = "" if r["statuses_ok"] else "  [a solve stopped at its limit: counts below are not results]"
        say(f"   {r['source']:<9} budget {budget_label(r['budget']):>9} | published plan"
            f" {int(r['base_paper_served']):>2} pairs {int(r['base_paper_repeaters']):>2} repeaters{unfinished}")
        for v in variants:
            say(f"     {v:<18} unreachable in published plan {int(r[f'unreachable_{v}']):>2}"
                f" (median wait/T2 {r[f'median_wait_over_t2_{v}']:.2f}) | waiting-aware plan,"
                f" paper rates {int(r[f'wait_paper_{v}_served']):>2}p {int(r[f'wait_paper_{v}_repeaters']):>2}r"
                f" [{r[f'wait_paper_{v}_kind']}] | buffered rates {int(r[f'wait_coord_{v}_served']):>2}p"
                f" {int(r[f'wait_coord_{v}_repeaters']):>2}r [{r[f'wait_coord_{v}_kind']}]")


def summary(frame, variant, plan_prefix="wait_coord") -> None:
    lhs = frame[(frame["source"] == "lhs") & (frame["statuses_ok"] == 1)]
    tag = f"{plan_prefix}_{variant}"
    say(f"   {'budget':>9} | {'n':>4} | published plan has an unreachable pair | pairs unreachable:"
        f" mean  max | plan changes | routing | served | memory | coverage lost: mean  max")
    for budget, g in lhs.groupby("budget", sort=False):
        un = g[f"unreachable_{variant}"]
        kinds = g[f"{tag}_kind"].value_counts()
        n_change = int((g[f"{tag}_kind"] != "identical").sum())
        lost = -g[f"{tag}_d_served"]
        say(f"   {budget_label(budget):>9} | {len(g):>4} | {int((un > 0).sum()):>4}"
            f" ({100 * (un > 0).mean():>3.0f}%)                          |"
            f" {un.mean():>22.2f} {un.max():>4.0f} | {n_change:>4} ({100 * n_change / len(g):>3.0f}%) |"
            f" {int(kinds.get('routing', 0)):>7} | {int(kinds.get('served_set_only', 0)) + int(kinds.get('served_and_memory', 0)):>6} |"
            f" {int(kinds.get('memory_only', 0)):>6} | {lost.mean():>18.2f} {lost.max():>4.0f}")


def direction(frame, tag) -> None:
    changed = frame[(frame["source"] == "lhs") & (frame["statuses_ok"] == 1)
                    & (frame[f"{tag}_kind"] != "identical")]
    if changed.empty:
        say("     none")
        return
    say(f"     over {len(changed)} changed instances (waiting-aware minus published):")
    for col, label in [(f"{tag}_d_served", "pairs served"),
                       (f"{tag}_d_repeaters", "repeaters built"),
                       (f"{tag}_d_mean_hops", "mean hops, pairs both serve"),
                       (f"{tag}_d_mean_worst_link_km", "mean worst link km, both serve"),
                       (f"{tag}_d_mean_width", "mean width, pairs both serve")]:
        v = changed[col].dropna()
        if v.empty:
            continue
        say(f"       {label:<32} mean {v.mean():+8.2f} | lower {int((v < 0).sum()):>4} |"
            f" same {int((v == 0).sum()):>4} | higher {int((v > 0).sum()):>4}")
    j = changed[f"{tag}_site_jaccard"]
    say(f"       site-set overlap (Jaccard)       median {j.median():.2f} | min {j.min():.2f}")


def where(frame, variant) -> None:
    lhs = frame[(frame["source"] == "lhs") & (frame["budget"] >= UNLIMITED)]
    if lhs.empty:
        return
    lhs = lhs.assign(hit=(lhs[f"unreachable_{variant}"] > 0).astype(int))
    bins = {
        "t2_s": ([0, 5e-3, 20e-3, 1.0], ["T2 < 5 ms", "5-20 ms", "> 20 ms"]),
        "generation_rate_hz": ([0, 1e4, 5e4, 1e9], ["rate < 10 kHz", "10-50 kHz", "> 50 kHz"]),
    }
    for col, (edges, labels) in bins.items():
        cut = pd.cut(lhs[col], bins=edges, labels=labels)
        shares = lhs.groupby(cut, observed=False)["hit"].agg(["mean", "size"])
        say("     " + " | ".join(f"{lab}: {100 * row['mean']:.0f}% of {int(row['size'])}"
                                  for lab, row in shares.iterrows()))


def report(frame, variants, heading) -> None:
    say("\n   Presets:")
    presets_table(frame, variants)
    for v in variants:
        say(f"\n   {heading}, gate '{v}', buffered rates (both corrections), Latin hypercube points:")
        summary(frame, v, "wait_coord")
        say(f"   same gate, Eq. (2) rates (gate only):")
        summary(frame, v, "wait_paper")
        say("     Direction of the change, both corrections:")
        direction(frame, f"wait_coord_{v}")
        say("     Share of points (unlimited budget) whose published plan has an unreachable pair:")
        where(frame, v)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--memory-points", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    # The default candidate set makes the budgeted models several times larger;
    # 120 s left some solves unfinished when other jobs shared the machine.
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--gap", type=float, default=1e-6)
    parser.add_argument("--serve-first", action="store_true")
    parser.add_argument("--path-strategy", default="candidates", choices=["candidates", "legacy"])
    args = parser.parse_args()

    budgets = [UNLIMITED, 20, 10]
    bonus = SERVE_BONUS if args.serve_first else 0.0
    n, memory_points = args.n, args.memory_points
    suffix = "_servefirst" if args.serve_first else ""
    if args.quick:
        n, memory_points = 6, 2
        suffix += "_quick"
    if args.path_strategy != "candidates":
        suffix += f"_{args.path_strategy}"

    say("=" * 78)
    say("W5: DOES THE COHERENCE GATE NEED TO BOUND WAITING TIME?")
    say("=" * 78)
    say(f"   objective: {'serve_first' if args.serve_first else 'published (serving optional)'}")

    topo = build_ca9(spacing_km=80.0)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=20,
                            strategy=args.path_strategy)
    say(f"   candidate paths ({args.path_strategy}): {path_statistics(paths)}")

    def preset(name, hw):
        return {"id": name, "source": name, "hardware": hw,
                "values": {"generation_rate_hz": hw.generation_rate_hz,
                           "t2_s": hw.t_repeater_memory_s,
                           "link_fidelity": hw.link_fidelity,
                           "swap_success": hw.swap_success}}

    presets = [preset("midrange", TCENTRE_MIDRANGE), preset("projected", TCENTRE_PROJECTED)]
    lhs = [{"id": f"lhs{i:03d}", "source": "lhs", "hardware": hardware_from_sweep(v), "values": v}
           for i, v in enumerate(latin_hypercube_points(n=n, seed=20260910))]
    variants = list(VARIANTS)

    say(f"\n-- 1. Memory ceiling 100: {len(presets) + len(lhs)} points x {len(budgets)} budgets x"
        f" {2 + 2 * len(variants)} plans")
    started = time.time()
    frame, details = run_all(topo, paths, presets + lhs, budgets, 100, None, variants, args.jobs,
                             args.time_limit, args.gap, bonus)
    say(f"   done in {(time.time() - started) / 60:.1f} min;"
        f" {int((frame['statuses_ok'] == 0).sum())} instances with a non-optimal solve")
    save_frame(frame, f"w5_waiting{suffix}.csv")
    save_frame(details, f"w5_waiting_preset_paths{suffix}.csv")
    report(frame, variants, "memory 100")

    say(f"\n-- 2. Memory ceiling 2048, primary gate only: can buying memory recover the plan?")
    wide = log_width_grid(2048, n_points=12)
    started = time.time()
    mem, mem_details = run_all(topo, paths, presets + lhs[:memory_points], budgets, 2048, wide,
                               [PRIMARY], args.jobs, args.time_limit, args.gap, bonus)
    say(f"   done in {(time.time() - started) / 60:.1f} min")
    save_frame(mem, f"w5_waiting_memory2048{suffix}.csv")
    save_frame(mem_details, f"w5_waiting_memory2048_preset_paths{suffix}.csv")
    report(mem, [PRIMARY], "memory 2048")

    say("\n" + "=" * 78)
    say("W5 COMPLETE")
    say("=" * 78)
    (RESULTS / f"w5_waiting{suffix}.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"   wrote results/w5_waiting{suffix}.log")


if __name__ == "__main__":
    main()
