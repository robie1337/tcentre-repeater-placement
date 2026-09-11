"""How often does the default candidate path set reach the full path set's optimum?

The default generator (qrp.paths, strategy="candidates") is a heuristic. The
network MILP shares repeater memory and the repeater budget between pairs, so
no finite per-pair rule is guaranteed to keep every path the optimum needs.
This script measures how often the default set misses, and by how much, on
networks small enough to enumerate every simple path and solve exactly.

Two network families:
  random   3 to 5 cities, 3 to 7 sites, random links of 20 to 60 km
  ladder   two parallel rows of sites with rungs, four cities, link lengths
           of 30 or 40 km so that equal-quality routes are common, built so
           that pairs compete for the same sites

Every network gets a tight repeater budget (1 to 4) and small memories (4, 8
or 16), so pairs compete for repeaters.

Two campaigns:
  recovery   default settings (tie_cap 4, hop_slack 1, k_simple 6) under C band
             and O band hardware, three rate models and two coherence models
  settings   every combination of tie_cap {1, 2, 4, 8}, hop_slack {0, 1, 2, 3}
             and k_simple {2, 4, 6, 10} on a subset, to see what the defaults
             buy and what smaller settings lose

A candidate set is a subset of the full set, so its optimum can never be
higher. "Recovered" means the same objective to 1e-6 relative. An instance
where either solve is not optimal is recorded and left out of the rates.

Run:  python scripts/w7_candidate_recovery.py [--networks 400] [--settings-networks 80] [--jobs 8]
"""

from __future__ import annotations

import argparse
import itertools
import time

import networkx as nx
import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp.hardware import POURYOUSEF_BASELINE, TCENTRE_MIDRANGE
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from qrp.solver import INFEASIBLE, OPTIMAL
from qrp.topology import Topology

MAX_HOPS = 6
MAX_EXHAUSTIVE = 20_000
TIME_LIMIT_S = 120.0
DEFAULTS = {"tie_cap": 4, "hop_slack": 1, "k_simple": 6}
SETTINGS_GRID = list(itertools.product([1, 2, 4, 8], [0, 1, 2, 3], [2, 4, 6, 10]))

HARDWARE = {
    # The paper's baseline with a rate high enough that serving pays.
    "c_band": POURYOUSEF_BASELINE.with_(generation_rate_hz=1e4),
    # Midrange T centre at the measured nuclear coherence.
    "o_band": TCENTRE_MIDRANGE.with_(t_repeater_memory_s=0.112, t_endnode_memory_s=0.112),
}
RECOVERY_COMBOS = [
    ("c_band", "paper", "paper"),
    ("c_band", "coordinated", "paper"),
    ("o_band", "paper", "paper"),
    ("o_band", "coordinated", "paper"),
    ("o_band", "ext", "paper"),
    ("o_band", "paper", "decay_optimistic"),
]
SETTINGS_COMBOS = [("c_band", "paper", "paper"), ("o_band", "coordinated", "paper")]
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------


def topology_from_edges(edges, cities, name) -> Topology:
    graph = nx.Graph()
    for u, v, length in edges:
        for node in (u, v):
            if node not in graph:
                graph.add_node(node, kind="city" if node in cities else "site")
        graph.add_edge(u, v, length_km=float(length))
    sites = tuple(n for n in graph.nodes if n not in set(cities))
    return Topology(graph=graph, cities=tuple(cities), sites=sites, name=name, spacing_km=0.0)


def random_network(rng) -> tuple[Topology, float]:
    cities = [f"C{i}" for i in range(int(rng.integers(3, 6)))]
    sites = [f"S{i}" for i in range(int(rng.integers(3, 8)))]
    nodes = cities + sites
    density = float(rng.uniform(0.2, 0.45))
    order = list(rng.permutation(nodes))
    edges = [(a, b, float(rng.uniform(20, 60))) for a, b in zip(order[:-1], order[1:])]
    present = {frozenset((a, b)) for a, b, _ in edges}
    for a, b in itertools.combinations(nodes, 2):
        if frozenset((a, b)) not in present and rng.random() < density:
            edges.append((a, b, float(rng.uniform(20, 60))))
    return topology_from_edges(edges, cities, "random"), 70.0


def ladder_network(rng) -> tuple[Topology, float]:
    rungs = int(rng.integers(3, 6))
    top = [f"T{i}" for i in range(rungs)]
    bottom = [f"B{i}" for i in range(rungs)]
    length = lambda: float(rng.choice([30.0, 40.0]))  # noqa: E731
    edges = []
    for row in (top, bottom):
        edges += [(a, b, length()) for a, b in zip(row[:-1], row[1:])]
    edges += [(t, b, length()) for t, b in zip(top, bottom)]
    mid = rungs // 2
    edges += [("West", top[0], length()), ("West", bottom[0], length()),
              ("East", top[-1], length()), ("East", bottom[-1], length()),
              ("North", top[mid], length()), ("South", bottom[mid], length())]
    return topology_from_edges(edges, ["West", "East", "North", "South"], "ladder"), 45.0


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------


def objective(result) -> tuple[float, int, frozenset]:
    """Utility, pairs served and sites; an empty candidate list means serve nothing."""
    if result.status == INFEASIBLE:
        return 0.0, 0, frozenset()
    return result.utility, result.served_pairs, frozenset(result.repeaters)


def solve_one(topo, paths, hardware_key, rate_model, coherence_model, budget, memories):
    config = NetworkConfig(repeater_memories=memories, endnode_memories=memories,
                           max_repeaters=budget, require_all_pairs=False,
                           use_demand_weights=False, rate_model=rate_model,
                           coherence_model=coherence_model)
    started = time.perf_counter()
    result = solve_placement(topo, paths, HARDWARE[hardware_key], config,
                             time_limit_s=TIME_LIMIT_S, mip_gap=0.0)
    return result, time.perf_counter() - started


def run_network(seed: int, settings_campaign: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    family = "ladder" if seed % 3 == 0 else "random"
    topo, max_link = ladder_network(rng) if family == "ladder" else random_network(rng)
    pairs = topo.demand_pairs()
    budget = int(rng.integers(1, 5))
    memories = int(rng.choice([4, 8, 16]))
    base = {"seed": seed, "family": family, "n_nodes": topo.graph.number_of_nodes(),
            "n_pairs": len(pairs), "budget": budget, "memories": memories}

    try:
        full_paths = enumerate_paths(topo, pairs, max_link_km=max_link, max_hops=MAX_HOPS,
                                     strategy="exhaustive", max_exhaustive=MAX_EXHAUSTIVE)
    except ValueError:
        return [{**base, "campaign": "skipped", "note": "too many simple paths"}]
    n_full = sum(len(v) for v in full_paths.values())

    combos = SETTINGS_COMBOS if settings_campaign else RECOVERY_COMBOS
    settings = ([dict(zip(("tie_cap", "hop_slack", "k_simple"), s)) for s in SETTINGS_GRID]
                if settings_campaign else [DEFAULTS])

    rows = []
    for hardware_key, rate_model, coherence_model in combos:
        full, full_s = solve_one(topo, full_paths, hardware_key, rate_model, coherence_model,
                                 budget, memories)
        full_u, full_served, full_sites = objective(full)
        for setting in settings:
            started = time.perf_counter()
            paths = enumerate_paths(topo, pairs, max_link_km=max_link, max_hops=MAX_HOPS,
                                    strategy="candidates", **setting)
            generate_s = time.perf_counter() - started
            reduced, reduced_s = solve_one(topo, paths, hardware_key, rate_model,
                                           coherence_model, budget, memories)
            red_u, red_served, red_sites = objective(reduced)
            resolved = (full.status in (OPTIMAL, INFEASIBLE)
                        and reduced.status in (OPTIMAL, INFEASIBLE))
            rows.append({
                **base, **setting,
                "campaign": "settings" if settings_campaign else "recovery",
                "hardware": hardware_key, "rate_model": rate_model,
                "coherence_model": coherence_model,
                "full_paths": n_full, "candidate_paths": sum(len(v) for v in paths.values()),
                "full_status": full.status, "candidate_status": reduced.status,
                "resolved": resolved,
                "full_utility": full_u, "candidate_utility": red_u,
                "gap_bits": full_u - red_u,
                "recovered": resolved and red_u >= full_u - 1e-6 * max(1.0, abs(full_u)),
                "full_served": full_served, "candidate_served": red_served,
                "served_gap": full_served - red_served,
                "same_sites": full_sites == red_sites,
                "full_solve_s": full_s, "candidate_solve_s": reduced_s,
                "generate_s": generate_s,
            })
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def recovery_report(frame: pd.DataFrame) -> None:
    usable = frame[frame["resolved"]]
    say(f"   {len(frame)} instances, {len(frame) - len(usable)} left out for an unresolved solve")
    say(f"   {'hardware':<8} {'rate':<11} {'coherence':<17} {'n':>5} {'recovered':>10}"
        f" {'misses':>6} {'gap bits: median':>17} {'max':>7} {'served gap max':>15}"
        f" {'paths: cand/full':>17}")
    for (hw, rate, coh), g in usable.groupby(["hardware", "rate_model", "coherence_model"],
                                              sort=False):
        misses = g[~g["recovered"]]
        med = misses["gap_bits"].median() if len(misses) else 0.0
        mx = misses["gap_bits"].max() if len(misses) else 0.0
        ratio = (g["candidate_paths"] / g["full_paths"]).median()
        say(f"   {hw:<8} {rate:<11} {coh:<17} {len(g):>5} {100 * g['recovered'].mean():>9.1f}%"
            f" {len(misses):>6} {med:>17.4f} {mx:>7.4f} {int(g['served_gap'].max()):>15}"
            f" {ratio:>17.2f}")
    for family, g in usable.groupby("family"):
        say(f"   {family:<7} family: {100 * g['recovered'].mean():.1f}% recovered over {len(g)} instances")
    say(f"   overall: {100 * usable['recovered'].mean():.2f}% of {len(usable)} instances;"
        f" largest gap {usable['gap_bits'].max():.4f} bits;"
        f" median path generation {1e3 * usable['generate_s'].median():.1f} ms")


def settings_report(frame: pd.DataFrame) -> None:
    usable = frame[frame["resolved"]]
    say(f"   {len(frame)} instances, {len(frame) - len(usable)} left out for an unresolved solve")
    say(f"   {'tie_cap':>7} {'hop_slack':>9} {'k_simple':>8} | {'recovered':>9} {'max gap bits':>12}"
        f" {'max served gap':>14} | {'mean paths':>10} {'mean gen ms':>11}")
    table = (usable.groupby(["tie_cap", "hop_slack", "k_simple"])
             .agg(recovered=("recovered", "mean"), max_gap=("gap_bits", "max"),
                  max_served=("served_gap", "max"), paths=("candidate_paths", "mean"),
                  gen=("generate_s", "mean"))
             .reset_index())
    for _, r in table.iterrows():
        mark = "  <- default" if (r["tie_cap"], r["hop_slack"], r["k_simple"]) == (4, 1, 6) else ""
        say(f"   {int(r['tie_cap']):>7} {int(r['hop_slack']):>9} {int(r['k_simple']):>8} |"
            f" {100 * r['recovered']:>8.1f}% {r['max_gap']:>12.4f} {int(r['max_served']):>14} |"
            f" {r['paths']:>10.1f} {1e3 * r['gen']:>11.1f}{mark}")
    perfect = table[table["recovered"] >= 1.0]
    if len(perfect):
        cheapest = perfect.sort_values("paths").iloc[0]
        say(f"   smallest setting that recovered every instance: tie_cap {int(cheapest['tie_cap'])},"
            f" hop_slack {int(cheapest['hop_slack'])}, k_simple {int(cheapest['k_simple'])},"
            f" {cheapest['paths']:.1f} paths on average")
    else:
        say("   no setting recovered every instance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", type=int, default=400)
    parser.add_argument("--settings-networks", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    from joblib import Parallel, delayed

    say("=" * 78)
    say("W7: HOW OFTEN THE DEFAULT CANDIDATE SET REACHES THE FULL-PATH OPTIMUM")
    say("=" * 78)
    say(f"   hop limit {MAX_HOPS}, exhaustive enumeration capped at {MAX_EXHAUSTIVE} paths per pair,"
        f" exact solves (mip_gap 0)")

    for campaign, n, settings_flag in [("recovery", args.networks, False),
                                        ("settings", args.settings_networks, True)]:
        seeds = [args.seed + (0 if campaign == "recovery" else 1_000_000) + i for i in range(n)]
        say(f"\n-- {campaign}: {n} networks")
        started = time.time()
        out = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(run_network)(s, settings_flag) for s in seeds
        )
        frame = pd.DataFrame.from_records([row for rows in out for row in rows])
        skipped = int((frame["campaign"] == "skipped").sum())
        frame = frame[frame["campaign"] != "skipped"]
        say(f"   done in {(time.time() - started) / 60:.1f} min; {skipped} networks skipped"
            f" for exceeding the enumeration cap")
        save_frame(frame, f"w7_candidate_{campaign}.csv")
        (recovery_report if campaign == "recovery" else settings_report)(frame)

    say("\n" + "=" * 78)
    (RESULTS / "w7_candidate_recovery.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("   wrote results/w7_candidate_recovery.log")


if __name__ == "__main__":
    main()
