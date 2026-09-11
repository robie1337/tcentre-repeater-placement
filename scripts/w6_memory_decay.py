"""Waiting time as gradual decoherence, at the measured nuclear coherence.

W5 gated paths on a hard cut-off, mean storage time <= T2, and tested it at
the preset coherence times (10 ms midrange, 100 ms projected). The measured
hydrogen nuclear echo T2 is 112 ms (Song et al. 2025), above the sweep box,
and a hard cut-off hides the fidelity a pair loses while it waits well inside
T2. This script repeats the comparison at T2 = 112 ms with decoherence
modelled as gradual decay.

Decay model
-----------
The formulas live in qrp.waiting and the plans come from the model itself
through ``NetworkConfig.coherence_model``. Link e is ready after an
exponential time with rate lambda_e = -ln(1 - P_e) / tau_e, where
P_e = 1 - (1 - p_e)^W and the round lasts tau_e = max(1 / generation rate,
2 l_e / c) (heralded clock). Each stored half of a pair decays the Werner
parameter by exp(-t / T2), and the expectation over link times uses shared
exponential samples. Fidelity is (3 mu + 1) / 4, rates are Eq. (2) unchanged,
and a path whose fidelity falls to 1/2 is unusable. This uses the idle echo
T2, which is optimistic if the memory dephases faster while the electron
keeps attempting.

Plans, all under the paper's objective and Eq. (2) rates:
  published   coherence_model="paper"
  hard_gate   coherence_model="waiting_gate", mean storage time <= T2
  decay       coherence_model="decay_pessimistic", every pair waits
  opt         coherence_model="decay_optimistic", one pair waits

Until 11 September 2026 the hard gate here used a Monte Carlo estimate of the
storage time; it now uses the model's analytic one, which W5 also uses.

Run:  python scripts/w6_memory_decay.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp import physics, waiting
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import DECAY_SAMPLE_COLUMNS, NetworkConfig, build_model
from qrp.paths import enumerate_paths, path_statistics
from qrp.solver import solve
from qrp.topology import build_ca9

UNLIMITED = 10**6
MAX_HOPS = 20
TIME_LIMIT_S = 900.0
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def config_for(budget: int, coherence_model: str) -> NetworkConfig:
    return NetworkConfig(repeater_memories=100, endnode_memories=100, max_repeaters=budget,
                         require_all_pairs=False, use_demand_weights=False, rate_model="paper",
                         coherence_model=coherence_model)


def decay_tables(paths, hardware, config):
    """(path nodes, width) -> decay factor per bound, and storage time in seconds.

    Used to score the published plan under decay, including the paths decay
    would remove, which the model's own candidate list no longer carries.
    """
    longest = max(p.hops for pair_paths in paths.values() for p in pair_paths)
    samples = waiting.unit_exponential_samples(config.decay_samples,
                                               max(DECAY_SAMPLE_COLUMNS, longest),
                                               config.decay_seed)
    t2 = min(hardware.t_repeater_memory_s, hardware.t_endnode_memory_s)
    factors = {bound: {} for bound in waiting.DECAY_BOUNDS}
    storage = {}
    for pair_paths in paths.values():
        for path in pair_paths:
            probs = [physics.link_success(length, hardware.alpha_db_per_km, hardware.eta_emission,
                                          hardware.eta_detection)
                     for length in path.link_lengths_km]
            tau = physics.tau_e2e(path.link_lengths_km)
            for width in config.widths():
                rates = waiting.link_ready_rates(path.link_lengths_km, probs, width,
                                                 hardware.generation_rate_hz, config.waiting_clock)
                key = (path.nodes, width)
                for bound in waiting.DECAY_BOUNDS:
                    factors[bound][key] = waiting.storage_decay_factor(rates, tau, t2, samples, bound)
                storage[key] = waiting.expected_storage_time(rates, tau)
    return factors, storage


def rescored(candidate, factor):
    """Candidate with fidelity and utility reduced by the storage decay, or None."""
    fidelity = waiting.decayed_fidelity(candidate.fidelity, factor)
    utility = physics.utility(candidate.rate, fidelity)
    if not math.isfinite(utility):
        return None
    return dataclasses.replace(candidate, fidelity=fidelity, utility=utility)


def solve_mode(topo, paths, hardware, budget, coherence_model):
    built = build_model(topo, paths, hardware, config_for(budget, coherence_model))
    if built is None:
        return []
    result = solve(built.problem, backend="highs", time_limit_s=TIME_LIMIT_S, mip_gap=1e-6)
    if not result.optimal:
        raise RuntimeError(f"{coherence_model} budget {budget}: {result.status}")
    return [built.candidates[j] for j in range(built.n_candidates) if result.x[j] > 0.5]


def repeaters(chosen) -> set[str]:
    return {node for c in chosen for node in c.path.repeaters}


def main() -> None:
    say("=" * 78)
    say("W6: WAITING TIME AS GRADUAL DECAY, AT THE MEASURED NUCLEAR T2")
    say("=" * 78)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-strategy", default="candidates", choices=["candidates", "legacy"])
    args = parser.parse_args()
    suffix = "" if args.path_strategy == "candidates" else f"_{args.path_strategy}"

    topo = build_ca9(spacing_km=80.0)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=MAX_HOPS,
                            strategy=args.path_strategy)
    say(f"   candidate paths ({args.path_strategy}): {path_statistics(paths)}")

    def with_t2(hw, t2):
        return hw.with_(t_repeater_memory_s=t2, t_endnode_memory_s=t2)

    points = [
        ("midrange, T2 10 ms (preset)", TCENTRE_MIDRANGE),
        ("midrange, T2 112 ms (measured)", with_t2(TCENTRE_MIDRANGE, 0.112)),
        ("projected, T2 100 ms (preset)", TCENTRE_PROJECTED),
        ("projected, T2 112 ms (measured)", with_t2(TCENTRE_PROJECTED, 0.112)),
    ]
    budgets = [UNLIMITED, 20, 10]

    rows = []
    for label, hardware in points:
        factors, storage = decay_tables(paths, hardware, config_for(UNLIMITED, "paper"))
        decay, decay_opt = factors["pessimistic"], factors["optimistic"]
        t2 = min(hardware.t_repeater_memory_s, hardware.t_endnode_memory_s)
        say(f"\n-- {label}")
        for budget in budgets:
            pub = solve_mode(topo, paths, hardware, budget, "paper")
            gate = solve_mode(topo, paths, hardware, budget, "waiting_gate")
            dec = solve_mode(topo, paths, hardware, budget, "decay_pessimistic")
            opt = solve_mode(topo, paths, hardware, budget, "decay_optimistic")

            opt_broken, opt_f = 0, []
            for c in pub:
                new = rescored(c, decay_opt[(c.path.nodes, c.width)])
                if new is None:
                    opt_broken += 1
                    opt_f.append(0.5)
                else:
                    opt_f.append(new.fidelity)

            # The published plan re-scored with decay.
            pub_f, pub_f_decayed, broken, pub_u_decay = [], [], 0, 0.0
            unreachable = 0
            for c in pub:
                key = (c.path.nodes, c.width)
                unreachable += storage[key] > t2
                new = rescored(c, decay[key])
                pub_f.append(c.fidelity)
                if new is None:
                    broken += 1
                    pub_f_decayed.append(0.5)
                else:
                    pub_f_decayed.append(new.fidelity)
                    pub_u_decay += new.utility
            dec_u = sum(c.utility for c in dec)
            pub_pairs = {c.pair: c for c in pub}
            dec_pairs = {c.pair: c for c in dec}
            common = pub_pairs.keys() & dec_pairs.keys()
            rerouted = sum(pub_pairs[p].path.nodes != dec_pairs[p].path.nodes for p in common)
            rewidthed = sum(pub_pairs[p].width != dec_pairs[p].width for p in common)
            store_ratio = [storage[(c.path.nodes, c.width)] / t2 for c in pub]
            regret = (dec_u - pub_u_decay) / len(dec) if dec else 0.0

            bl = "unlimited" if budget >= UNLIMITED else str(budget)
            say(f"   budget {bl:>9} | published {len(pub):>2} pairs {len(repeaters(pub)):>2} repeaters,"
                f" median storage/T2 {np.median(store_ratio) if store_ratio else float('nan'):.2f}")
            say(f"     hard gate : {unreachable:>2} published pairs over T2 | plan {len(gate):>2} pairs"
                f" {len(repeaters(gate)):>2} repeaters")
            say(f"     decay, one pair waits (optimistic): published mean F"
                f" -> {np.mean(opt_f) if opt_f else float('nan'):.3f}, {opt_broken} pairs fall to F <= 1/2"
                f" | plan {len(opt):>2} pairs {len(repeaters(opt)):>2} repeaters")
            say(f"     decay, all pairs wait (pessimistic): published plan mean F {np.mean(pub_f) if pub_f else float('nan'):.3f}"
                f" -> {np.mean(pub_f_decayed) if pub_f_decayed else float('nan'):.3f},"
                f" {broken} pairs fall to F <= 1/2 | decay-aware plan {len(dec):>2} pairs"
                f" {len(repeaters(dec)):>2} repeaters, {rerouted} rerouted, {rewidthed} re-widthed,"
                f" regret {regret:.2f} bits/pair")
            rows.append({
                "case": label, "t2_ms": 1e3 * t2, "budget": budget,
                "published_pairs": len(pub), "published_repeaters": len(repeaters(pub)),
                "published_median_storage_over_t2": float(np.median(store_ratio)) if store_ratio else float("nan"),
                "hard_gate_unreachable": unreachable, "hard_gate_pairs": len(gate),
                "hard_gate_repeaters": len(repeaters(gate)),
                "published_mean_F": float(np.mean(pub_f)) if pub_f else float("nan"),
                "published_mean_F_decayed": float(np.mean(pub_f_decayed)) if pub_f_decayed else float("nan"),
                "published_pairs_broken_by_decay": broken,
                "decay_pairs": len(dec), "decay_repeaters": len(repeaters(dec)),
                "decay_rerouted": rerouted, "decay_rewidthed": rewidthed,
                "decay_regret_bits_per_pair": regret,
                "opt_published_mean_F_decayed": float(np.mean(opt_f)) if opt_f else float("nan"),
                "opt_published_pairs_broken": opt_broken,
                "opt_pairs": len(opt), "opt_repeaters": len(repeaters(opt)),
            })

    save_frame(pd.DataFrame(rows), f"w6_memory_decay{suffix}.csv")
    say("\n" + "=" * 78)
    (RESULTS / f"w6_memory_decay{suffix}.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"   wrote results/w6_memory_decay{suffix}.log")


if __name__ == "__main__":
    main()
