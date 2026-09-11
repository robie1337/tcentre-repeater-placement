"""Waiting time as gradual decoherence, at the measured nuclear coherence.

W5 gated paths on a hard cut-off, mean storage time <= T2, and tested it at
the preset coherence times (10 ms midrange, 100 ms projected). The measured
hydrogen nuclear echo T2 is 112 ms (Song et al. 2025), above the sweep box,
and a hard cut-off hides the fidelity a pair loses while it waits well inside
T2. This script repeats the comparison at T2 = 112 ms with decoherence
modelled as gradual decay.

Decay model
-----------
Link e is ready after an exponential time T_e with rate
lambda_e = -ln(1 - P_e) / tau_e, where P_e = 1 - (1 - p_e)^W and the round
lasts tau_e = max(1 / generation rate, 2 l_e / c) (heralded clock). Until the
last link is ready, the pair on link e sits in two memories for
max_k T_k - T_e. Each stored half decays the Werner parameter by
exp(-t / T2), so

  mu_e2e = s^(h-1) mu_L^h * E[ exp(-2 * sum_e (T_max - T_e) / T2) ] * exp(-tau_e2e / T2)

with the expectation estimated from shared exponential samples. Fidelity is
(3 mu_e2e + 1) / 4, rates are Eq. (2) unchanged, and a path whose fidelity
falls to 1/2 is unusable. This is a simple memory model: exponential decay
at the idle echo T2, which is optimistic if the memory dephases faster while
the electron keeps attempting.

Plans, all under the paper's objective and Eq. (2) rates:
  published   the paper's gates only
  hard_gate   plus mean storage time <= T2 (W5's primary gate)
  decay       fidelity reduced by the decay above, no hard gate

Run:  python scripts/w6_memory_decay.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
import qrp.model as model_module
from qrp import physics
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import NetworkConfig
from qrp.paths import enumerate_paths, path_statistics
from qrp.solver import solve
from qrp.topology import build_ca9

UNLIMITED = 10**6
C_FIBRE = physics.C_FIBRE_KM_PER_S
N_SAMPLES = 4000
MAX_HOPS = 20
_ORIGINAL_BUILD = model_module.build_candidates
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def link_rates(path, hardware, width) -> np.ndarray:
    lams = []
    for length in path.link_lengths_km:
        p = physics.link_success(length, hardware.alpha_db_per_km, hardware.eta_emission,
                                 hardware.eta_detection)
        P = physics.link_success_multiplexed(p, width)
        tau = max(1.0 / hardware.generation_rate_hz, 2.0 * length / C_FIBRE)
        lams.append(-math.log1p(-P) / tau if P < 1.0 else 1e15)
    return np.asarray(lams)


def memory_tables(paths, hardware, widths, samples):
    """(path nodes, width) -> (decay factor on mu, mean storage time in seconds)."""
    t2 = hardware.t_repeater_memory_s
    decay, decay_opt, storage = {}, {}, {}
    for pair_paths in paths.values():
        for path in pair_paths:
            h = path.hops
            tau_e2e = physics.tau_e2e(path.link_lengths_km)
            for w in widths:
                times = samples[:, :h] / link_rates(path, hardware, w)
                t_max = times.max(axis=1)
                t_min = times.min(axis=1)
                held = h * t_max - times.sum(axis=1)
                tail = math.exp(-tau_e2e / t2)
                # Pessimistic: every link's pair idles until the last link is ready.
                decay[(path.nodes, w)] = float(np.mean(np.exp(-2.0 * held / t2))) * tail
                # Optimistic: only one pair idles, for the longest gap. Any
                # swap schedule sits between the two.
                decay_opt[(path.nodes, w)] = float(np.mean(np.exp(-2.0 * (t_max - t_min) / t2))) * tail
                storage[(path.nodes, w)] = float(np.mean(t_max - t_min)) + tau_e2e
    return decay, decay_opt, storage


def decayed(candidate, factor):
    """Candidate with fidelity and utility reduced by the storage decay, or None."""
    mu = (4.0 * candidate.fidelity - 1.0) / 3.0 * factor
    fidelity = (3.0 * mu + 1.0) / 4.0
    utility = physics.utility(candidate.rate, fidelity)
    if not math.isfinite(utility):
        return None
    return dataclasses.replace(candidate, fidelity=fidelity, utility=utility)


def solve_mode(topo, paths, hardware, budget, mode, decay, storage):
    config = NetworkConfig(repeater_memories=100, endnode_memories=100, max_repeaters=budget,
                           require_all_pairs=False, use_demand_weights=False, rate_model="paper")
    t2 = hardware.t_repeater_memory_s

    def patched(paths_, hardware_, config_, demand_weight=None):
        out = []
        for cand in _ORIGINAL_BUILD(paths_, hardware_, config_, demand_weight=demand_weight):
            key = (cand.path.nodes, cand.width)
            if mode == "hard_gate":
                if storage[key] <= t2:
                    out.append(cand)
            elif mode == "decay":
                new = decayed(cand, decay[key])
                if new is not None:
                    out.append(new)
            else:
                out.append(cand)
        return out

    model_module.build_candidates = patched
    try:
        built = model_module.build_model(topo, paths, hardware, config)
    finally:
        model_module.build_candidates = _ORIGINAL_BUILD
    if built is None:
        return []
    # The default candidate set makes the budgeted models several times larger
    # than the legacy ones; 120 s was not always enough to prove optimality.
    result = solve(built.problem, backend="highs", time_limit_s=900.0, mip_gap=1e-6)
    if not result.optimal:
        raise RuntimeError(f"{mode} budget {budget}: {result.status}")
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
    samples = np.random.default_rng(20260911).exponential(size=(N_SAMPLES, MAX_HOPS))

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
        widths = NetworkConfig(repeater_memories=100, endnode_memories=100).widths()
        decay, decay_opt, storage = memory_tables(paths, hardware, widths, samples)
        t2 = hardware.t_repeater_memory_s
        say(f"\n-- {label}")
        for budget in budgets:
            pub = solve_mode(topo, paths, hardware, budget, "published", decay, storage)
            gate = solve_mode(topo, paths, hardware, budget, "hard_gate", decay, storage)
            dec = solve_mode(topo, paths, hardware, budget, "decay", decay, storage)
            opt = solve_mode(topo, paths, hardware, budget, "decay", decay_opt, storage)

            opt_broken, opt_f = 0, []
            for c in pub:
                new = decayed(c, decay_opt[(c.path.nodes, c.width)])
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
                new = decayed(c, decay[key])
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
