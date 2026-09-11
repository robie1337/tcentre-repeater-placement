"""Does the attempt clock or the swap schedule change the waiting-time result?

W6 found that memory decay while links wait costs the published plan much of
its fidelity at the measured nuclear T2 of 112 ms. That result rests on two
modelling choices this script varies:

  clock      how long an attempt round lasts (qrp.waiting.link_ready_rates)
             heralded            max(1 / rate, 2 l / c), herald from the far node
             heralded_midpoint   max(1 / rate, l / c), herald from a midpoint station
             source              1 / rate, no wait for the herald
  schedule   how long qubits are stored (qrp.waiting.storage_decay_factor)
             optimistic          one pair waits from the first link ready to the last
             swap_asap           exact swap-as-soon-as-possible, deterministic swaps
             pessimistic         every pair waits for the last link

For the midrange and projected presets at T2 = 112 ms, unlimited budget, it
reports the published plan's mean fidelity after decay, the pairs decay pushes
to F <= 1/2, and the pairs a decay-aware plan serves, for every clock and
schedule.

A single communication qubit cannot start its next attempt before the herald
of the last one arrives, and the T centre demonstration (Afzal et al.,
arXiv:2406.01704) runs one herald per attempt, so the heralded clocks are the
physical ones; source is shown to mark where the effect comes from.

Two approximations checked in W8 can be loosened here: ``--width-grid all``
offers every width from 1 to 100 instead of eight, and ``--max-hops`` raises
the hop cap. Both make the models much larger, so pair them with
``--backend gurobi``. Output files carry a suffix when either is not the
default.

Run:  python scripts/w11_clock_schedule.py [--backend gurobi] [--width-grid all] [--max-hops 24]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp import physics, waiting
from qrp.hardware import TCENTRE_MIDRANGE, TCENTRE_PROJECTED
from qrp.model import DECAY_SAMPLE_COLUMNS, NetworkConfig, build_model
from qrp.paths import enumerate_paths, path_statistics
from qrp.solver import solve
from qrp.topology import build_ca9

T2_S = 0.112
TIME_LIMIT_S = 900.0
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def plan(topo, paths, hardware, width_grid, backend, coherence_model="paper", clock="heralded"):
    config = NetworkConfig(require_all_pairs=False, coherence_model=coherence_model,
                           waiting_clock=clock, width_grid=width_grid)
    built = build_model(topo, paths, hardware, config)
    if built is None:
        return []
    result = solve(built.problem, backend=backend, time_limit_s=TIME_LIMIT_S, mip_gap=1e-6)
    if not result.optimal:
        raise RuntimeError(f"{coherence_model} {clock}: {result.status}")
    return [built.candidates[j] for j in range(built.n_candidates) if result.x[j] > 0.5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="highs", choices=["highs", "gurobi", "cplex"])
    parser.add_argument("--width-grid", default="log8", choices=["log8", "all"])
    parser.add_argument("--max-hops", type=int, default=20)
    args = parser.parse_args()
    width_grid = None if args.width_grid == "log8" else tuple(range(1, 101))
    suffix = ""
    if args.width_grid != "log8":
        suffix += f"_widths_{args.width_grid}"
    if args.max_hops != 20:
        suffix += f"_hops{args.max_hops}"

    say("=" * 78)
    say("W11: DOES THE ATTEMPT CLOCK OR THE SWAP SCHEDULE CHANGE THE RESULT?")
    say("=" * 78)
    say(f"   backend {args.backend}, width grid {args.width_grid}, max hops {args.max_hops}")
    topo = build_ca9(spacing_km=80.0)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=args.max_hops)
    say(f"   candidate paths: {path_statistics(paths)}")
    config = NetworkConfig()
    longest = max(p.hops for pair_paths in paths.values() for p in pair_paths)
    samples = waiting.unit_exponential_samples(config.decay_samples,
                                               max(DECAY_SAMPLE_COLUMNS, longest), config.decay_seed)

    rows = []
    for name, base in (("midrange", TCENTRE_MIDRANGE), ("projected", TCENTRE_PROJECTED)):
        hardware = base.with_(t_repeater_memory_s=T2_S, t_endnode_memory_s=T2_S)
        published = plan(topo, paths, hardware, width_grid, args.backend)
        repeaters = {n for c in published for n in c.path.repeaters}
        mean_f = float(np.mean([c.fidelity for c in published]))
        say(f"\n-- {name}, T2 {1e3 * T2_S:.0f} ms: published plan {len(published)} pairs,"
            f" {len(repeaters)} repeaters, mean F {mean_f:.3f}")
        say(f"   {'clock':<18} | {'schedule':<11} | {'mean F':>6} {'F <= 1/2':>8} | decay-aware plan")
        for clock in waiting.WAITING_CLOCKS:
            for bound in waiting.DECAY_BOUNDS:
                fidelities, broken = [], 0
                for c in published:
                    probs = [physics.link_success(length, hardware.alpha_db_per_km,
                                                  hardware.eta_emission, hardware.eta_detection)
                             for length in c.path.link_lengths_km]
                    rates = waiting.link_ready_rates(c.path.link_lengths_km, probs, c.width,
                                                     hardware.generation_rate_hz, clock)
                    factor = waiting.storage_decay_factor(rates, c.tau_e2e_s, T2_S, samples, bound)
                    f = waiting.decayed_fidelity(c.fidelity, factor)
                    if f <= 0.5:
                        broken += 1
                        f = 0.5
                    fidelities.append(f)
                aware = plan(topo, paths, hardware, width_grid, args.backend, f"decay_{bound}", clock)
                aware_reps = {n for c in aware for n in c.path.repeaters}
                decayed_mean = float(np.mean(fidelities))
                say(f"   {clock:<18} | {bound:<11} | {decayed_mean:>6.3f} {broken:>8} |"
                    f" {len(aware):>2} pairs, {len(aware_reps):>2} repeaters")
                rows.append({"hardware": name, "t2_ms": 1e3 * T2_S, "clock": clock, "schedule": bound,
                             "width_grid": args.width_grid, "max_hops": args.max_hops,
                             "backend": args.backend,
                             "published_pairs": len(published), "published_mean_F": mean_f,
                             "published_mean_F_decayed": decayed_mean,
                             "published_pairs_broken": broken,
                             "aware_pairs": len(aware), "aware_repeaters": len(aware_reps)})

    save_frame(pd.DataFrame(rows), f"w11_clock_schedule{suffix}.csv")
    say("\n" + "=" * 78)
    (RESULTS / f"w11_clock_schedule{suffix}.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"   wrote results/w11_clock_schedule{suffix}.log")


if __name__ == "__main__":
    main()
