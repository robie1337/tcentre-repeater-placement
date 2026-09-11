"""Is the closed-form swap-ASAP decay right, and what do failed swaps do to it?

``qrp.waiting.swap_asap_qubit_time`` gives the exact memory storage time of
swap-as-soon-as-possible when every swap succeeds and each link's time to
success is exponential. Two things the closed form leaves out are checked
here against a direct event simulation of the protocol:

  geometric rounds   links succeed on a whole number of attempt rounds rather
                     than at an exponential time with the same mean rate
  failed swaps       a swap succeeds with probability q; on failure both input
                     pairs are destroyed and every link they covered restarts,
                     while pairs elsewhere on the path keep waiting

Simulation: a ready link is a pair covering one segment. When two adjacent
segments both hold a pair, the node between them swaps at once. Each stored
qubit multiplies the Werner parameter by exp(-t / T2), and a swap multiplies
the Werner parameters of its inputs, so decay exponents add. The first pair
covering the whole path is delivered, and its decay factor is recorded.

Routes: at each hop count, the CA9 candidate path with the shortest worst
link, at memory width 100, projected hardware, T2 = 112 ms, heralded clock.
The classical delay after the last swap is left out of both sides.

Run:  python scripts/w12_swap_event_check.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from _common import RESULTS, save_frame
from qrp import physics, waiting
from qrp.hardware import TCENTRE_PROJECTED
from qrp.paths import enumerate_paths
from qrp.topology import build_ca9

T2_S = 0.112
WIDTH = 100
HOPS = (4, 8, 12, 16, 19)
#: (label, swap success, geometric rounds, samples)
VARIANTS = (("q=1 exponential", 1.0, False, 4000), ("q=1 geometric", 1.0, True, 4000),
            ("q=0.95", 0.95, False, 2000), ("q=0.7", 0.7, False, 600))
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def ready_time(rng, lam, per_round, tau, geometric):
    if geometric:
        return tau * rng.geometric(per_round) if per_round < 1.0 else tau
    return rng.exponential(1.0 / lam)


def simulate(lams, per_round, taus, q, rng, geometric):
    """One delivery. Returns the decay exponent of the delivered pair."""
    h = len(lams)
    ready = [ready_time(rng, lams[e], per_round[e], taus[e], geometric) for e in range(h)]
    segments = {}  # start link -> [end link, decay exponent, time of last update]
    pending = set(range(h))
    while True:
        e = min(pending, key=lambda k: ready[k])
        t = ready[e]
        pending.discard(e)
        segments[e] = [e, 0.0, t]
        start = e
        while True:
            end = segments[start][0]
            if end + 1 in segments:
                s1, s2 = start, end + 1
            else:
                left = next((s for s, v in segments.items() if v[0] == start - 1), None)
                if left is None:
                    break
                s1, s2 = left, start
            v1, v2 = segments.pop(s1), segments.pop(s2)
            d1 = v1[1] + 2.0 * (t - v1[2]) / T2_S
            d2 = v2[1] + 2.0 * (t - v2[2]) / T2_S
            if rng.random() < q:
                segments[s1] = [v2[0], d1 + d2, t]
                start = s1
            else:
                for link in range(s1, v2[0] + 1):
                    ready[link] = t + ready_time(rng, lams[link], per_round[link], taus[link], geometric)
                    pending.add(link)
                break
        if 0 in segments and segments[0][0] == h - 1:
            return segments[0][1]


def main() -> None:
    say("=" * 78)
    say("W12: CLOSED-FORM SWAP-ASAP DECAY AGAINST AN EVENT SIMULATION")
    say("=" * 78)
    hw = TCENTRE_PROJECTED
    topo = build_ca9(spacing_km=80.0)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=20)
    by_hops = {}
    for pair_paths in paths.values():
        for p in pair_paths:
            if p.hops not in by_hops or p.max_link_km < by_hops[p.hops].max_link_km:
                by_hops[p.hops] = p
    samples = waiting.unit_exponential_samples(20000, 20, 11)
    rng = np.random.default_rng(7)

    say(f"   projected hardware, T2 {1e3 * T2_S:.0f} ms, width {WIDTH}, heralded clock;"
        " Werner factor of the delivered pair, mean +- standard error")
    say(f"   {'hops':>4} {'worst km':>8} | {'closed form':>11} | " +
        " | ".join(f"{label:>17}" for label, *_ in VARIANTS))
    rows = []
    for hops in HOPS:
        if hops not in by_hops:
            continue
        path = by_hops[hops]
        probs = [physics.link_success(length, hw.alpha_db_per_km, hw.eta_emission, hw.eta_detection)
                 for length in path.link_lengths_km]
        per_round = [physics.link_success_multiplexed(p, WIDTH) for p in probs]
        taus = [max(1.0 / hw.generation_rate_hz, 2.0 * length / physics.C_FIBRE_KM_PER_S)
                for length in path.link_lengths_km]
        lams = waiting.link_ready_rates(path.link_lengths_km, probs, WIDTH, hw.generation_rate_hz,
                                        "heralded")
        closed = waiting.storage_decay_factor(lams, 0.0, T2_S, samples, "swap_asap")
        cells = []
        row = {"hops": hops, "worst_link_km": path.max_link_km, "closed_form": closed}
        for label, q, geometric, n in VARIANTS:
            factors = np.exp(-np.array([simulate(lams, per_round, taus, q, rng, geometric)
                                        for _ in range(n)]))
            mean, se = float(factors.mean()), float(factors.std() / math.sqrt(n))
            row[f"{label} mean"], row[f"{label} se"] = mean, se
            cells.append(f"{mean:.4f} +- {se:.4f}")
        rows.append(row)
        say(f"   {hops:>4} {path.max_link_km:>8.0f} | {closed:>11.4f} | " +
            " | ".join(f"{c:>17}" for c in cells))

    frame = pd.DataFrame(rows)
    save_frame(frame, "w12_swap_event_check.csv")
    worst_z = max(abs(r["closed_form"] - r["q=1 exponential mean"]) / r["q=1 exponential se"] for r in rows)
    geo = max(abs(r["q=1 geometric mean"] / r["closed_form"] - 1) for r in rows)
    q95 = max(1 - r["q=0.95 mean"] / r["closed_form"] for r in rows)
    q70 = max(1 - r["q=0.7 mean"] / r["closed_form"] for r in rows)
    say(f"\n   closed form against the deterministic simulation: at most {worst_z:.1f} standard errors apart")
    say(f"   geometric rounds change the factor by at most {100 * geo:.1f} per cent")
    say(f"   failed swaps lower it by at most {100 * q95:.1f} per cent at q = 0.95"
        f" and {100 * q70:.1f} per cent at q = 0.7")
    say("\n" + "=" * 78)
    (RESULTS / "w12_swap_event_check.log").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("   wrote results/w12_swap_event_check.log")


if __name__ == "__main__":
    main()
