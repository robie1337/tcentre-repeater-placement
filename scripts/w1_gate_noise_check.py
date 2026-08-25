"""Week 1: how much does the gate and measurement fidelity choice matter?

Eq. (5) carries a per-swap factor P_2 (4 eta^2 - 1) / 3. Pouryousef sets the
gate fidelity P_2 and measurement fidelity eta to approximately 1 and says so.
The T centre has measured values for both, and substituting them changes the
answer far more than any of the four parameters this project claims to sweep.

The presets in hardware.py keep both at 1.0, so that the only differences
from the paper's baseline are attenuation, link fidelity, swap success and
coherence. This script measures what that choice is worth, which is the
honest way to report an assumption that consequential.

Run:  python scripts/w1_gate_noise_check.py
"""

from __future__ import annotations

import pandas as pd

from _common import banner, save_frame, step
from qrp import physics, topology
from qrp.hardware import (
    GATE_NOISE_MEASURED,
    TCENTRE_MIDRANGE,
    TCENTRE_PROJECTED,
)
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths

CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=False,
    use_demand_weights=False,
)


def max_hops_before_floor(link_fidelity, gate, meas, limit=200):
    """Longest path whose fidelity still clears the classical floor of 1/2."""
    best = 0
    for hops in range(1, limit + 1):
        if physics.e2e_fidelity(link_fidelity, hops, gate, meas) > 0.5:
            best = hops
        else:
            break
    return best


def main() -> None:
    banner("WEEK 1: sensitivity to the gate and measurement fidelity assumption")

    print("Measured T centre values used in the comparison:")
    print(f"   two-qubit gate fidelity  P_2  = {GATE_NOISE_MEASURED['gate_fidelity']}"
          "   (Afzal 2024)")
    print(f"   measurement fidelity     eta  = {GATE_NOISE_MEASURED['measurement_fidelity']}"
          "   (Higginbottom 2022, single-shot electron readout)")

    swap_factor = (
        GATE_NOISE_MEASURED["gate_fidelity"]
        * (4 * GATE_NOISE_MEASURED["measurement_fidelity"] ** 2 - 1)
        / 3
    )
    print(f"   per-swap Werner factor        = {swap_factor:.4f}  (1.0 when both are perfect)")

    step("Effect on the hop budget")
    rows = []
    for label, fidelity in [("midrange F_L = 0.96", 0.96), ("projected F_L = 0.998", 0.998)]:
        perfect = max_hops_before_floor(fidelity, 1.0, 1.0)
        noisy = max_hops_before_floor(
            fidelity,
            GATE_NOISE_MEASURED["gate_fidelity"],
            GATE_NOISE_MEASURED["measurement_fidelity"],
        )
        rows.append({"case": label, "max_hops_perfect": perfect, "max_hops_measured": noisy})
        print(f"   {label:<24} {perfect:>3} hops -> {noisy:>3} hops")

    save_frame(pd.DataFrame(rows), "w1_gate_noise_hops.csv")

    step("Effect on the network result")
    topo = topology.build_ca9(spacing_km=80.0)
    pairs = topo.demand_pairs()
    paths = enumerate_paths(topo, pairs, max_link_km=300.0, max_hops=20)

    records = []
    for label, base in [("midrange", TCENTRE_MIDRANGE), ("projected", TCENTRE_PROJECTED)]:
        for assumption, overrides in [
            ("perfect gates (used in this project)", {}),
            ("measured T centre gates", GATE_NOISE_MEASURED),
        ]:
            hardware = base.with_(**overrides) if overrides else base
            result = solve_placement(topo, paths, hardware, CONFIG)
            records.append(
                {
                    "hardware": label,
                    "assumption": assumption,
                    "status": result.status,
                    "utility": result.utility,
                    "served_pairs": result.served_pairs,
                    "n_repeaters": result.n_repeaters,
                }
            )
            print(f"   {label:<10} {assumption:<38} "
                  f"served {result.served_pairs:>2}/18   utility {result.utility:8.2f}")

    frame = pd.DataFrame(records)
    save_frame(frame, "w1_gate_noise_effect.csv")

    step("What this means")
    print("   Substituting the measured gate and readout fidelities changes the")
    print("   result by more than any of the four parameters being swept. Two")
    print("   things follow.")
    print()
    print("   First, this project keeps both at 1.0, matching the paper, so that")
    print("   the comparison with published results isolates the four intended")
    print("   changes. The report states this rather than burying it.")
    print()
    print("   Second, an open question: is single-shot electron readout")
    print("   fidelity the right quantity for eta in Eq. (5)? Eq. (5) wants the")
    print("   fidelity of the Bell-state measurement that performs the swap.")
    print("   Those are not obviously the same number, and the difference")
    print("   decides whether the T centre can support long chains at all.")

    banner("GATE NOISE CHECK COMPLETE")


if __name__ == "__main__":
    main()
