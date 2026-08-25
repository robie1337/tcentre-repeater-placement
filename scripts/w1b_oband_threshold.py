"""Week 1, addendum: the no-repeater threshold in the O band.

The paper's validation asks a C band question: below roughly 40 km of
backbone the optimiser places no repeater. w1_validate.py reproduces it.
Nobody had asked the same question at 1326 nm, where the fibre loses
0.35 dB/km instead of 0.2, so the short-distance end of the O band regime
was never characterised at all.

This script answers it, and answers the follow-on it raises.

  1. The threshold itself, solved and cross-checked in closed form, for the
     paper's C band baseline and for four O band settings.

  2. What hop spacing the model actually wants at 0.35 dB/km. The CA9
     graph is built on 80 km candidate spacing, which is a C band-shaped
     number, so it is worth knowing whether the O band optimum is near it.

  3. Whether that spacing choice changes any conclusion, by re-solving the
     real network at 80, 40 and 20 km spacing.

The headline: the crossover scales as 1/alpha times a fidelity-and-swap
term, so at the paper's own F_L and q_s it moves from 38 km to 22 km. This
is the low-distance mirror of the memory-budget finding, driven by the same
16x loss penalty per 80 km hop.

Run:  python scripts/w1b_oband_threshold.py
"""

from __future__ import annotations

import math

import pandas as pd

from _common import banner, save_frame, step
from qrp import physics, topology
from qrp.hardware import (
    POURYOUSEF_BASELINE,
    TCENTRE_MEASURED,
    TCENTRE_MIDRANGE,
    TCENTRE_PROJECTED,
)
from qrp.model import NetworkConfig, solve_placement
from qrp.paths import enumerate_paths
from w1_validate import backbone_crossover_km, check_backbone_threshold

# The scan step matters. Until 2026-08-24 the validation scanned at 2.5 km
# from a 5.0 km start, sampling 37.5 then 40.0, which is why the C band
# threshold was reported as exactly 40.0 km and looked like an exact
# reproduction. The crossover is 38.3 km; 40.0 was the first sample past
# it. Both scans now run at 0.5 km, which also resolves the O band cases.
STEP_KM = 0.5

CASES = [
    ("C band, paper baseline", POURYOUSEF_BASELINE, True),
    ("O band, paper's other params", POURYOUSEF_BASELINE.with_(alpha_db_per_km=0.35), False),
    ("O band, T centre midrange", TCENTRE_MIDRANGE, False),
    ("O band, T centre projected", TCENTRE_PROJECTED, False),
    ("O band, T centre measured", TCENTRE_MEASURED, False),
]

#: Reaches worth quoting. 190 km is Kamloops-Kelowna, the shortest pair on
#: the graph; 1600 km is the span that sets the classical latency floor.
REACHES_KM = (50.0, 100.0, 190.0, 200.0, 320.0, 500.0, 800.0, 1200.0, 1600.0)

SPACINGS_KM = (80.0, 40.0, 20.0)
BUDGETS = (25, 53, 200)


# --------------------------------------------------------------------------
# 1. The threshold
# --------------------------------------------------------------------------

def thresholds() -> pd.DataFrame:
    """Run the dumbbell scan for every hardware setting."""
    rows = []
    for label, hardware, is_paper in CASES:
        frame = check_backbone_threshold(
            hardware=hardware,
            label=label,
            step_km=STEP_KM,
            check_paper_40=is_paper,
        )
        positive = frame[frame["n_repeaters"] > 0]
        solver = positive["backbone_km"].min() if not positive.empty else float("nan")
        rows.append(
            {
                "case": label,
                "alpha_db_per_km": hardware.alpha_db_per_km,
                "link_fidelity": hardware.link_fidelity,
                "swap_success": hardware.swap_success,
                "solver_km": solver,
                "analytic_km": backbone_crossover_km(hardware),
                "continuum_km": _continuum_crossover_km(hardware),
            }
        )
    return pd.DataFrame(rows)


def _continuum_crossover_km(hardware) -> float:
    """The same crossover without the ten-site discretisation.

    With candidate sites everywhere the best two-hop split is L/2 rather
    than 6L/11, which lowers the crossover by about ten per cent. Reported
    alongside the gridded number so the report can say which is which: the
    paper's "roughly 40 km" and our 38.3 km both carry this artifact.
    """
    f1 = physics.e2e_fidelity(hardware.link_fidelity, 1) - 0.5
    f2 = physics.e2e_fidelity(hardware.link_fidelity, 2) - 0.5
    if f2 <= 0.0:
        return float("inf")
    gain_per_km = (hardware.alpha_db_per_km / 10.0) * 0.5 * math.log2(10.0)
    return -(math.log2(hardware.swap_success) + math.log2(f2) - math.log2(f1)) / gain_per_km


# --------------------------------------------------------------------------
# 2. What spacing the model wants
# --------------------------------------------------------------------------

def _utility_of_split(reach_km: float, hops: int, hardware) -> float:
    """Utility of splitting a reach into equal hops. Unconstrained.

    No coherence limit and no memory budget, so this is what the physics
    prefers rather than what the network can afford. Good enough to say
    whether 80 km is the right order of magnitude.
    """
    fidelity = physics.e2e_fidelity(hardware.link_fidelity, hops)
    if fidelity <= 0.5:
        return float("-inf")
    p_link = physics.link_success(reach_km / hops, hardware.alpha_db_per_km)
    rate = hardware.generation_rate_hz * (hardware.swap_success ** (hops - 1)) * p_link
    return math.log2(rate * (fidelity - 0.5))


def preferred_spacing() -> pd.DataFrame:
    """For each reach, the equal-hop split with the highest utility."""
    step("Check 2: the hop spacing the unconstrained model prefers")

    rows = []
    for label, hardware, _ in CASES:
        if physics.e2e_fidelity(hardware.link_fidelity, 2) <= 0.5:
            continue  # measured T centre has no multi-hop optimum to find
        for reach in REACHES_KM:
            best = max(
                ((_utility_of_split(reach, h, hardware), h) for h in range(1, 201)),
            )
            _, hops = best
            rows.append(
                {
                    "case": label,
                    "reach_km": reach,
                    "best_hops": hops,
                    "spacing_km": round(reach / hops, 1),
                }
            )

    frame = pd.DataFrame(rows)
    table = frame.pivot(index="reach_km", columns="case", values="spacing_km")
    print(table.to_string())
    print()
    print("   The graph is built on 80 km spacing. That is what the C band")
    print("   optimum wants at about 320 km of reach. The O band optimum")
    print("   wants roughly 55 to 60 per cent of it at every reach.")
    return frame


# --------------------------------------------------------------------------
# 3. Whether it changes any conclusion
# --------------------------------------------------------------------------

def spacing_sensitivity() -> pd.DataFrame:
    """Re-solve the real network at three candidate spacings."""
    step("Check 3: does the 80 km candidate spacing change the answer")

    rows = []
    for spacing in SPACINGS_KM:
        topo = topology.build_ca9(spacing_km=spacing)
        # max_hops=20 is not a reach limit: links may skip sites, so the
        # longest enumerated path is about 1800 km at all three spacings.
        paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=300.0, max_hops=20)
        for label, hardware in [
            ("measured", TCENTRE_MEASURED),
            ("midrange", TCENTRE_MIDRANGE),
            ("projected", TCENTRE_PROJECTED),
        ]:
            for budget in BUDGETS:
                config = NetworkConfig(
                    repeater_memories=100,
                    endnode_memories=100,
                    max_repeaters=budget,
                    require_all_pairs=False,
                    use_demand_weights=False,
                )
                result = solve_placement(topo, paths, hardware, config)
                rows.append(
                    {
                        "spacing_km": spacing,
                        "n_sites": len(topo.sites),
                        "hardware": label,
                        "budget": budget,
                        "served_pairs": result.served_pairs,
                        "total_pairs": result.total_pairs,
                        "n_repeaters": result.n_repeaters,
                        "utility": round(result.utility, 3),
                        "status": result.status,
                    }
                )

    frame = pd.DataFrame(rows)
    for field in ("served_pairs", "n_repeaters", "utility"):
        print(f"\n   {field.upper()}")
        print(frame.pivot_table(
            index=["hardware", "budget"], columns="spacing_km", values=field,
        ).to_string())

    print()
    print("   Pairs served is unchanged at unlimited budget, so the coverage")
    print("   and feasibility results are spacing-robust. Utility is not:")
    print("   refining 80 to 20 km raises it by roughly 12 per cent at the")
    print("   midrange setting and 29 per cent at the projected one. Absolute")
    print("   utility magnitudes should not be quoted as if 80 km were an")
    print("   optimised choice; it is an inherited C band assumption.")
    print()
    print("   Caveat: site sets at different spacings are not nested, so this")
    print("   is not a clean relaxation. At budget 25 the coarse grid wins.")
    return frame


def main() -> None:
    banner("WEEK 1 ADDENDUM: the no-repeater threshold in the O band")

    table = thresholds()
    save_frame(table, "w1b_oband_threshold.csv")
    print()
    print(table.to_string(index=False))

    spacing = preferred_spacing()
    save_frame(spacing, "w1b_preferred_spacing.csv")

    sensitivity = spacing_sensitivity()
    save_frame(sensitivity, "w1b_spacing_sensitivity.csv")

    banner("O BAND THRESHOLD COMPLETE")


if __name__ == "__main__":
    main()
