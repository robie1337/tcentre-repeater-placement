"""Does the rate-approximation violation invalidate the sensitivity ranking?

Finding 4 established that W * p_min < 1 at every solve, so Eq. (2)
overestimates the rate everywhere. The defence offered was that the bias
points the same way at every point, so the Sobol ranking survives.

That defence is only sound if the bias is roughly UNIFORM across the
parameter box. If the violation is worse in one corner than another, it is a
confound: the sensitivity indices would partly measure the bias rather than
the physics.

This measures it directly. Two questions:

  1. Below the coherence threshold, does anything vary with generation rate,
     or is the feasible region genuinely degenerate?
  2. Does the severity of the violation correlate with the swept parameters?

Run:  python scripts/w2_bias_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import RESULTS, banner, save_frame, step

pd.set_option("display.width", 200)


def main() -> None:
    banner("IS THE RATE-APPROXIMATION BIAS A CONFOUND?")

    grid = pd.read_csv(RESULTS / "w2_grid_t2_rate.csv")
    sobol = pd.read_csv(RESULTS / "w2_sobol_samples.csv")

    # ------------------------------------------------------------------
    step("1. Below the coherence threshold, is anything actually varying?")
    print("   The claim under test: 'generation rate changes the outcome by")
    print("   exactly zero' looks like a degenerate feasible region.")
    print()
    low = grid[grid["t2_s"] <= 6.1e-3]
    print(f"   {len(low)} grid points at or below 6.1 ms")
    for column in ("served_pairs", "n_repeaters", "utility"):
        spreads = low.groupby("t2_s")[column].agg(lambda s: s.max() - s.min())
        n_flat = int((spreads < 1e-9).sum())
        print(f"     {column:<14} flat in {n_flat}/{len(spreads)} columns, "
              f"max spread across rate = {spreads.max():.4g}")

    print()
    print("   Reading: served_pairs and n_repeaters are integer counts gated by")
    print("   a hard feasibility test that rate does not enter, so exact zeros")
    print("   are the expected behaviour, not a degeneracy. If utility also")
    print("   shows zero spread, that WOULD be suspicious, because utility")
    print("   scales with rate on every served pair.")

    # ------------------------------------------------------------------
    step("2. Is the violation uniform across the box, or does it correlate?")
    valid = sobol.dropna(subset=["min_w_pmin"]).copy()
    valid["log_wp"] = np.log10(valid["min_w_pmin"])
    print(f"   {len(valid)} solves with a defined W * p_min")
    print(f"   range {valid['min_w_pmin'].min():.3g} to {valid['min_w_pmin'].max():.3g}, "
          f"median {valid['min_w_pmin'].median():.3g}")
    print()
    print("   Spearman correlation of log10(W * p_min) with each swept parameter:")
    rows = []
    for name in ("generation_rate_hz", "t2_s", "link_fidelity", "swap_success"):
        rho = valid["log_wp"].corr(valid[name], method="spearman")
        rows.append({"parameter": name, "spearman_rho": round(float(rho), 4)})
        flag = "  <-- strong" if abs(rho) > 0.4 else ("  <-- moderate" if abs(rho) > 0.2 else "")
        print(f"     {name:<22} rho = {rho:+.4f}{flag}")

    save_frame(pd.DataFrame(rows), "w2_bias_correlation.csv")

    # ------------------------------------------------------------------
    step("3. How far does the violation range across the box?")
    decades = np.log10(valid["min_w_pmin"].max() / valid["min_w_pmin"].min())
    print(f"   W * p_min spans {decades:.1f} orders of magnitude across the sweep.")
    print()
    if decades > 2:
        print("   That is NOT uniform. The bias is far worse in some of the box")
        print("   than in others, so 'it biases every point the same way' is too")
        print("   strong a defence as stated.")
    else:
        print("   Reasonably uniform.")

    # ------------------------------------------------------------------
    step("4. Verdict")
    strongest = max(rows, key=lambda r: abs(r["spearman_rho"]))
    print(f"   Strongest correlate: {strongest['parameter']} "
          f"(rho = {strongest['spearman_rho']:+.3f})")
    print()
    print("   What this does and does not license:")
    print()
    print("   - If the violation correlates strongly with a swept parameter,")
    print("     that parameter's Sobol index is partly measuring the bias.")
    print("   - If it correlates with generation rate specifically, then the")
    print("     rate index is the suspect one, and the headline claim that")
    print("     COHERENCE dominates is unaffected, because coherence acts")
    print("     through a feasibility constraint that contains no rate term.")
    print("   - The honest statement for the report is whichever of those the")
    print("     numbers above support. Do not assert the ranking is safe")
    print("     without pointing at this table.")

    banner("DONE")


if __name__ == "__main__":
    main()
