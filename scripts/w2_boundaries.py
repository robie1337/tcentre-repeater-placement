"""Week 2, Thursday and Friday: boundaries and regime labels.

Two jobs.

First, the Latin hypercube tier. Extra samples spread across the whole box,
concentrated by rejection near points where the number of pairs served
changes, so the boundaries the grid only samples coarsely get sharpened.

Second, the regime description. A phase diagram is only a contribution if the
regions have names and each boundary has a mechanism attached, so this fits a
shallow decision tree to the served-pairs surface and prints the rules it
finds in plain language.

Run:  python scripts/w2_boundaries.py [--n 500] [--jobs 6]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import RESULTS, banner, save_frame, step
from qrp.hardware import LABELS, LOG_SCALED, SWEEP_ORDER
from qrp.model import NetworkConfig
from qrp.sweep import SweepContext, latin_hypercube_points, run_points

CONFIG = NetworkConfig(
    repeater_memories=100,
    endnode_memories=100,
    max_repeaters=10**6,
    require_all_pairs=False,
    use_demand_weights=False,
)


def describe_regimes(frame: pd.DataFrame, target: str = "served_pairs") -> str:
    """Fit a shallow tree to the surface and read the rules back out.

    A decision tree is used because the output is what we want in the report:
    a small set of threshold rules in the original parameter units, rather
    than a black-box fit. Depth is capped at three so the rules stay
    describable in a paragraph.
    """
    from sklearn.tree import DecisionTreeRegressor, export_text

    features = list(SWEEP_ORDER)
    X = frame[features].copy()
    for name in features:
        if name in LOG_SCALED:
            X[name] = np.log10(X[name])

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=0)
    tree.fit(X.to_numpy(), frame[target].to_numpy())

    names = [
        f"log10({LABELS[n]})" if n in LOG_SCALED else LABELS[n] for n in features
    ]
    text = export_text(tree, feature_names=names, decimals=3)

    importance = pd.DataFrame(
        {"parameter": [LABELS[n] for n in features],
         "tree_importance": tree.feature_importances_}
    ).sort_values("tree_importance", ascending=False, ignore_index=True)

    return text, importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    banner("WEEK 2 BOUNDARIES: Latin hypercube tier and regime labels")

    context = SweepContext.build(
        spacing_km=80.0, max_link_km=300.0, max_hops=20, config=CONFIG
    )

    step(f"Latin hypercube sample, {args.n} points")
    points = latin_hypercube_points(args.n)
    frame = run_points(context, points, n_jobs=args.jobs)
    print(f"   feasible: {(frame['status'] == 'optimal').sum()} / {len(frame)}")
    print(f"   pairs served ranges {frame['served_pairs'].min()} "
          f"to {frame['served_pairs'].max()}")
    save_frame(frame, "w2_lhs_samples.csv")

    # Pool with the grid samples if they exist, so the regime fit sees
    # everything computed so far rather than the hypercube alone.
    pooled = [frame]
    for name in ("w2_grid_rate_fidelity.csv", "w2_grid_t2_swap.csv",
                 "w2_grid_t2_rate.csv"):
        path = RESULTS / name
        if path.exists():
            pooled.append(pd.read_csv(path))
    combined = pd.concat(pooled, ignore_index=True)
    print(f"   pooled with grid samples: {len(combined)} points total")
    save_frame(combined, "w2_all_samples.csv")

    step("Regime rules for the number of pairs served")
    try:
        text, importance = describe_regimes(combined, "served_pairs")
        print(text)
        print(importance.to_string(index=False))
        save_frame(importance, "w2_regime_importance.csv")
        (RESULTS / "w2_regime_tree.txt").write_text(text, encoding="utf-8")
        print(f"   wrote results/w2_regime_tree.txt")
    except ImportError:
        print("   scikit-learn is not installed, so the regime fit is skipped.")
        print("   pip install scikit-learn to enable it.")

    step("Where the network can serve every pair")
    full = combined[combined["all_pairs_served"] == 1]
    if full.empty:
        print("   No sampled hardware point serves all 18 pairs.")
        print("   The binding limit is end-node coherence: the longest western")
        print("   pair needs about 16 ms and the sweep box tops out at 100 ms,")
        print("   so check the per-parameter minima below.")
    else:
        for name in SWEEP_ORDER:
            print(f"   {LABELS[name]:<26} minimum needed: {full[name].min():.4g}")

    banner("BOUNDARIES COMPLETE")


if __name__ == "__main__":
    main()
