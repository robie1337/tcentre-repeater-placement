"""Run the whole of weeks 1 and 2 end to end.

Order matters: the validation runs first because nothing downstream is worth
looking at until the model reproduces published results, and the boundary
script pools the grid samples so it has to run after the grid.

Run:  python scripts/run_all.py [--quick]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> float:
    command = [sys.executable, str(HERE / script), *args]
    print("\n" + "=" * 78)
    print(f"RUNNING  {script} {' '.join(args)}")
    print("=" * 78, flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=HERE.parent)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")
    print(f"\n[{script} finished in {elapsed / 60:.1f} min]", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small samples, for checking the pipeline runs")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    grid_n = "8" if args.quick else "24"
    sobol_n = "64" if args.quick else "1024"
    lhs_n = "50" if args.quick else "500"
    jobs = str(args.jobs)

    started = time.time()

    # Week 1
    run("w1_validate.py")
    run("w1b_oband_threshold.py")
    run("w1_surfnet_cliff.py")
    run("w1_gate_noise_check.py")
    run("w1_single_case.py")

    # Week 2
    run("w2_grid_sweep.py", "--n", grid_n, "--jobs", jobs)
    run("w2_sobol.py", "--n", sobol_n, "--jobs", jobs)
    run("w2_boundaries.py", "--n", lhs_n, "--jobs", jobs)
    run("w2_diagnose.py")

    total = time.time() - started
    print("\n" + "=" * 78)
    print(f"ALL STAGES COMPLETE in {total / 60:.1f} min")
    print("=" * 78)
    print("Tables in results/, figures in figures/.")


if __name__ == "__main__":
    main()
