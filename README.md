# Repeater placement for silicon T centres

A research workspace for a reproduction and extension of the quantum repeater
placement model of Pouryousef et al., IEEE TQE 2024, arXiv:2308.16264v3. The
objective and constraints are theirs. The hardware is the silicon T centre,
which emits at 1326 nm instead of 1550 nm.

Stage 1 is complete: the model reproduces the source paper, has been swept
over its hardware parameters, and has been externally reviewed. The technical
report is in progress. The results have not been validated against a
discrete-event simulator.

## Contents

| | |
|---|---|
| [PROBLEM.md](PROBLEM.md) | the problem, the method constraints, what counts as an answer |
| [MODEL.md](MODEL.md) | how the model works, what it assumes, where the hardware numbers come from |
| [VALIDATION.md](VALIDATION.md) | every check run against the source paper, and every error found, including ours |
| [FINDINGS.md](FINDINGS.md) | the results |
| `src/qrp/` | physics, hardware, topology, paths, model, solver, sweep, sensitivity, figures |
| `scripts/` | `w1_*` validation, `w1b_*` the O band threshold, `w2_*` sweeps, `w3*` the fixes, `w4_*` to `w6_*` whether the corrections change the plan, `w7_*` to `w12_*` checks on the approximations, the solvers, the cost of rerunning, the attempt clock and the swap schedule |
| `tests/` | 324 tests, including exhaustive and brute-force checks of the candidate path set on small networks |
| `results/`, `figures/` | regenerable output, tracked so the tables can be checked |

## Install and run

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

`run_all.py --quick` runs the same pipeline at small sample sizes to check
everything works, in a couple of minutes.

Tests:

```bash
PYTHONPATH=src python -m pytest tests -q
```

## Attribution

The literature review, the problem definition and the research direction are
my own work. Claude (Anthropic) wrote portions of the code and helped draft
parts of the documentation. Every number here comes from code in this
repository that anyone can run.
