# The model

Decision variables are a binary `y` per (user pair, path, width) combination
and a binary `r` per candidate repeater location. Utility is
`log2(R (F - 1/2))`, which is not linear, so each (path, width) combination
is evaluated offline and enters as a constant coefficient. That is the
paper's own device and it keeps the program linear.

Candidate paths come from an exact dynamic program rather than the k-shortest
paths heuristic the paper uses. For a path with h links the fidelity depends
only on h, and the rate is proportional to the success probability of the
worst link, so among all h-link paths the best one is whichever minimises its
longest link. That is a bottleneck shortest path with a hop constraint, and
it is solvable exactly. Paths beaten on both hop count and worst link are
dropped, which is a strict dominance and cuts the candidate set by about 60
per cent.


## Why there is no simulator

The model needs two numbers per candidate path: an end-to-end rate and an
end-to-end fidelity. Both have closed forms in the source paper, Eqs. (2) and
(5). They enter the objective as constant coefficients, so the solver never
handles a quantum state.

Three things support the choice:

1. The paper being adapted does not simulate either. It solves in CPLEX, and
   its published code depends on a solver and NetworkX.
2. Goodenough, Coopmans and Towsley (*Quantum* 9, 1744, 2025,
   arXiv:2404.07146) give exact analytic moments of swap-ASAP end-to-end
   fidelity for chains far longer than any route here, together with
   approximations they describe as exponentially tight, explicitly for fast
   optimisation without Monte Carlo.
3. A cross-validation of QuISP and SeQUeNCe (arXiv:2504.01290) found the
   simulators agree on fidelity under identical error parameters and disagree
   on timing. Computing fidelity analytically avoids a known discrepancy.

A discrete-event simulator would produce the same two numbers by sampling,
needing thousands of runs per estimate. Multiplied by the several thousand
solves in the sweep, that does not fit in two weeks.


## Assumptions

These are the things a reviewer should push on, listed rather than buried.

**The end-to-end time formula is a reconstruction.** The paper says
`tau_e2e` accounts for classical message exchange under a sequential
protocol, but does not give the closed form in the abstract or the HTML
rendering. `physics.tau_e2e` uses the natural sequential reading, one
heralding round trip per link summed along the path. The 3.2 ms agreement
above supports it. Check it against Section II-A before the report goes out;
if it is wrong, one function changes.

**Gate and measurement fidelity are held at 1.0.** Eq. (5) carries a per-swap
factor `P_2 (4 eta^2 - 1) / 3`. The paper sets both to approximately 1. The T
centre has measured values, 98.6 per cent gate fidelity and 94.6 per cent
single-shot readout, and substituting them changes the result by more than
any parameter this project sweeps: the hop budget falls from 20 to 5 at
F_L = 0.96, and the projected hardware goes from serving all 18 user pairs to
serving 11. The presets keep both at 1.0 so that the only differences from the
paper's baseline are the four intended ones.

`w1_gate_noise_check.py` measures the effect rather than ignoring it. **An
open question:** Eq. (5) wants the fidelity of the Bell state
measurement that performs the swap. Single-shot electron readout fidelity is
not obviously the same quantity, and the difference decides whether the T
centre can support long chains at all.

**Widths are sampled, not enumerated.** The paper enumerates every path width
up to `min(D, W_E)`. Utility grows as `log2(W)`, so this project uses an
eight-point log-spaced grid, which samples the objective uniformly at an
eighth of the variables. Pass `width_grid=tuple(range(1, 101))` to
`NetworkConfig` to reproduce the paper exactly.

**The rate formula has a stated regime.** Eq. (2) is valid where
`W * p_min >> 1`. Every sweep record carries that quantity so results outside
the regime can be flagged rather than quietly reported.

**Candidate sites are evenly spaced, not real PoPs.** The operator does not publish
PoP locations. Sites are placed at uniform 80 km spacing along each corridor.

**The graph has two components.** The western and eastern halves of the nine
city set are separated by roughly 2000 km through Thunder Bay with no CA9
city between them in the node set. Vault note 37 lists that edge as a
placeholder rather than a verified route, so it is off by default. Pass
`bridge=True` to `build_ca9` to include it.

**Serving every pair is optional on the CA9 runs.** With the requirement
on, one unservable pair makes the whole instance infeasible and the result
says nothing about the other seventeen. With it off, the number of pairs
served becomes the output and the contour where it drops below eighteen is
exactly the feasibility boundary. The validation runs, which need the
infeasibility cliff, keep the requirement on.


## Where the hardware numbers come from

| Quantity | Range | Source |
|---|---|---|
| Generation rate | 1 kHz to 200 kHz | DeAbreu 2023 nanobeam; Afzal 2024 projection |
| Coherence time | 1 ms to 100 ms | Senichev 2025, Nature Nanotechnology |
| Link fidelity | 0.92 to 0.998 | Higginbottom 2025; Afzal 2024 |
| Swap success | 0.50 to 0.95 | linear-optical ceiling; deterministic matter-mediated |
| O band attenuation | 0.35 dB/km | Corning SMF-28 at 1326 nm |

Every bound is bracketed by a published measurement on some platform, so none
of the ranges are invented.
