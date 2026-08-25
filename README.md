# Repeater placement for silicon T centres

Where do you put quantum repeaters on a real fibre network, and how does that
answer move as the hardware improves? This reproduces the placement MIP of
Pouryousef et al. (IEEE TQE 2024, arXiv:2308.16264v3), swaps in silicon
T centre hardware, and reports what breaks.

The objective and constraints are theirs, unchanged. What changes is the
wavelength: T centres emit at 1326 nm, where fibre loses 0.35 dB/km instead
of the 0.2 the paper assumes at 1550 nm. Link fidelity, swap success and
coherence time are swept rather than fixed, across ranges bracketed by
published measurements.

The main result is a failure of transfer. Eq. (2) is stated to hold where
`W * p_min >> 1`; at 1326 nm it never does, median 0.033 over 5,286 solves.
Reaching the stated regime needs ~631 memories per node against ~40 at
1550 nm. The paper uses 100. The same 16x loss penalty shows up at the other
end of the distance axis too: repeaters start paying off at 22 km, not 38.

Coherence time dominates every outcome measured, but on this geography that
is a statement about Canada rather than about T centres. The longest route is
1600 km, light needs 8 ms each way, so nothing matters until a memory
survives 16.6 ms.

Runs use CA9, a nine-city Canadian long-haul topology, 4,810 km of fibre, 53
candidate sites at 80 km spacing, 18 demand pairs. Route distances are real.
The operator does not publish PoP locations, so sites sit at uniform spacing
along each corridor.

[PROBLEM.md](PROBLEM.md) states the problem and what counts as an answer.
[VALIDATION.md](VALIDATION.md) has every check and every error found,
including this project's own. [FINDINGS.md](FINDINGS.md) has the results.

## There is no network simulator here, and that is deliberate

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

## Layout

```
src/qrp/
  physics.py      Eqs. (1), (2), (5), (6), (7) and the timing model
  hardware.py     parameter sets, presets, and the four-parameter sweep box
  topology.py     the nine city CA9 graph and the paper's test cases
  paths.py        candidate path enumeration by exact dynamic programming
  model.py        the MILP: variables, constraints, solving, result extraction
  solver.py       HiGHS and Gurobi backends over a common sparse form
  sweep.py        grid, line and Latin hypercube sampling, run in parallel
  sensitivity.py  Saltelli sampling and Sobol indices
  figures.py      every figure the report needs

scripts/
  w1_validate.py            Tuesday: reproduce the paper's published results
  w1_gate_noise_check.py    the assumption that matters most, measured
  w1_single_case.py         Friday: one complete case, drawn on the map
  w2_grid_sweep.py          Tuesday: the phase diagrams
  w2_sobol.py               Wednesday: which parameter moves the result
  w2_boundaries.py          Thursday and Friday: boundaries and regime rules
  run_all.py                all of the above, in order
```

## What the model does

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

## Reproductions

`w1_validate.py` checks the model against results someone else published.

| Check | Paper | This model |
|---|---|---|
| Backbone length below which no repeater is placed | about 40 km | 40.0 km |
| End-node coherence below which nothing is feasible | 3.2 ms on SURFnet | 3.211 ms on SURFnet |
| Three solvers, same objective | n/a | HiGHS, Gurobi and CPLEX agree to 1e-6 |

The coherence result is run on the authors' own topology. Their repository
ships `data/SurfnetCore.gml`, which carries a measured fibre length on every
edge, so the distances are theirs rather than assumed. Real fibre distances
between the four end nodes Rabbie et al. name are:

| Pair | Fibre | Coherence it requires |
|---|---|---|
| Enschede to Groningen | 250.7 km | 2.507 ms |
| Delft to Enschede | 299.5 km | 2.995 ms |
| Enschede to Maastricht | 301.8 km | 3.018 ms |
| Delft to Maastricht | 320.5 km | 3.205 ms |
| Delft to Groningen | 320.7 km | 3.207 ms |
| Groningen to Maastricht | 495.8 km | 4.958 ms |

The paper uses four of these six and does not say which. Ten of the fifteen
possible choices include the 496 km pair and imply a limit near 5 ms; the
other five exclude it and imply 3.2 ms. Solving one of those five gives
**3.211 ms** against the published 3.2 ms.

That is a reproduction with no tuned parameter, and it is what supports the
reconstructed `tau_e2e`, which matters because the paper does not give the
closed form.

An earlier version of this check used a straight line of fibre and asserted
that SURFnet corresponded to 320 km, because `2 x 320 / c` gives 3.2 ms. That
was circular, the length having been chosen to produce the published answer.
`scripts/w1_validate.py` still runs the straight-line sweep, but only as an
internal consistency check that the cliff obeys `2L/c`;
`scripts/w1_surfnet_cliff.py` is the real comparison.

CPLEX is included because it is the solver the paper used. The pip package is
the Community Edition, free and needing no registration, capped at 1000
variables. Every instance here fits: the full CA9 model is 809 variables,
and all three solvers return 200.860328 on it.

### Cross-check against the authors' published code

Their repository is `github.com/pooryousefshahrooz/q_net_planning`, and it
depends on CPLEX and NetworkX with no simulator, which is the claim made at
the top of this README.

Their `solver.py` builds the link-based objective as

```python
alpha = 0.02 * np.log(10)
objective = sum(-alpha * minCapacity[i]
                + sum(whichmemory[i, m] * np.log(m) for m in list_W)
                + np.log(network.q) * hops[i]
                for i in list_C)
```

That is the same utility used here, with the logarithm expanded term by term:

| Their term | This code | Agreement |
|---|---|---|
| `-alpha * minCapacity[i]`, `alpha = 0.02 ln 10` | `ln p_min` where `p_min = 10^(-0.02 l)` | exact |
| `sum whichmemory[i,m] * ln(m)` | `ln W` over the enumerated width grid | exact |
| `ln(q) * hops[i]` | `(h - 1) ln q_s` | differs by the constant `ln q_s` per pair |

The hop term differs by one because they count `h` swaps where Eq. (2) has
`h - 1`. Since their flow constraints force exactly one path per user pair,
that is a constant offset across the whole objective and does not move the
optimum.

Their constraints also line up: repeater memory bounded by `D[n] * y[n]`,
repeater count bounded by `network.R`, flow conservation with unit source and
sink, and one memory choice per pair.

The one real difference is that their link-based objective has no fidelity
term at all, which is the simplification the paper states it makes to keep
the program linear. This project uses the path-based formulation instead and
keeps fidelity in, which is the point of the exercise.

## What the sweep found

Numbers below come from 6,144 Saltelli solves plus 1,728 grid solves and 500
Latin hypercube solves. The whole pipeline runs in about four minutes on six
cores.

### Coherence time dominates, and nothing else is close

Total-effect Sobol indices, share of variance explained:

| Parameter | Utility | Pairs served | Repeaters placed |
|---|---|---|---|
| Coherence time T2 | 0.79 | 0.91 | 0.94 |
| Generation rate | 0.19 | 0.07 | 0.08 |
| Swap success | 0.09 | 0.10 | 0.08 |
| Link fidelity | 0.03 | 0.05 | 0.02 |

Link fidelity barely registers. That is not because fidelity does not matter
physically but because the published range, 0.92 to 0.998, is already good
enough that end-to-end fidelity is not what stops a path being used.
Coherence is, because end-to-end time scales with route length and the
western routes are long.

If the ranking is read as a roadmap question, the answer is that coherence
time buys more than the other three combined.

### Hardware requirements for the full network

With every other parameter at the top of its published range, this is what
each one has to reach before all eighteen city pairs can be served:

| Parameter | Needed | Published range |
|---|---|---|
| Coherence time T2 | 16.6 ms | 1 to 100 ms |
| Generation rate | 74.5 kHz | 1 to 200 kHz |
| Link fidelity | 0.974 | 0.92 to 0.998 |
| Swap success | 0.897 | 0.50 to 0.95 |

The 16.6 ms figure has a clean mechanism. The longest western route,
Vancouver to Saskatoon, is about 1600 km of fibre, and the end-to-end time
under the sequential protocol is `2L/c`, which is 16 ms. Everything else
follows from that.

### The rate approximation does not survive the move to the O band

This is the most consequential finding and it was not anticipated.

Eq. (2) is stated as valid where `W * p_min >> 1`. Across the sweep it never
holds: every one of the 5,286 solves with a defined value has `W * p_min < 1`,
median 0.033.

The mechanism is the wavelength. At the C band value of 0.2 dB/km an 80 km
link transmits 2.5e-2 of the photons sent, so about 40 memories per node put
you in the valid regime, and the paper's experiments use 100. At the O band
value of 0.35 dB/km the same link transmits 1.6e-3, and reaching the same
regime needs about 631.

So a published placement model, moved from 1550 nm to 1326 nm, silently
leaves the regime its own rate equation is stated to hold in. Reported rates
in the low-rate corner are optimistic as a result. It does not change the
parameter ranking, since it biases every point the same way, but it is a real
limitation and it points at the next question: either the T centre network
needs far more multiplexing than the literature assumes, or the placement
model needs a rate equation that stays valid at low success probability.

### Repeaters start paying off at 22 km, not 38

The source paper validates its model with a threshold: below roughly 40 km of
backbone the optimiser places no repeater at all, because one direct link
beats two hops plus a swap. That check had only ever been run at 1550 nm.

Running it at 1326 nm moves the threshold, and the closed form says why. The
crossover scales as `1/alpha` times a term in link fidelity and swap success,
so raising attenuation from 0.2 to 0.35 dB/km pulls it in by a factor of
0.2/0.35:

| Setting | Solver | Closed form |
|---|---|---|
| C band, paper's parameters | 38.5 km | 38.3 km |
| O band, paper's other parameters | 22.0 km | 21.9 km |
| O band, T centre midrange | 12.5 km | 12.1 km |
| O band, T centre projected | 2.0 km | 1.5 km |
| O band, T centre measured | never | infinite |

The last row is not a solver failure. At the measured link fidelity of 0.60 a
single swap drops a path below the classical floor of 1/2, so a repeatered
path never has finite utility and a direct link wins at every distance.
Measured T centre hardware has no repeater regime at all, which is the same
thing the 30 km laboratory demonstration says.

Read together with the memory-budget result, this is one effect seen from two
ends. The O band network is repeater-hungry at every scale.

Reproduce with `python scripts/w1b_oband_threshold.py`.

## Assumptions that are choices, not facts

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

## Sources for the hardware numbers

| Quantity | Range | Source |
|---|---|---|
| Generation rate | 1 kHz to 200 kHz | DeAbreu 2023 nanobeam; Afzal 2024 projection |
| Coherence time | 1 ms to 100 ms | Senichev 2025, Nature Nanotechnology |
| Link fidelity | 0.92 to 0.998 | Higginbottom 2025; Afzal 2024 |
| Swap success | 0.50 to 0.95 | linear-optical ceiling; deterministic matter-mediated |
| O band attenuation | 0.35 dB/km | Corning SMF-28 at 1326 nm |

Every bound is bracketed by a published measurement on some platform, so none
of the ranges are invented.

## Attribution

The literature review, the problem definition and the research direction are
my own work. Claude (Anthropic) wrote portions of the code and helped draft
parts of the documentation. Every number here comes from code in this
repository that anyone can run.

## Licence

MIT. See [LICENSE](LICENSE).
