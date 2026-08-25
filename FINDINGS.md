# Findings

Numbers come from 6,144 Saltelli solves, 1,728 grid solves, 500 Latin
hypercube solves, and the threshold scans. The pipeline runs in about five
minutes on six cores.

Coherence time dominates every outcome. Total-effect Sobol indices, as a
share of variance explained:

| Parameter | Utility | Pairs served | Repeaters placed |
|---|---|---|---|
| Coherence time | 0.79 | 0.91 | 0.94 |
| Generation rate | 0.19 | 0.07 | 0.08 |
| Swap success | 0.09 | 0.10 | 0.08 |
| Link fidelity | 0.03 | 0.05 | 0.02 |

Link fidelity barely registers, which says more about the sweep box than
about physics. The published range starts at 0.92, already good enough that
fidelity is not what stops a path being used. The mechanism behind the
coherence result is geography: the longest route is about 1600 km, light
takes 8 ms to cross it each way, and any protocol waiting for an
acknowledgement needs a memory that survives 16.6 ms.

Read that ranking with one caveat. The utility-based indices are confounded,
since the rate bias correlates with coherence at -0.684. The indices on pairs
served and on feasibility are clean, because the mechanism there is
rate-free, and those are the ones the report leads with. The bias is signed,
which makes the coherence advantage an upper bound.

Below roughly 6 ms of coherence, generation rate changes the outcome by
exactly zero. Not approximately zero: the response is flat across the full
rate axis in 10 of 24 grid columns. Below that boundary no amount of
entanglement generation helps, because the memory cannot hold what it
produces long enough for it to be useful.

With every other parameter at the top of its published range, this is what
each one has to reach before all eighteen pairs can be served:

| Parameter | Needed | Published range | Position |
|---|---|---|---|
| Coherence time | 16.6 ms | 1 to 100 ms | 61% |
| Generation rate | 74.5 kHz | 1 to 200 kHz | 74% |
| Link fidelity | 0.974 | 0.92 to 0.998 | 69% |
| Swap success | 0.897 | 0.50 to 0.95 | 88% |

All four sit between 61 and 88 per cent of the way up. Serving the whole
network needs essentially the entire roadmap, not part of it.

## Where the rate equation breaks

This is the main result and it was not anticipated.

Eq. (2) of the source paper is stated to hold where `W * p_min >> 1`. It
never holds here. All 5,286 solves with a defined value sit below 1, median
0.033.

The cause is the wavelength. At 0.2 dB/km an 80 km link transmits 2.5e-2 of
the photons sent, so about 40 memories per node reach the stated regime, and
the paper's experiments use 100. At 0.35 dB/km the same link transmits
1.6e-3, and the same regime needs about 631. A published placement model,
moved from 1550 nm to 1326 nm, silently leaves the regime its own rate
equation is stated to hold in, and reported rates in the low-rate corner are
optimistic as a result.

This survives every weakness in the project. It does not depend on the timing
reconstruction, the site spacing, the width sampling, or the swap-noise
question.

Raising the memory ceiling does not rescue it. With up to 2048 memories
available and wide allocations on offer, the optimiser buys between 256 and
1024 and still leaves 15 of 18 paths below the validity threshold. A rate
model valid at low success probability is structurally necessary for O band
planning rather than a patch for a corner case.

The same penalty shows up at the other end of the distance axis. The source
paper's validation threshold, below which no repeater is worth placing, had
only ever been run at 1550 nm. The crossover scales as `1/alpha` times a term
in link fidelity and swap success, so it moves:

| Setting | Solver | Closed form | Continuum |
|---|---|---|---|
| C band, paper's parameters | 38.5 km | 38.3 km | 34.9 km |
| O band, paper's other parameters | 22.0 km | 21.9 km | 19.9 km |
| O band, T centre midrange | 12.5 km | 12.1 km | 11.0 km |
| O band, T centre projected | 2.0 km | 1.5 km | 1.4 km |
| O band, T centre measured | never | infinite | infinite |

Solver and closed form agree everywhere within the scan step, and
38.3 × (0.2/0.35) = 21.9 reproduces the second row exactly. The last row is
not a solver failure: at the measured link fidelity of 0.60, one swap drops a
path below the classical floor of 1/2, so a repeatered path never has finite
utility and the direct link wins at every distance. Measured hardware has no
repeater regime at all, which is what the 30 km laboratory demonstration
already showed.

So the O band network is repeater-hungry at every scale, and it is the same
16x loss penalty seen from both ends.

## Buffering and robustness

The memoryless rate floor collapses coverage to 3 of 18 pairs. The buffered
model matches the ceiling at 18 of 18. T centre coherence spans 12 to 125
slot round-trips, so buffering is exactly what this hardware provides. That
is the coherence result arrived at from an independent direction, and it is
the cleanest way to say why coherence dominates: coherence is what converts
attempts into a usable rate.

Coherence has finished first in every configuration tested, across two rate
models, two swap-noise brackets, and four- and five-parameter sweeps. Under
the conservative buffered model the total-effect index is 0.84 on pairs
served and 0.78 on utility. Swept across the full corrected swap-noise
bracket [0.71, 0.997] it stays between 0.82 and 0.91, while swap quality
never exceeds 0.10.

## Limits

The model has not been validated against a discrete-event simulator. No
comparison was run against other qubit platforms, so nothing here says the
T centre is better or worse than a nitrogen-vacancy centre or a trapped ion
under the same optimisation.

The verdict the numbers support is narrow. Today's measured hardware serves
short metro and regional links and nothing longer. The full nine-city network
works only under the complete published roadmap, and the projected swap
quality of roughly 0.997 only just clears what a comparable 900 km study
requires. There is no margin in that.
