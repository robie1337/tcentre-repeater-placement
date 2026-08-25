# The problem

## Statement

Take a published mixed integer program for quantum repeater placement,
validated at 1550 nm. Move it to 1326 nm, where the silicon T centre emits.
Determine what the model says about a continental fibre network under that
substitution, and determine what the substitution breaks.

Formally, given a graph `G = (V, E)` with end nodes `C ⊂ V`, candidate
repeater sites `S ⊂ V`, edge lengths in kilometres, a set of demand pairs
`D ⊆ C × C`, and a hardware vector

```
h = (alpha, F_L, q_s, R_gen, T2)
```

choose a subset of sites to equip and a path with a memory allocation for
each demand pair, maximising

```
sum over pairs of log2( R_e2e * (F_e2e - 1/2) )
```

subject to the repeater budget, the per-node memory budget, and the
requirement that end-to-end time stays inside end-node coherence. The
objective, the constraints, and the closed forms for `R_e2e` and `F_e2e` are
taken unchanged from Pouryousef et al., IEEE TQE 2024, arXiv:2308.16264v3.

The only intended change is `alpha`, from 0.2 dB/km to 0.35 dB/km, with
`F_L`, `q_s`, `R_gen` and `T2` swept across ranges bracketed by published
measurements rather than fixed at assumed values.

## What counts as an answer

1. The model reproduces the source paper's published results before it is
   modified. Reproduction means the paper's own reported numbers on the
   paper's own test topologies, not qualitative agreement.

2. Every hardware bound is bracketed by a published measurement on some
   platform. No invented ranges.

3. The sensitivity result is reported with its confounds stated, not as a
   bare ranking.

4. Any place where the substitution takes the model outside its own stated
   validity conditions is found and reported, rather than left to a reader
   to discover.

Point 4 is where the actual contribution turned out to be.

## Constraints on method

No discrete-event simulation. The placement model consumes one rate and one
fidelity per candidate path, both of which have closed forms in the source
paper, so a simulator would add cost without adding information. This is a
planning model and the repository says so wherever a result depends on it.

No solver lock-in. Results must agree across at least two independent
solvers, and the default backend must be open source so that anyone can run
the work without a commercial licence.

No unverified citation. Every hardware number must trace to a primary source,
checked against that source rather than against a secondary note. This
constraint was added partway through, after an incorrect attribution was
found to have driven a modelling decision for three months.

No claim beyond the model. The study says where repeaters go under a given
set of assumptions. It does not say the technology works, and any statement
about readiness has to be traceable to a measured number.

## What was deliberately not done

A cross-platform comparison against nitrogen-vacancy centres, silicon-vacancy
centres, and trapped ions through the same model. It is the interesting
follow-up and it would have displaced the reproduction work, so it was
deferred.

Validation against a discrete-event simulator such as NetSquid or SeQUeNCe.
This remains the largest open gap in the work and is stated as such in
[VALIDATION.md](VALIDATION.md).

## Why this problem

Two gaps in the published literature motivated it.

No published placement model puts a named matter-qubit platform inside the
optimisation. Placement papers use generic hardware parameters; hardware
papers characterise devices without asking where they should go. The
intersection was empty at the time of the literature survey.

Separately, published placement work uses European and American research
networks. Canadian carrier geography is different in a way that matters: the
distances are long enough that the speed of light, rather than any hardware
property, sets the floor on what a memory has to survive.

## Success criteria for the report

The report frames the wavelength-transfer failure as the contribution and the
optimiser as the evidence for it. It states the sensitivity result on the
outcome measures where the mechanism is clean, and publishes the correlation
table alongside the confounded ones. It does not claim the ranking is
platform-independent, because that was never tested.
