# Validation and corrections

Nothing in this project is worth reading until the model reproduces results
someone else published. This file records what was checked, what passed, and
what turned out to be wrong. The errors include several of this project's
own.

## Reproductions

Run with `python scripts/w1_validate.py`.

| Check | Published | This model | Status |
|---|---|---|---|
| No-repeater threshold, C band | roughly 40 km | 38.5 km, 38.3 km in closed form | pass, see note below |
| Coherence cliff on SURFnet | 3.2 ms | 3.211 ms on the authors' own topology file | pass |
| Solver agreement | not applicable | HiGHS, Gurobi and CPLEX agree to 1e-6 | pass |
| Formulation against the authors' published code | not applicable | matches term by term | pass |

The coherence cliff was reproduced on the authors' own topology file rather
than on a reconstruction, which makes it the strongest of the four.

**The 40 km threshold was never exactly 40 km.** Until 24 August 2026 this
table reported it as exactly 40.0 km, which read as an exact reproduction of
the paper's figure. It was an artifact of the scan grid: the sweep stepped
2.5 km from a 5.0 km start, sampling 37.5 then 40.0, and 40.0 was simply the
first sample past the crossover.

The crossover is 38.3 km in closed form and 38.5 km from the solver at 0.5 km
resolution. Both scans now run at 0.5 km. This is still a pass against
"roughly 40 km", but it is not an exact hit and the report does not present
it as one.

Both figures also carry a discretisation effect. With ten equally spaced
candidate sites the best two-hop split leaves a worst link of 6/11 of the
backbone rather than 1/2, which inflates the crossover by about ten per cent.
With candidate sites everywhere the crossover is 34.9 km.

## Third-party data

`data/SurfnetCore.gml` is not redistributed here. It is the source paper
authors' own data, their repository carries no licence file, and the file
appears to originate from the Internet Topology Zoo before that. Neither
licence could be confirmed, so relicensing it under this repository's MIT
terms is not something we can do. Fetch it from
`github.com/pooryousefshahrooz/q_net_planning` and drop it in `data/`, and
the coherence cliff check reproduces as documented. Without it that one check
skips and the rest of the pipeline runs.

The seven lines of the authors' solver quoted below are a quotation for
comparison, attributed, not a copy of their program.

## How each one was done

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

### Against the authors' published code

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

## Checks on our own assumptions

**The rate equation leaves its stated regime.** Eq. (2) of the source
paper is stated to hold where `W * p_min >> 1`. Across 5,286 solves with a
defined value it never holds, median 0.033.

This is reported as a finding rather than fixed silently, because it is the
main result. Three rate models are implemented so that the conclusion can be
tested against the assumption: the paper's own, a memoryless floor, and a
buffered model valid at low success probability. The parameter ranking
survives all three.

**The swap-noise term was swept rather than assumed.** Eq. (5) carries a
per-swap factor whose physical identity is ambiguous for this platform.
Rather than pick a number, the combined factor is swept across [0.71, 0.997],
bracketed at the low end by measured gate and SPAM figures and at the high
end by projected cavity-assisted readout. The ranking does not
change anywhere in that bracket.

**Candidate spacing is a C band choice.** The 80 km site spacing was
inherited without being tested. Checked on 24 August 2026 by re-solving at
80, 40 and 20 km:

- Pairs served does not change at unlimited budget. Coverage and feasibility
  conclusions are robust to the spacing.
- Utility does change. Refining from 80 km to 20 km raises it by 12 per cent
  at the midrange setting and 29 per cent at the projected one.

So coverage results stand as reported, and absolute utility magnitudes are
not quoted as though 80 km were optimised. The site sets at different
spacings are not nested, so this is not a clean relaxation.

## Citation audit

A sweep against arXiv primary sources on 16 and 24 August 2026 checked every
attribution this project relies on. Six were wrong. Four had propagated into
code comments.

| As recorded | Verified source |
|---|---|
| 0.946 readout fidelity, "Higginbottom 2022" | Raha, Chen, Phenicie, Ourari, Dibos and Thompson, Nat. Commun. 11, 1605 (2020). A single **erbium** ion, not a T centre. arXiv:2103.07580 reports no readout fidelity at all. |
| arXiv:2504.15467, "Senichev et al." | Song, Zhang, Komza, Fiaschi, Xiong, Zhi, Dhuey, Schwartzberg, Schenkel, Hautier, Zhang and Sipahigil, Nat. Nanotechnol. (2025) |
| 900 km Bonn to Berlin study, "Hahn et al." | Ferreira da Silva, Avis, Slater and Wehner, arXiv:2303.03234, QST 9, 045041 (2024) |
| arXiv:2311.04858, "Asadi and Simmons" | Simmons, single author |
| arXiv:2508.06474, "Bourassa" | Taherizadegan, Kimiaee Asadi, Ji, Higginbottom and Simon |
| arXiv:2404.07146, "Iñesta et al." | Goodenough, Coopmans and Towsley, Quantum 9, 1744 (2025) |

Two of these changed more than a name.

The 0.946 figure is from the wrong platform entirely. It drove a modelling
decision for three months before anyone checked it. It is retained in the
code only so the earlier sensitivity check stays reproducible, marked as
misattributed, and it appears nowhere in the reported results.

The coherence measurements are an independent group's work, not the device
vendor's. The verified values are 0.41(2) ms for the electron spin echo,
112(12) ms for the hydrogen nuclear spin, and 67(7) ms for the silicon
nuclear spin. This makes the coherence evidence stronger than the project had
been treating it.

The audit left two things open. The "220 ms" upper bound this project had
been quoting for nuclear coherence appears in none of the sources checked.
Until it turns up, the verified claim is 112(12) ms.

The sweep box's lower coherence bound of 1 ms sits below the measured
electron echo of 0.41 ms. The bound is defensible only because the model
assumes a nuclear memory, and the report has to say which memory it means.

## What is not validated

This is a planning model. It has not been validated against a discrete-event
simulator, so its rate and fidelity predictions inherit whatever the source
paper's closed forms get wrong. Every result here should be read as a
statement about the model, not a prediction about a network.
