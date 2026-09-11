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
| Backbone length below which no repeater is placed | about 40 km | 38.5 km |
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
variables. The validation dumbbells fit, and all three solvers agree on them.
The full CA9 model no longer does. Under the legacy path generator it was 809
variables at the projected preset, and all three solvers returned 200.860328.
Under the default generator it is 4,454 variables at the same preset and
2,421 at midrange. HiGHS returns the same 200.860328 on the larger model, but
both free licences refuse it: CPLEX Community reports `too_large`, and so does
the size-limited Gurobi licence. Cross-solver agreement on the full default
model therefore needs a full Gurobi or CPLEX licence. With an academic Gurobi
licence it holds: solved exactly, Gurobi returns 200.860328 on the 4,454
variable projected model, as HiGHS does, and 132.174790 at a budget of 20
repeaters, again matching HiGHS. On this machine Gurobi took 0.4 s and 3.7 s
for those two instances where HiGHS took minutes.

`w9_solver_agreement.py` checks agreement on models that fit the free
licences and still use the default path set: CA9 with one width (up to 611
variables) and with three widths (up to 1,709), plus the legacy model (809).
It covers the two presets and 10 Latin hypercube points, three budgets, three
rate models and two coherence models, 648 models in all. HiGHS and Gurobi
proved all 594 models with candidates optimal, and wherever two solvers both
reported optimal their objectives agree to within 1.5e-9 relative. CPLEX
Community refused the 54 three-width models as too large. No solver disagreed
with another on any objective.

That run exposed two faults in the solver layer, both since fixed and
tested. The solves were requested as exact, but each backend passed the gap on
only when it was positive, so a gap of zero left every solver at its own
default of 1e-4. And CPLEX names status 102 "optimal_tolerance" in this
release, which the layer did not recognise, so 8 CPLEX solves that had in fact
reached the same optimum as HiGHS and Gurobi were recorded as stopped at a
limit. The agreement above therefore holds at a 1e-4 gap. An exact rerun is
still to be done.

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
tested against the assumption: the paper's own, a memoryless model, and a
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

## Code review, September 2026

An anonymous review of the repository found that two parts of the code did
not do what the documentation said. Both are fixed and tested. Rerunning the
affected results began on 11 September 2026. So far the preset CA9 solves,
the gate-noise and single-case checks, W5 and W6 have been rerun. W4, the
site-spacing check and the Sobol sweeps have not, and the answers below that
would depend on them are left out until they are.

### Solve outcomes

The solver layer distinguished optimal, infeasible and error, and nothing
else. A solve that stopped at its time limit had no status of its own, so it
could carry the -50 utility the paper reserves for infeasible instances, and
the Sobol analysis would have counted it as hardware that cannot work. Every
backend now reports one of the seven statuses listed in MODEL.md, with gap and
bound kept separately, and `tests/test_solver_status.py` checks the mapping
for each backend.

The committed result files record every solve as optimal or infeasible,
apart from 18 W4 serve-first solves recorded as errors and left out of that
analysis. The rerun turned up one more gap: the size-limited Gurobi licence
raised an exception on a large model instead of returning a status. It now
reports `too_large`, as CPLEX Community already did.

### Candidate paths

The old generator ran a dynamic program over (hop count, node) for the best
worst link and the shortest total length, then dropped any path another path
beat on both. The documentation called that exact. It was not, for three
reasons, each now covered by `tests/test_candidate_paths.py`:

- Ties were discarded. Each state kept one predecessor, so equal paths
  through different repeaters collapsed to one. On CA9 that discarded 16,764
  tied alternatives.
- The program ran over walks, and a walk that revisited a node was rejected
  only after reconstruction. On CA9, 178 of 322 (pair, hop count) states were
  lost that way.
- Pruning is not safe when repeaters are shared. In the test network one pair
  can take a two-hop route through its own repeater or a three-hop route
  through two repeaters that a second pair also needs. With a budget of two
  repeaters the old generator keeps only the two-hop route and serves one
  pair. Exhaustive enumeration serves both.

The default generator keeps the Pareto frontier over hop count and worst
link, with ties and one hop of slack, plus the shortest simple paths by total
length and by summed loss (MODEL.md). It makes no claim to be exact. On 24
small random networks, 12 seeds under two rate models, it reaches the optimum
found by enumerating every simple path. Exhaustive enumeration cannot replace
it on CA9, where at least one pair has more than 200,000 simple paths within
20 hops.

Rerunning at finer site spacing exposed a problem in the new generator
itself. Its two weighted families ranked simple paths by weight and dropped
those over the hop limit afterwards, from the first 120. The cheapest paths by
loss use many short links. On the 40 km grid all 120 were over 20 hops, 31 to
40 on the longest pairs, so the loss family kept nothing, and at 80 km it kept
4 of 6 on the longest pair. The hop limit now acts inside Yen's search, and
`tests/test_hop_limited_paths.py` checks the result against brute force on
random graphs. At 80 km the fix changes 85 of the 552 paths: 41 added, 44
removed.

| Site spacing | Legacy paths | Default paths | Default generation time |
|---|---|---|---|
| 80 km | 94 | 549 | 2.4 s |
| 40 km | 129 | 846 | 16.7 s |
| 20 km | 162 | 1,072 | 97 s |

Times are wall-clock on the development machine with other jobs running.
Before the fix, generation at 20 km had not finished after 19 minutes.

### What the fixes changed

On the twelve preset CA9 solves (two hardware presets, three budgets, Eq. (2)
and buffered rates), the default generator changes the utility of one:
projected hardware, unlimited budget, buffered rates, up 0.505 bits with the
same pairs served and the same sites.

W6 keeps every pair and repeater count in all twelve of its cases. The mean
fidelities under decay move by at most 0.011, and the number of pairs pushed
to F <= 1/2 moves by one in each cell of the T2 = 112 ms table in FINDINGS.md.
The routes themselves did change. The paper's objective cannot tell apart
routes with the same hop count and worst link, but their waiting times
differ: in the midrange published plan at T2 = 10 ms, with the same 12 pairs
and 54 repeaters, median storage time over T2 went from 2.73 to 3.36. A
waiting-time result that depends on which tie the solver returns is an
argument for putting waiting time inside the optimisation rather than
scoring it afterwards.

The gate-noise check gives the same numbers as before. The single-case run
matches in eight of its nine cases. The exception is projected hardware with
a budget of 25 repeaters, which now serves 15 pairs instead of 14 (utility
156.38 against 149.09) with a different set of 25 sites. That is the larger
candidate set finding a better plan, not a change in the model.

W5, rerun with a 900 s limit and every solve optimal, gives the same shares
of published plans with an unreachable pair as before, within one point in
every cell: 80, 71 and 68 per cent under the strictest gate at unlimited,
20 and 10 repeaters. One preset did change. With 2048 memories, projected
hardware and a budget of 10, the published plan now serves 11 pairs instead
of 5 with the same 10 repeaters, because the larger candidate set offers
routes with fewer, longer links. Nine of those 11 pairs fail the waiting
gate, against one of 5 before, and the waiting-aware plan serves 5 either
way. The published objective rewards those routes because it cannot see
waiting time.

### Answers to the review's questions

1. **Can the old path pruning be proven exact for the full MILP?** No.
   Repeater memory and the repeater budget are shared between pairs, and a
   per-path dominance argument ignores them.

2. **What demonstrates the failure?** The shared-repeater network in
   `tests/test_candidate_paths.py`. The old generator keeps one route per pair
   and serves one of two pairs. Enumerating every simple path serves both.

3. **Which candidate strategy gives the best trade-off?** The default
   generator described above. It matches exhaustive enumeration on 24 small
   random networks and builds 549 CA9 paths in about two seconds. The price
   is a model about five times larger, 4,454 variables instead of 809, which
   rules out the free Gurobi and CPLEX licences on CA9 and makes each sweep
   slower.

4. **Can CA9 support exhaustive simple-path enumeration?** No. At least one
   pair has more than 200,000 simple paths within 20 hops.

5. **Which solver statuses were misclassified?** Every outcome other than
   optimal and infeasible. A solve stopped at its time limit had no status of
   its own and could reach the model as infeasible, and the size-limited
   Gurobi licence raised an exception instead of reporting. None of the
   committed result files records such a solve, apart from the 18 W4 errors
   already excluded from that analysis.

6. **How much does waiting-aware coherence change the main results?** It
   depends on how waiting is modelled. As a hard cut-off, mean storage time no
   longer than T2, it makes 68 to 80 per cent of published plans contain an
   unreachable pair across the sampled hardware box, but at the measured
   nuclear T2 of 112 ms it removes no midrange pair and one projected pair.
   As gradual decay at 112 ms it matters at every budget tested. With the
   exact storage time of swap-as-soon-as-possible and a herald from a midpoint
   station, the projected plan's mean fidelity at unlimited budget falls from
   0.980 to 0.642, 9 of its 18 pairs end at or below 1/2, and a decay-aware
   plan serves 13 pairs (`w11_clock_schedule.py`). The effect depends on
   attempts waiting for their herald: without that wait the fidelity falls
   only to 0.905. Both the gate and the decay are now options of the model
   (`NetworkConfig.coherence_model`), but the Sobol ranking has not been
   recomputed under either.

8. **Which conclusions are robust to candidate-site spacing?** Coverage is.
   The network was re-solved at 80, 40 and 20 km site spacing, with a 900 s
   limit per solve. At a budget of 200 repeaters, pairs served are the same
   at every spacing: 12 at midrange, 18 at projected, and none with measured
   hardware. Utility is not: it rises 12 per cent at midrange and 29 per cent
   at projected from 80 to 20 km. So absolute utility is not quoted as though
   80 km were an optimised choice. Three budgeted projected solves at 40 and
   20 km did not prove optimality within the limit and are not counted.

9. **Which hardware conclusions remain with defensible gate and readout
   values?** No Bell-state measurement fidelity for T centres has been
   published, so there is no defensible single value to substitute. The 0.946
   readout figure was an erbium result and is now quarantined in
   `qrp.legacy`. The model holds gate and measurement fidelity at 1.0 and
   sweeps the combined swap quality across [0.71, 0.997] instead. Across that
   bracket coherence stayed first (total-effect index 0.82 to 0.91) and swap
   quality never exceeded 0.10. Those Sobol runs used the legacy path
   generator and have not been rerun.

## Second code review, September 2026

A second anonymous review, of commit `2012857`, accepted the path and
status fixes above and raised what they left open. Each item below is fixed
and covered by a test.

- **Several paths per pair.** `NetworkConfig` allowed more than one path per
  pair, and the model would then have counted each chosen path as a served
  pair and added their utilities. Every committed result used one path per
  pair, so none was affected. Any other value now raises.
- **Impossible hardware.** `Hardware` accepted fidelities outside [0, 1],
  negative attenuation and non-positive rates or coherence times. It now
  rejects them.
- **The SURFnet loader.** It matched end-node names by substring, took the
  first match, and skipped a name it could not find. It now prefers an exact
  label, accepts a single partial match, and raises on a missing or
  ambiguous name. The coherence cliff check itself was not rerun, because
  the authors' topology file is not in this repository.
- **Provenance.** Sweep rows now record the coherence model, site spacing,
  link cap, hop cap and width grid alongside the rate model, path strategy,
  solver and gap.
- **Ties.** Path costs within 12 significant figures are treated as equal,
  and equal costs go to fewer hops, then to node order. That policy is now
  tested in both directions.
- **Wording.** The rate models are no longer called a floor and a ceiling,
  and the three T centre presets are labelled as composites of separate
  experiments rather than as single devices.

The largest change is that waiting-aware coherence is now part of the model.
`NetworkConfig.coherence_model` offers the waiting gate and the two decay
bounds (MODEL.md), built on `qrp.waiting`, and W5 and W6 plan through it
instead of replacing `build_candidates` while they run. Rerun that way, W5
reproduces every row of its four result files exactly, and W6 reproduces
every pair count, repeater count and fidelity it reported before.
The only change is the median ratio of storage time to T2, now analytic
rather than a Monte Carlo estimate, which moved by at most 0.02.

Two assumptions the review did not name, but a reader of the waiting-time
result would, have since been checked against the literature and in code.

- **The swap schedule.** The decay was bounded from two sides rather than
  computed. `coherence_model="decay_swap_asap"` now uses the exact storage
  time of swap-as-soon-as-possible with deterministic swaps, in closed form
  for links of unequal length, which lies between the two bounds on every
  sample tested. Kamin et al. (PRR 5, 023086, 2023) identify this schedule as
  the least-dephasing of the fast schemes. `w12_swap_event_check.py` checks
  it against an event simulation (MODEL.md gives the numbers). Failed swaps
  lower the delivered fidelity a little further than the model does.
- **The attempt clock.** A single communication qubit cannot start its next
  attempt before the last one's herald arrives, and the T centre
  demonstration (Afzal et al., arXiv:2406.01704) runs that way. The herald
  comes from the far node in the model's default, a round trip of `2 l / c`,
  but from a station at the link midpoint it takes `l / c`.
  `waiting_clock="heralded_midpoint"` adds that geometry, and
  `w11_clock_schedule.py` reports the result under all three clocks and
  schedules.

## What is not validated

This is a planning model. It has not been validated against a discrete-event
simulator, so its rate and fidelity predictions inherit whatever the source
paper's closed forms get wrong. Every result here should be read as a
statement about the model, not a prediction about a network.
