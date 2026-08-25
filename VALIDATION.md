# Validation and corrections

Nothing in this project is worth reading until the model reproduces results
someone else published. This file records what was checked, what passed, and
what turned out to be wrong. The errors include several of this project's
own.

## Reproductions of the source paper

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

## Checks on our own assumptions

**The rate equation leaves its own validity regime.** Eq. (2) of the source
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

## The largest remaining gap

This is a planning model. It has not been validated against a discrete-event
simulator, so its rate and fidelity predictions inherit whatever the source
paper's closed forms get wrong. Every result here should be read as a
statement about the model, not a prediction about a network.
