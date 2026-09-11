"""Memory waiting time, and memory decay, while the links of a path come up.

The source paper gates memory coherence with two hard cut-offs, Eqs. (13)
and (14), on propagation delay. Neither counts the time a memory holds one
link's pair while the other links on the path keep failing. In the regime
Eq. (2) assumes, W * p_min >> 1, that wait is a round or two. At 1326 nm
W * p_min is about 0.03 and the wait runs to tens of rounds.

These functions model that wait. They back ``NetworkConfig.coherence_model``
and were first written inside scripts W5 and W6.

Model
-----
Link e succeeds in an attempt round with P_e = 1 - (1 - p_e)^W. A round lasts
1 / generation rate on the "source" clock, or max(1 / generation rate,
2 l_e / c) on the "heralded" clock, where a link cannot retry before its
herald returns. The time until link e is ready is taken as exponential with
rate lambda_e = -ln(1 - P_e) / tau_e, a continuous stand-in for the geometric
number of rounds, and the links are independent.

Storage time is E[max_e T_e] - E[min_e T_e] + tau_e2e: how long the first
ready pair waits for the last, plus the paper's classical delay.

Decay multiplies the Werner parameter by exp(-t / T2) for every stored half of
a pair, using the idle coherence time, which is optimistic if the memory
dephases faster while its neighbour keeps attempting. How long each pair is
stored depends on the swap schedule, which is not modelled; two bounds are
offered instead (``storage_decay_factor``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .physics import C_FIBRE_KM_PER_S, link_success_multiplexed

WAITING_CLOCKS = ("heralded", "source")
DECAY_BOUNDS = ("optimistic", "pessimistic")

#: Rate used for a link that succeeds every round, so it is ready at once.
CERTAIN_LINK_RATE = 1e15


def link_ready_rates(
    link_lengths_km: Sequence[float],
    link_probs: Sequence[float],
    width: int,
    generation_rate_hz: float,
    clock: str = "heralded",
    c_fibre_km_per_s: float = C_FIBRE_KM_PER_S,
) -> np.ndarray:
    """Exponential rate, per second, at which each link of a path becomes ready."""
    if clock not in WAITING_CLOCKS:
        raise ValueError(f"unknown clock {clock!r}; choose from {WAITING_CLOCKS}")
    rates = []
    for length, p in zip(link_lengths_km, link_probs):
        per_round = link_success_multiplexed(p, width)
        tau = 1.0 / generation_rate_hz
        if clock == "heralded":
            tau = max(tau, 2.0 * length / c_fibre_km_per_s)
        rates.append(-math.log1p(-per_round) / tau if per_round < 1.0 else CERTAIN_LINK_RATE)
    return np.asarray(rates, dtype=float)


def expected_max_exponential(rates: Sequence[float]) -> float:
    """E[max] of independent exponential times, by integrating the survival function.

    The integral runs on a 4000-point log grid from 1e-4 of the fastest mean
    to 60 times the slowest, which leaves a truncation error below e^-60.
    """
    lam = np.asarray(rates, dtype=float)
    if len(lam) == 1:
        return float(1.0 / lam[0])
    t = np.geomspace(1e-4 / lam.max(), 60.0 / lam.min(), 4000)
    survival = 1.0 - np.prod(1.0 - np.exp(-np.outer(t, lam)), axis=1)
    return float(t[0] + np.trapezoid(survival, t))


def expected_storage_time(rates: Sequence[float], tau_e2e_s: float) -> float:
    """E[max T] - E[min T] + tau_e2e, in seconds."""
    lam = np.asarray(rates, dtype=float)
    return expected_max_exponential(lam) - 1.0 / float(lam.sum()) + tau_e2e_s


def unit_exponential_samples(n_samples: int, n_links: int, seed: int) -> np.ndarray:
    """Shared unit-mean exponential samples, one column per link.

    Dividing column e by lambda_e gives samples of T_e. Sharing one array
    across paths and widths makes the decay factors reproducible and keeps
    comparisons between paths free of independent sampling noise. A path
    uses the first h columns, so its factor does not depend on how many
    columns the array has.
    """
    return np.random.default_rng(seed).exponential(size=(n_samples, n_links))


def storage_decay_factor(
    rates: Sequence[float],
    tau_e2e_s: float,
    t2_s: float,
    samples: np.ndarray,
    bound: str,
) -> float:
    """Expected factor on the Werner parameter from decay while links wait.

    ``optimistic``: only one pair waits, from the first link ready to the
    last. Any swap schedule stores at least that much. ``pessimistic``: every
    link's pair waits until the last link is ready, which covers schedules
    that swap no later than that. Both also decay the final pair over the
    classical delay tau_e2e. Each stored half contributes exp(-t / T2), hence
    the factor of two.
    """
    if bound not in DECAY_BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; choose from {DECAY_BOUNDS}")
    lam = np.asarray(rates, dtype=float)
    hops = len(lam)
    if samples.shape[1] < hops:
        raise ValueError(f"samples have {samples.shape[1]} columns, the path has {hops} links")
    times = samples[:, :hops] / lam
    t_max = times.max(axis=1)
    if bound == "optimistic":
        held = t_max - times.min(axis=1)
    else:
        held = hops * t_max - times.sum(axis=1)
    tail = math.exp(-tau_e2e_s / t2_s)
    return float(np.mean(np.exp(-2.0 * held / t2_s))) * tail


def decayed_fidelity(fidelity: float, factor: float) -> float:
    """Werner fidelity after scaling its Werner parameter by ``factor``."""
    mu = (4.0 * fidelity - 1.0) / 3.0 * factor
    return (3.0 * mu + 1.0) / 4.0
