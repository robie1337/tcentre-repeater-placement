"""Waiting time and memory decay, checked against independent calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from qrp import physics, waiting

RATES = np.array([30.0, 55.0, 12.0, 80.0])  # per second


def test_expected_max_of_identical_exponentials_is_harmonic():
    lam, n = 40.0, 6
    harmonic = sum(1.0 / k for k in range(1, n + 1))
    assert waiting.expected_max_exponential([lam] * n) == pytest.approx(harmonic / lam, rel=1e-4)


def test_storage_time_matches_monte_carlo():
    tau = 3e-3
    times = np.random.default_rng(7).exponential(size=(400_000, len(RATES))) / RATES
    simulated = float(np.mean(times.max(axis=1) - times.min(axis=1))) + tau
    assert waiting.expected_storage_time(RATES, tau) == pytest.approx(simulated, rel=1e-2)


def test_single_link_waits_only_for_the_classical_delay():
    assert waiting.expected_storage_time([25.0], 2e-3) == pytest.approx(2e-3)


def _exact_pessimistic(rates, tau, t2):
    """E[exp(-a (h T_max - sum T))] with a = 2 / T2, as a one-dimensional integral.

    Conditioning on link k being last, at time t, every other link e is ready
    before t and contributes E[exp(-a (t - T_e)); T_e < t]
    = lambda_e (exp(-lambda_e t) - exp(-a t)) / (a - lambda_e).
    """
    a = 2.0 / t2

    def integrand(t):
        total = 0.0
        for k, lam_k in enumerate(rates):
            term = lam_k * math.exp(-lam_k * t)
            for e, lam_e in enumerate(rates):
                if e != k:
                    term *= lam_e * (math.exp(-lam_e * t) - math.exp(-a * t)) / (a - lam_e)
            total += term
        return total

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return value * math.exp(-tau / t2)


def test_pessimistic_decay_matches_exact_integral():
    tau, t2 = 4e-3, 0.112
    samples = waiting.unit_exponential_samples(200_000, 4, seed=11)
    estimate = waiting.storage_decay_factor(RATES, tau, t2, samples, "pessimistic")
    assert estimate == pytest.approx(_exact_pessimistic(RATES, tau, t2), rel=1e-2)


def test_optimistic_bound_is_never_below_pessimistic():
    samples = waiting.unit_exponential_samples(4000, 20, seed=3)
    for t2 in (0.01, 0.112, 1.0):
        opt = waiting.storage_decay_factor(RATES, 1e-3, t2, samples, "optimistic")
        pess = waiting.storage_decay_factor(RATES, 1e-3, t2, samples, "pessimistic")
        assert 0.0 < pess <= opt <= 1.0


def test_bounds_coincide_for_two_links():
    samples = waiting.unit_exponential_samples(4000, 20, seed=3)
    opt = waiting.storage_decay_factor(RATES[:2], 1e-3, 0.05, samples, "optimistic")
    pess = waiting.storage_decay_factor(RATES[:2], 1e-3, 0.05, samples, "pessimistic")
    assert opt == pytest.approx(pess)


def test_one_link_decays_only_over_the_classical_delay():
    samples = waiting.unit_exponential_samples(100, 20, seed=3)
    for bound in waiting.DECAY_BOUNDS:
        factor = waiting.storage_decay_factor([25.0], 2e-3, 0.1, samples, bound)
        assert factor == pytest.approx(math.exp(-2e-3 / 0.1))


def test_infinite_coherence_does_not_decay():
    samples = waiting.unit_exponential_samples(100, 20, seed=3)
    assert waiting.storage_decay_factor(RATES, 1e-3, float("inf"), samples, "pessimistic") == 1.0


def test_decayed_fidelity_limits():
    assert waiting.decayed_fidelity(0.96, 1.0) == pytest.approx(0.96)
    assert waiting.decayed_fidelity(0.96, 0.0) == pytest.approx(0.25)


def test_heralded_clock_is_never_faster_than_the_source_clock():
    lengths, probs = [80.0, 240.0], [physics.link_success(80.0, 0.35), physics.link_success(240.0, 0.35)]
    heralded = waiting.link_ready_rates(lengths, probs, 10, 20e3, "heralded")
    source = waiting.link_ready_rates(lengths, probs, 10, 20e3, "source")
    assert np.all(heralded <= source)
    # At 20 kHz a round is 50 microseconds, shorter than either herald round trip.
    assert heralded[1] < source[1]


def test_unknown_options_raise():
    samples = waiting.unit_exponential_samples(10, 4, seed=1)
    with pytest.raises(ValueError):
        waiting.link_ready_rates([10.0], [0.5], 1, 1e3, clock="sundial")
    with pytest.raises(ValueError):
        waiting.storage_decay_factor(RATES, 0.0, 0.1, samples, "median")
    with pytest.raises(ValueError):
        waiting.storage_decay_factor(RATES, 0.0, 0.1, samples[:, :2], "optimistic")


def _event_swap_asap_qubit_time(times):
    """Walk the chain node by node: each interior node swaps once both its links exist."""
    t_max = max(times)
    total = 0.0
    for node in range(1, len(times)):
        swap = max(times[node - 1], times[node])
        total += (swap - times[node - 1]) + (swap - times[node])
    return total + (t_max - times[0]) + (t_max - times[-1])


def test_swap_asap_matches_an_event_walk():
    samples = waiting.unit_exponential_samples(3000, 20, seed=5)
    t2, tau = 0.05, 1e-3
    times = samples[:, : len(RATES)] / RATES
    expected = np.mean([math.exp(-_event_swap_asap_qubit_time(row) / t2) for row in times])
    expected *= math.exp(-tau / t2)
    assert waiting.storage_decay_factor(RATES, tau, t2, samples, "swap_asap") == pytest.approx(expected, rel=1e-10)


def test_swap_asap_lies_between_the_bounds():
    samples = waiting.unit_exponential_samples(4000, 20, seed=3)
    rates = np.array([30.0, 55.0, 12.0, 80.0, 20.0, 45.0, 9.0])
    for t2 in (0.01, 0.112, 1.0):
        opt, asap, pess = (waiting.storage_decay_factor(rates, 1e-3, t2, samples, bound)
                           for bound in ("optimistic", "swap_asap", "pessimistic"))
        assert pess <= asap + 1e-15
        assert asap <= opt + 1e-15


def test_swap_asap_equals_both_bounds_for_two_links():
    samples = waiting.unit_exponential_samples(4000, 20, seed=3)
    values = [waiting.storage_decay_factor(RATES[:2], 1e-3, 0.05, samples, bound)
              for bound in waiting.DECAY_BOUNDS]
    assert values == pytest.approx([values[0]] * 3)


def test_midpoint_herald_halves_the_round_on_long_links():
    length = 240.0
    probs = [physics.link_success(length, 0.35)]
    # At 20 kHz one attempt is 50 microseconds, shorter than either herald delay.
    midpoint = waiting.link_ready_rates([length], probs, 10, 20e3, "heralded_midpoint")
    far = waiting.link_ready_rates([length], probs, 10, 20e3, "heralded")
    assert midpoint[0] == pytest.approx(2.0 * far[0])
