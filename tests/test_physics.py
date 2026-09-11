"""Physics checks against numbers published in the source paper."""

from __future__ import annotations

import math

import pytest

from qrp import physics


class TestLinkSuccess:
    def test_zero_length_is_lossless(self):
        assert physics.link_success(0.0, 0.2) == pytest.approx(1.0)

    def test_paper_form_matches_alpha_002(self):
        """The paper writes p_l(l) = 10^(-0.02 l) for 0.2 dB/km."""
        for length in (10.0, 50.0, 100.0, 300.0):
            assert physics.link_success(length, 0.2) == pytest.approx(
                10.0 ** (-0.02 * length)
            )

    def test_oband_is_lossier_than_cband(self):
        assert physics.link_success(100.0, 0.35) < physics.link_success(100.0, 0.2)

    def test_vault_single_span_table(self):
        """Vault note 37 tabulates transmission at 0.35 dB/km."""
        expected = {30.0: 0.089, 50.0: 0.018, 100.0: 0.00032}
        for length, want in expected.items():
            got = physics.link_success(length, 0.35)
            assert got == pytest.approx(want, rel=0.05)

    def test_efficiencies_are_squared(self):
        bare = physics.link_success(50.0, 0.2)
        with_eff = physics.link_success(50.0, 0.2, eta_emission=0.5, eta_detection=0.5)
        assert with_eff == pytest.approx(bare * 0.25 * 0.25)

    def test_inverse_round_trips(self):
        length = physics.link_length_for_success(0.01, 0.35)
        assert physics.link_success(length, 0.35) == pytest.approx(0.01)

    def test_negative_length_rejected(self):
        with pytest.raises(ValueError):
            physics.link_success(-1.0, 0.2)


class TestFidelity:
    def test_werner_mu_round_trip(self):
        for fidelity in (0.5, 0.75, 0.95, 1.0):
            mu = physics.werner_mu(fidelity)
            assert (1.0 + 3.0 * mu) / 4.0 == pytest.approx(fidelity)

    def test_single_hop_returns_link_fidelity(self):
        """With h = 1 there are no swaps, so F_e2e must equal F_L."""
        for fidelity in (0.6, 0.8, 0.95, 0.998):
            assert physics.e2e_fidelity(fidelity, 1) == pytest.approx(fidelity)

    def test_four_hop_reference_value(self):
        """The headline check.

        Vault note on this paper: "At their default F_L = 0.95, a four-link
        path already delivers only F_e2e ~ 0.82."

        Eq. (5) with perfect gates gives 1/4 + 3/4 * ((4*0.95-1)/3)^4
        = 0.25 + 0.75 * (0.9333...)^4 = 0.819126, which rounds to the 0.82
        recorded in the vault.
        """
        assert physics.e2e_fidelity(0.95, 4) == pytest.approx(0.819126, abs=1e-5)

    def test_decays_toward_quarter(self):
        """Eq. (5) has a floor of 1/4 as the hop count grows.

        Convergence is geometric in mu = 0.9333, so it is slow: at 40 hops
        the fidelity is still 0.297. Two hundred hops gets within 1e-6.
        """
        values = [physics.e2e_fidelity(0.95, h) for h in (1, 2, 5, 10, 40, 200)]
        assert values == sorted(values, reverse=True)
        assert values[-2] == pytest.approx(0.2975, abs=1e-3)
        assert values[-1] == pytest.approx(0.25, abs=1e-3)

    def test_perfect_hardware_stays_perfect(self):
        assert physics.e2e_fidelity(1.0, 12) == pytest.approx(1.0)

    def test_imperfect_gates_reduce_fidelity(self):
        perfect = physics.e2e_fidelity(0.95, 4)
        noisy = physics.e2e_fidelity(0.95, 4, gate_fidelity=0.98, measurement_fidelity=0.95)
        assert noisy < perfect

    def test_zero_hops_rejected(self):
        with pytest.raises(ValueError):
            physics.e2e_fidelity(0.95, 0)


class TestRate:
    def test_single_hop_has_no_swap_penalty(self):
        assert physics.e2e_rate(0.5, 1, 10, 0.01) == pytest.approx(10 * 0.01)

    def test_each_swap_costs_a_factor_qs(self):
        one = physics.e2e_rate(0.5, 1, 1, 1.0)
        three = physics.e2e_rate(0.5, 3, 1, 1.0)
        assert three == pytest.approx(one * 0.25)

    def test_generation_rate_scales_linearly(self):
        base = physics.e2e_rate(0.5, 3, 4, 0.02, generation_rate_hz=1.0)
        scaled = physics.e2e_rate(0.5, 3, 4, 0.02, generation_rate_hz=1000.0)
        assert scaled == pytest.approx(base * 1000.0)

    def test_approximation_ratio(self):
        assert physics.rate_approximation_ratio(64, 0.5) == pytest.approx(32.0)


class TestSwapWernerOverride:
    def test_override_matches_explicit_computation(self):
        explicit = physics.e2e_fidelity(
            0.96, 5, gate_fidelity=0.986, measurement_fidelity=0.946
        )
        factor = 0.986 * (4 * 0.946**2 - 1) / 3
        via_override = physics.e2e_fidelity(0.96, 5, swap_werner=factor)
        assert via_override == pytest.approx(explicit)

    def test_override_one_reproduces_perfect_gates(self):
        assert physics.e2e_fidelity(0.96, 5, swap_werner=1.0) == pytest.approx(
            physics.e2e_fidelity(0.96, 5)
        )

    def test_override_beats_gate_arguments(self):
        """When both are given, the override wins."""
        a = physics.e2e_fidelity(0.96, 5, gate_fidelity=0.5, swap_werner=1.0)
        assert a == pytest.approx(physics.e2e_fidelity(0.96, 5))


class TestCoordinatedRate:
    def test_multiplexed_success_limits(self):
        assert physics.link_success_multiplexed(0.01, 1) == pytest.approx(0.01)
        assert physics.link_success_multiplexed(0.5, 10**6) == pytest.approx(1.0)
        # small-p regime reduces to W*p
        assert physics.link_success_multiplexed(1e-5, 100) == pytest.approx(
            1e-3, rel=1e-3
        )

    def test_expected_rounds_single_link(self):
        """One link with success P waits about 1/P rounds."""
        assert physics.expected_rounds_all_links([1e-3]) == pytest.approx(
            1e3, rel=0.02
        )

    def test_expected_rounds_harmonic_limit(self):
        """h identical rare links wait about H_h / P rounds."""
        h, prob = 4, 1e-3
        harmonic = sum(1.0 / k for k in range(1, h + 1))
        got = physics.expected_rounds_all_links([prob] * h)
        assert got == pytest.approx(harmonic / prob, rel=0.02)

    def test_expected_rounds_certain_links_are_free(self):
        assert physics.expected_rounds_all_links([1.0, 1.0]) == pytest.approx(1.0)

    def test_coordinated_reduces_to_cadence_at_high_wp(self):
        """When links always succeed, rate is R_att * q^(h-1)."""
        rate = physics.e2e_rate_coordinated(0.9, 3, 10**6, [0.5, 0.5, 0.5], 1000.0)
        assert rate == pytest.approx(1000.0 * 0.9**2, rel=0.05)

    def test_coordinated_below_paper_model_everywhere(self):
        """The coordinated model must sit below Eq. (2)'s pipelined rate."""
        import random

        rng = random.Random(7)
        for _ in range(25):
            hops = rng.randint(1, 12)
            width = rng.choice([1, 4, 16, 64, 256])
            probs = [10 ** rng.uniform(-5, -0.5) for _ in range(hops)]
            q_s = rng.uniform(0.5, 0.95)
            paper = physics.e2e_rate(q_s, hops, width, min(probs), 1.0)
            coordinated = physics.e2e_rate_coordinated(q_s, hops, width, probs, 1.0)
            assert coordinated <= paper * 1.02

    def test_coordinated_monotone_in_width(self):
        probs = [1e-4, 5e-4, 2e-4]
        rates = [
            physics.e2e_rate_coordinated(0.7, 3, w, probs, 1.0)
            for w in (1, 8, 64, 512)
        ]
        assert rates == sorted(rates)


class TestExtRate:
    def test_single_hop_is_exactly_w_times_p(self):
        """With no swaps, EXT is the expectation of one binomial: W * p."""
        assert physics.e2e_rate_ext(0.5, 1, 64, [0.01]) == pytest.approx(0.64)
        assert physics.e2e_rate_ext(0.5, 1, 100, [1e-4]) == pytest.approx(0.01)

    def test_converges_to_paper_model_at_high_wp(self):
        """In Eq. (2)'s own regime the two formulas agree to a few percent."""
        probs = [0.5, 0.5, 0.5]
        ext = physics.e2e_rate_ext(0.8, 3, 2048, probs, 1.0)
        paper = physics.e2e_rate(0.8, 3, 2048, 0.5, 1.0)
        assert ext == pytest.approx(paper, rel=0.03)

    def test_below_paper_model_everywhere(self):
        import random

        rng = random.Random(11)
        for _ in range(20):
            hops = rng.randint(1, 10)
            width = rng.choice([1, 8, 64, 256])
            probs = [10 ** rng.uniform(-5, -0.3) for _ in range(hops)]
            q_s = rng.uniform(0.5, 0.95)
            ext = physics.e2e_rate_ext(q_s, hops, width, probs, 1.0)
            paper = physics.e2e_rate(q_s, hops, width, min(probs), 1.0)
            assert ext <= paper * (1.0 + 1e-9)

    def test_collapses_at_low_wp_multi_hop(self):
        """At W*p << 1 the same-round requirement multiplies small numbers."""
        ext = physics.e2e_rate_ext(1.0, 3, 10, [1e-3] * 3, 1.0)
        paper = physics.e2e_rate(1.0, 3, 10, 1e-3, 1.0)
        # E[min of 3 Binomial(10, 1e-3)] ~ P(all three fire at once) ~ (1e-2)^3
        assert ext < paper / 1e3

    def test_running_minimum_distribution_sums_to_one(self):
        """The recursion must conserve probability; check via h=2 exact case."""
        import numpy as np
        from scipy.stats import binom

        width, p1, p2 = 12, 0.13, 0.31
        # brute-force E[min(X1, X2)]
        pmf1 = binom.pmf(np.arange(width + 1), width, p1)
        pmf2 = binom.pmf(np.arange(width + 1), width, p2)
        expected = sum(
            min(i, j) * pmf1[i] * pmf2[j]
            for i in range(width + 1)
            for j in range(width + 1)
        )
        got = physics.e2e_rate_ext(1.0, 2, width, [p1, p2], 1.0)
        assert got == pytest.approx(expected, rel=1e-9)


class TestUtility:
    def test_matches_log2_definition(self):
        assert physics.utility(8.0, 1.0) == pytest.approx(math.log2(8.0 * 0.5))

    def test_classical_floor_is_worthless(self):
        assert physics.utility(1e6, 0.5) == -math.inf

    def test_below_floor_is_worthless(self):
        assert physics.utility(1e6, 0.4) == -math.inf

    def test_zero_rate_is_worthless(self):
        assert physics.utility(0.0, 0.99) == -math.inf

    def test_increasing_in_both_arguments(self):
        base = physics.utility(10.0, 0.9)
        assert physics.utility(20.0, 0.9) > base
        assert physics.utility(10.0, 0.95) > base


class TestTiming:
    def test_propagation_delay(self):
        """1000 km of fibre at c/1.5 is 5 ms."""
        assert physics.tau_link(1000.0) == pytest.approx(5e-3)

    def test_repeater_deadline_is_a_round_trip(self):
        assert physics.tau_repeater_deadline(200.0) == pytest.approx(
            2.0 * physics.tau_link(200.0)
        )

    def test_e2e_is_sequential_sum_of_round_trips(self):
        lengths = [100.0, 200.0, 150.0]
        assert physics.tau_e2e(lengths) == pytest.approx(
            sum(2.0 * physics.tau_link(x) for x in lengths)
        )

    def test_surfnet_scale_matches_paper_cliff(self):
        """Sanity check on the 3.2 ms end-node coherence cliff.

        SURFnet user pairs sit at 200 to 250 km. A few hops over that
        distance should give an end-to-end time of a few milliseconds, which
        is the order of magnitude at which the paper reports infeasibility.
        """
        tau = physics.tau_e2e([80.0, 80.0, 90.0])
        assert 1e-3 < tau < 1e-2
