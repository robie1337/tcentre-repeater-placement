"""Closed-form physics for the repeater placement model.

Every formula here is taken from Pouryousef et al., "Resource Placement for
Rate and Fidelity Maximization in Quantum Networks", IEEE TQE 2024
(arXiv:2308.16264v3). Equation numbers in the docstrings are the paper's own.

Nothing in this module simulates anything. The placement model consumes an
end-to-end rate and an end-to-end fidelity per candidate path, and both have
closed forms, so there is no discrete-event simulator anywhere in this project.

Reference check (see tests/test_physics.py): with perfect gates and
F_L = 0.95, a four-link path gives F_e2e = 0.8189, which is the 0.82 quoted
in the vault notes for this paper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Speed of light in silica fibre, km/s. c / n with n ~ 1.5.
C_FIBRE_KM_PER_S = 2.0e5


# --------------------------------------------------------------------------
# Elementary link
# --------------------------------------------------------------------------

def link_success(
    length_km: float,
    alpha_db_per_km: float,
    eta_emission: float = 1.0,
    eta_detection: float = 1.0,
) -> float:
    """Per-attempt success probability of an elementary link. Paper Eq. (1).

    The paper writes p_l(l) = 10^(-alpha * l) with alpha = 0.02, which is
    0.2 dB/km expressed as 10^(-0.2 * l / 10). This function takes the
    attenuation in dB/km directly so that swapping the C band value (0.2)
    for the T centre O band value (0.35) is a single argument change.

    The two efficiency factors are not in the paper, which folds them into
    its dimensionless rate. They default to 1.0, which reproduces Eq. (1)
    exactly, and are exposed because the T centre hardware has measured
    emission and detection efficiencies.
    """
    if length_km < 0:
        raise ValueError(f"length_km must be non-negative, got {length_km}")
    transmission = 10.0 ** (-alpha_db_per_km * length_km / 10.0)
    return eta_emission**2 * eta_detection**2 * transmission


def link_length_for_success(
    p_target: float,
    alpha_db_per_km: float,
    eta_emission: float = 1.0,
    eta_detection: float = 1.0,
) -> float:
    """Inverse of :func:`link_success`. Used for sanity checks and figures."""
    if not 0.0 < p_target <= 1.0:
        raise ValueError(f"p_target must be in (0, 1], got {p_target}")
    prefactor = eta_emission**2 * eta_detection**2
    if p_target > prefactor:
        return 0.0
    return -10.0 * math.log10(p_target / prefactor) / alpha_db_per_km


# --------------------------------------------------------------------------
# Werner states and fidelity
# --------------------------------------------------------------------------

def werner_mu(fidelity: float) -> float:
    """Werner parameter mu_L from link fidelity F_L. Paper Eq. (4).

    The paper writes the link state as
        rho = mu_L |psi+><psi+| + (1 - mu_L) I/4
    with F_L = (1 + 3 mu_L) / 4, so mu_L = (4 F_L - 1) / 3.
    """
    return (4.0 * fidelity - 1.0) / 3.0


def e2e_fidelity(
    link_fidelity: float,
    hops: int,
    gate_fidelity: float = 1.0,
    measurement_fidelity: float = 1.0,
    swap_werner: float | None = None,
) -> float:
    """End-to-end fidelity after hops-1 swaps. Paper Eq. (5).

        F_e2e = 1/4 + 3/4 * (P_2 (4 eta^2 - 1) / 3)^(h-1) * ((4 F_L - 1)/3)^h

    P_2 is the two-qubit gate fidelity and eta the measurement fidelity.
    Both default to 1.0, the paper's own "gate and measurement fidelity
    approximately 1" assumption.

    ``swap_werner`` overrides the whole per-swap factor P_2 (4 eta^2 - 1)/3
    with a single number. That is how the swap-noise sweep works: rather
    than resolving which physical quantity eta is for a given platform, the
    combined factor is swept across a justified bracket.

    With h = 1 there are no swaps and this returns F_L exactly.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")

    mu_link = werner_mu(link_fidelity)
    if swap_werner is not None:
        mu_swap = swap_werner
    else:
        mu_swap = gate_fidelity * (4.0 * measurement_fidelity**2 - 1.0) / 3.0
    return 0.25 + 0.75 * (mu_swap ** (hops - 1)) * (mu_link**hops)


# --------------------------------------------------------------------------
# Rate
# --------------------------------------------------------------------------

def e2e_rate(
    swap_success: float,
    hops: int,
    width: int,
    p_min: float,
    generation_rate_hz: float = 1.0,
) -> float:
    """End-to-end entanglement rate. Paper Eq. (2).

        R_e2e(p, W) = q_s^(h-1) * W * p_min

    The paper's expression is dimensionless (ebits per attempt round).
    Multiplying by the attempt rate gives ebits per second, which is what
    the T centre generation-rate sweep varies. generation_rate_hz defaults
    to 1.0, reproducing the paper's dimensionless form.

    Valid where W * p_min >> 1; :func:`rate_approximation_ratio` reports how
    far a given case is from that regime.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    return generation_rate_hz * (swap_success ** (hops - 1)) * width * p_min


def rate_approximation_ratio(width: int, p_min: float) -> float:
    """W * p_min, the quantity the paper needs to be much greater than 1.

    Pouryousef inherits this approximation and so do we. The sweep records
    this per solve so that any result sitting outside the valid regime can
    be flagged rather than quietly reported.
    """
    return width * p_min


# --------------------------------------------------------------------------
# Rate, corrected for the low-success regime
# --------------------------------------------------------------------------
#
# Eq. (2) above is stated by the paper as valid where W * p_min >> 1. On the
# O-band network that condition fails everywhere (median W*p_min = 0.033),
# so the functions below implement a conservative alternative that stays
# meaningful at low success probability.
#
# Model: each link runs W multiplexed attempts per round, so the per-round
# link success is P = 1 - (1-p)^W, which reduces to W*p when W*p << 1 and
# saturates at 1. An end-to-end pair needs every link ready, so delivery
# waits for the slowest link: the expected number of rounds is E[max of h
# geometric variables]. A failed swap is assumed to force a full restart,
# which divides the rate by q_s^(h-1). Both choices are conservative, so
# this is a conservative reference model to compare with Eq. (2)'s optimistic
# pipelined model. Nothing here proves the physical rate lies between them.
# Exact citations for each component are recorded in the report; the
# geometric maximum is evaluated through the standard exponential
# order-statistics integral.


def link_success_multiplexed(p: float, width: int) -> float:
    """Per-round success of a W-fold multiplexed link: 1 - (1-p)^W.

    Computed via expm1/log1p so it stays accurate when p is tiny and W is
    large, which is exactly the regime that matters here.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if p == 1.0:
        return 1.0
    return -math.expm1(width * math.log1p(-p))


def expected_rounds_all_links(per_round_probs: Sequence[float]) -> float:
    """Expected rounds until every link has succeeded at least once.

    E[max of independent geometrics], evaluated by approximating each
    geometric with an exponential of rate lambda = -ln(1 - P) and
    integrating the survival function of the maximum:

        E[max] ~= integral of ( 1 - prod_e (1 - exp(-lambda_e t)) ) dt

    Exact in the small-P limit and within a round of the discrete answer
    elsewhere. For h identical links with small P this reduces to the
    classic harmonic-number result H_h / P.
    """
    import numpy as np

    lambdas = []
    for prob in per_round_probs:
        if not 0.0 < prob <= 1.0:
            raise ValueError(f"per-round probability must be in (0, 1], got {prob}")
        if prob >= 1.0 - 1e-12:
            continue  # completes on the first round; drops out of the max
        lambdas.append(-math.log1p(-prob))
    if not lambdas:
        return 1.0

    lam = np.asarray(lambdas)
    t_lo = 0.01 / lam.max()
    t_hi = 60.0 / lam.min()
    t = np.geomspace(t_lo, t_hi, 800)
    survival = 1.0 - np.prod(1.0 - np.exp(-np.outer(t, lam)), axis=1)
    return float(t_lo + np.trapezoid(survival, t))


def e2e_rate_coordinated(
    swap_success: float,
    hops: int,
    width: int,
    link_probs: Sequence[float],
    generation_rate_hz: float = 1.0,
) -> float:
    """Conservative end-to-end rate, valid at low W*p.

    One delivered pair per cycle of: wait for every link (the geometric
    maximum), then perform h-1 swaps, restarting everything if any fails.

        R = R_att * q_s^(h-1) / E[rounds until all links are ready]

    At high W*p this tends to R_att * q_s^(h-1) (one pair per round, no
    pipelining), which is below Eq. (2)'s multiplexed throughput; at low
    W*p it decays like W*p / H_h rather than pretending each round yields
    W*p pairs. The two formulas are reference approximations, not proven
    bounds on the physical rate.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    if len(link_probs) != hops:
        raise ValueError(f"expected {hops} link probabilities, got {len(link_probs)}")
    per_round = [link_success_multiplexed(p, width) for p in link_probs]
    rounds = expected_rounds_all_links(per_round)
    return generation_rate_hz * (swap_success ** (hops - 1)) / rounds


def e2e_rate_ext(
    swap_success: float,
    hops: int,
    width: int,
    link_probs: Sequence[float],
    generation_rate_hz: float = 1.0,
) -> float:
    """Q-CAST's EXT throughput metric (Shi & Qian, ACM SIGCOMM 2020).

    Exact expected end-to-end pairs per attempt round under a slotted model:
    each link fires Binomial(W, p_k) successes per round, the usable flow is
    the minimum across links, pairs do not carry over between rounds, and
    each delivered pair survives its h-1 swaps with probability q_s each.

        EXT = q_s^(h-1) * E[ min_k Binomial(W, p_k) ]

    computed by the paper's recursion for the distribution of the running
    minimum. Convention note: our swap exponent is h-1 (one Bell measurement
    per interior node), written q^h in Q-CAST's own notation where q also
    absorbs per-hop processing; the difference is a single factor of q_s.

    Properties worth knowing: with h = 1 this is exactly W * p (no swaps,
    expectation of a binomial); as W*p grows it converges to Eq. (2)'s
    q_s^(h-1) * W * p_min; at low W*p it correctly collapses, because the
    probability that every link fires in the same round becomes the product
    of small numbers. It is conservative for long-coherence memories, which
    could buffer pairs across rounds; the coordinated model above covers
    that reading. The two are reference approximations, not proven bounds.
    """
    import numpy as np
    from scipy.stats import binom

    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    if len(link_probs) != hops:
        raise ValueError(f"expected {hops} link probabilities, got {len(link_probs)}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    counts = np.arange(width + 1)
    running = binom.pmf(counts, width, link_probs[0])
    for p_k in link_probs[1:]:
        pmf = binom.pmf(counts, width, p_k)
        # Survival functions: P(X_k >= i) and P(running minimum > i).
        sf_link = np.concatenate((np.cumsum(pmf[::-1])[::-1], [0.0]))
        sf_running = np.concatenate((np.cumsum(running[::-1])[::-1], [0.0]))
        running = running * sf_link[:-1] + pmf * sf_running[1:]

    expected_min = float(np.dot(counts, running))
    return generation_rate_hz * (swap_success ** (hops - 1)) * expected_min


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def utility(rate: float, fidelity: float) -> float:
    """Per-pair utility. Paper Eqs. (6) and (7).

        U = log2(R_e2e * (F_e2e - 1/2))

    Returns -inf when the argument of the logarithm is non-positive, which
    happens when the fidelity has decayed to the classical floor of 1/2 or
    when the rate is zero. Callers treat -inf as "this path is unusable".
    """
    quality = fidelity - 0.5
    if quality <= 0.0 or rate <= 0.0:
        return -math.inf
    return math.log2(rate * quality)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def tau_link(length_km: float, c_fibre_km_per_s: float = C_FIBRE_KM_PER_S) -> float:
    """One-way optical propagation delay over a link, in seconds.

    The paper defines tau_l(l_uv) = l_uv / c. This is propagation only: it
    does not divide by the success probability, so it is a floor on the
    link establishment time rather than the expected time to success.
    """
    return length_km / c_fibre_km_per_s


def tau_e2e(
    link_lengths_km: Sequence[float],
    c_fibre_km_per_s: float = C_FIBRE_KM_PER_S,
) -> float:
    """End-to-end entanglement establishment time, in seconds.

    RECONSTRUCTION, NOT VERBATIM. The paper states that tau_e2e accounts for
    classical message exchange between consecutive repeaters under the
    sequential protocol of its Section II-A, but the abstract and HTML
    rendering do not give the closed form. We use the natural sequential
    reading, a heralding round trip per link taken in sequence:

        tau_e2e = sum over links of 2 * tau_l(l_i)

    The factor of two matches the paper's own repeater-deadline constraint
    Eq. (13), which is 2 * tau_l(l_e) <= T_RM.

    Check this against Section II-A of the paper before the results go into
    the report. If the paper takes the maximum rather than the sum, change
    this one function; nothing else in the project needs to move.
    """
    return sum(2.0 * tau_link(length, c_fibre_km_per_s) for length in link_lengths_km)


def tau_repeater_deadline(
    max_link_length_km: float,
    c_fibre_km_per_s: float = C_FIBRE_KM_PER_S,
) -> float:
    """Round-trip time on the longest link of a path. Paper Eq. (13) left side.

    A repeater must hold its half of an ebit for at least this long, so this
    is what the repeater memory coherence time T_RM has to cover.
    """
    return 2.0 * tau_link(max_link_length_km, c_fibre_km_per_s)
