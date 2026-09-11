"""Hardware parameter sets and the sweep box.

Two presets matter:

POURYOUSEF_BASELINE reproduces the paper's own settings, and is what the
week 1 validation runs against. Every value here is the paper's.

TCENTRE_* covers the silicon T centre. The numbers come from the
vault reference notes 36 and 42, which trace back to Afzal et al. 2024
(arXiv:2406.01704) for rates and fidelities and Song et al. 2025
(Nature Nanotechnology) for coherence times.

The sweep box at the bottom is the four-parameter space week 2 explores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# --------------------------------------------------------------------------
# Parameter container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """One point in hardware parameter space.

    Attributes
    ----------
    alpha_db_per_km
        Fibre attenuation. 0.2 at 1550 nm (C band, what the literature
        assumes). 0.35 at 1326 nm (O band, where T centres emit).
    link_fidelity
        F_L, the Werner fidelity of a freshly generated elementary link.
    swap_success
        q_s, probability that a Bell state measurement succeeds. 0.5 is the
        linear-optical ceiling; matter qubits can exceed it.
    generation_rate_hz
        Attempt rate of the entanglement source. Scales the end-to-end rate.
    t_repeater_memory_s
        T_RM, coherence time of a repeater memory.
    t_endnode_memory_s
        T_EM, coherence time of an end-node memory.
    gate_fidelity, measurement_fidelity
        P_2 and eta in Eq. (5). 1.0 reproduces the paper's assumption.
    eta_emission, eta_detection
        Photon emission and detection efficiencies. 1.0 reproduces Eq. (1).
    """

    alpha_db_per_km: float = 0.2
    link_fidelity: float = 0.95
    swap_success: float = 0.5
    generation_rate_hz: float = 1.0
    t_repeater_memory_s: float = float("inf")
    t_endnode_memory_s: float = float("inf")
    gate_fidelity: float = 1.0
    measurement_fidelity: float = 1.0
    eta_emission: float = 1.0
    eta_detection: float = 1.0
    #: Optional override for the whole per-swap Werner factor
    #: P_2 (4 eta^2 - 1) / 3 in Eq. (5). None means compute it from the two
    #: fidelities above. Used by the swap-noise sensitivity sweep, which
    #: brackets the factor instead of resolving which physical quantity eta
    #: is for the T centre.
    swap_werner: float | None = None

    def __post_init__(self) -> None:
        for name in ("link_fidelity", "swap_success", "gate_fidelity", "measurement_fidelity",
                     "eta_emission", "eta_detection"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.swap_werner is not None and not 0.0 <= self.swap_werner <= 1.0:
            raise ValueError(f"swap_werner must be None or lie in [0, 1], got {self.swap_werner}")
        if not (math.isfinite(self.alpha_db_per_km) and self.alpha_db_per_km >= 0.0):
            raise ValueError(f"alpha_db_per_km must be finite and non-negative, got {self.alpha_db_per_km}")
        if not (math.isfinite(self.generation_rate_hz) and self.generation_rate_hz > 0.0):
            raise ValueError(f"generation_rate_hz must be finite and positive, got {self.generation_rate_hz}")
        for name in ("t_repeater_memory_s", "t_endnode_memory_s"):
            value = getattr(self, name)
            # Infinity is allowed: the paper imposes no coherence limit.
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

    def with_(self, **changes) -> "Hardware":
        """Return a copy with the named fields replaced."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return {
            "alpha_db_per_km": self.alpha_db_per_km,
            "link_fidelity": self.link_fidelity,
            "swap_success": self.swap_success,
            "generation_rate_hz": self.generation_rate_hz,
            "t_repeater_memory_s": self.t_repeater_memory_s,
            "t_endnode_memory_s": self.t_endnode_memory_s,
            "gate_fidelity": self.gate_fidelity,
            "measurement_fidelity": self.measurement_fidelity,
            "eta_emission": self.eta_emission,
            "eta_detection": self.eta_detection,
            "swap_werner": self.swap_werner,
        }


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

#: The paper's own defaults. Section IV: alpha = 0.02 (0.2 dB/km),
#: F_L = 0.95, q_s = 1/2, gate and measurement fidelity ~ 1, and no
#: coherence limits imposed in the synthetic experiments.
POURYOUSEF_BASELINE = Hardware(
    alpha_db_per_km=0.2,
    link_fidelity=0.95,
    swap_success=0.5,
    generation_rate_hz=1.0,
    t_repeater_memory_s=float("inf"),
    t_endnode_memory_s=float("inf"),
    gate_fidelity=1.0,
    measurement_fidelity=1.0,
)

# --------------------------------------------------------------------------
# A modelling decision that has to be made explicitly
# --------------------------------------------------------------------------
#
# Eq. (5) carries a per-swap factor P_2 (4 eta^2 - 1) / 3, where P_2 is the
# two-qubit gate fidelity and eta the measurement fidelity. Pouryousef sets
# both to approximately 1 and says so.
#
# The T centre gate fidelity is measured at 98.4 to 98.6 per cent (Afzal
# 2024). A single-shot readout figure of 94.6 per cent was long attributed
# here to Higginbottom 2022; it is an erbium-ion result, and qrp.legacy holds
# the details. Substituting the pair changes
# the per-swap factor from
# 1.0 to 0.848, which is severe: the number of hops a path can carry before
# its fidelity falls to the classical floor of 1/2 drops from 20 to 5 at
# F_L = 0.96, and from 59 to 7 at F_L = 0.998. Almost no route across the
# CA9 graph survives that.
#
# The presets below therefore keep gate and measurement fidelity at 1.0, so
# that the only differences from the paper's baseline are the four
# parameters this project actually claims to vary: attenuation, link
# fidelity, swap success and coherence time. Changing one thing at a time is
# what makes the comparison to their published results mean anything.
#
# The effect of the measured values is not ignored. It is measured
# separately by scripts/w1_gate_noise_check.py, which is the honest way to
# report something this consequential. Whether the T centre's single-shot
# readout fidelity is the right quantity for eta in Eq. (5) is still open:
# readout fidelity and Bell-measurement fidelity are not obviously the same
# number.

#: T centre as measured today. Afzal 2024: remote Bell fidelity 0.60(8) at
#: 0.09 to 1 Hz. Song et al. 2025 (arXiv:2504.15467, Nat. Nanotech.):
#: hydrogen nuclear T2 of 112(12) ms by echo, silicon nuclear 67(7) ms,
#: electron 0.41(2) ms. Independent group (Berkeley/LBNL), not the vendor.
#: A link
#: fidelity of 0.60 falls below the 1/2 floor after a single swap, so this
#: preset shows where the hardware is, not a network that works. The values
#: come from separate experiments by different groups; no single device has
#: shown all of them at once, so even this preset is a composite.
TCENTRE_MEASURED = Hardware(
    alpha_db_per_km=0.35,
    link_fidelity=0.60,
    swap_success=0.5,
    generation_rate_hz=1.0,
    t_repeater_memory_s=112e-3,
    t_endnode_memory_s=112e-3,
)

#: Mid-range T centre, the centre of the sweep box. Reference point for the
#: single-case run at the end of week 1. A scenario, not a measured or
#: projected device.
TCENTRE_MIDRANGE = Hardware(
    alpha_db_per_km=0.35,
    link_fidelity=0.96,
    swap_success=0.70,
    generation_rate_hz=20e3,
    t_repeater_memory_s=10e-3,
    t_endnode_memory_s=10e-3,
)

#: Published projection: 200 kHz at F = 0.998 (Afzal 2024 SVII). Swap success
#: 0.95 and the 100 ms coherence are the top of the sweep box rather than part
#: of that projection, so this preset is a composite scenario, not a device
#: anyone has projected as a whole.
TCENTRE_PROJECTED = Hardware(
    alpha_db_per_km=0.35,
    link_fidelity=0.998,
    swap_success=0.95,
    generation_rate_hz=200e3,
    t_repeater_memory_s=100e-3,
    t_endnode_memory_s=100e-3,
)

# The gate and readout pair once used by scripts/w1_gate_noise_check.py is not
# here. Its 0.946 readout figure is an erbium-ion measurement, so it lives in
# qrp.legacy under a name that says so, and SWEEP_BOUNDS["swap_werner"]
# replaces it.


# --------------------------------------------------------------------------
# The sweep box
# --------------------------------------------------------------------------

#: Four swept parameters, with the bound justifications from vault note 42.
#: Each bound is bracketed by a published measurement on some platform, so
#: none of these ranges are invented.
#:
#:   generation_rate_hz  1 kHz   DeAbreu 2023 nanobeam
#:                       200 kHz Afzal 2024 projection
#:   t2_s                1 ms    low end of device coherence. NOTE: Song
#:                               et al. 2025 measure the ELECTRON echo at
#:                               0.41(2) ms, below this floor; the floor is
#:                               defensible only for the nuclear memory the
#:                               model actually assumes.
#:                       100 ms  nuclear echo, Song et al. 2025 / Knaut 2024
#:   link_fidelity       0.92    Higginbottom 2025 electrical initialisation
#:                       0.998   Afzal 2024 projection
#:   swap_success        0.50    linear-optical Bell measurement ceiling
#:                       0.95    deterministic matter-mediated swap
SWEEP_BOUNDS: dict[str, tuple[float, float]] = {
    "generation_rate_hz": (1.0e3, 200.0e3),
    "t2_s": (1.0e-3, 100.0e-3),
    "link_fidelity": (0.92, 0.998),
    "swap_success": (0.50, 0.95),
    # The per-swap "swap quality" s_q = P_2 (4 eta^2 - 1)/3 from Eq. (5),
    # named after da Silva/Avis/Wehner (QST 9, 045041 (2024)), who collapse
    # gate + measurement noise into one depolarizing parameter exactly this
    # way. Eta is, per the original derivation (Briegel et al. PRL 81, 5932
    # (1998); Duer et al. PRA 59, 169 (1999)), the single-qubit assignment
    # fidelity of each of the two measured qubits, hence the square.
    #
    # Bracket: lower end 0.71 from measured-today components (gates
    # 98.4-98.6% and nuclear SPAM 0.87-0.89, Afzal 2024 arXiv:2406.01704 --
    # conservative, since SPAM includes preparation error); upper end 0.997
    # from projected cavity-assisted readout 0.996-0.9996 (Wong & Chen 2025,
    # arXiv:2510.26797) with gates ~0.998 (assumption, no printed T-centre
    # projection exists). The architecture reads nuclear spins by repetitive
    # QND mapping through the electron (Simmons PRX Quantum 5, 010102,
    # citing Neumann Science 329, 542), so eta is repetition-improvable, not
    # pinned at any single-shot number.
    #
    # Not part of SWEEP_ORDER, so the default four-parameter sweeps are
    # unchanged; scripts opt in with SWEEP_ORDER_5.
    "swap_werner": (0.71, 0.997),
}

#: Parameters sampled on a log scale. Rate and coherence span two orders of
#: magnitude each, so uniform sampling would waste most points at the top.
LOG_SCALED: frozenset[str] = frozenset({"generation_rate_hz", "t2_s"})

SWEEP_ORDER: tuple[str, ...] = (
    "generation_rate_hz",
    "t2_s",
    "link_fidelity",
    "swap_success",
)

#: Five-parameter order including the per-swap noise factor. Used by the
#: swap-noise sensitivity run; everything else keeps SWEEP_ORDER.
SWEEP_ORDER_5: tuple[str, ...] = SWEEP_ORDER + ("swap_werner",)

#: Human-readable axis labels, used by every figure.
LABELS: dict[str, str] = {
    "generation_rate_hz": "Generation rate (Hz)",
    "t2_s": "Coherence time T2 (s)",
    "link_fidelity": "Link fidelity F_L",
    "swap_success": "Swap success q_s",
    "swap_werner": "Per-swap noise factor",
}


def hardware_from_sweep(
    values: dict[str, float],
    base: Hardware = TCENTRE_MIDRANGE,
) -> Hardware:
    """Turn one sweep point into a Hardware instance.

    The single coherence parameter t2_s sets both memory coherence times.
    The vault treats repeater and end-node memories as the same physical
    device, and splitting them would double the sweep dimension for a
    distinction the hardware does not currently make.
    """
    unknown = set(values) - set(SWEEP_BOUNDS)
    if unknown:
        raise KeyError(f"unknown sweep parameters: {sorted(unknown)}")

    t2 = values.get("t2_s", base.t_repeater_memory_s)
    return base.with_(
        generation_rate_hz=values.get("generation_rate_hz", base.generation_rate_hz),
        link_fidelity=values.get("link_fidelity", base.link_fidelity),
        swap_success=values.get("swap_success", base.swap_success),
        t_repeater_memory_s=t2,
        t_endnode_memory_s=t2,
        swap_werner=values.get("swap_werner", base.swap_werner),
    )
