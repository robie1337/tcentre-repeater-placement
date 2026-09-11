"""Values retained only to reproduce earlier results.

Nothing in this module is a current T centre measurement. Importing from here
is a deliberate act: the main hardware module does not re-export any of it.
"""

from __future__ import annotations

#: A gate and readout pair once used as "measured T centre values" by
#: scripts/w1_gate_noise_check.py.
#:
#: The 0.986 gate fidelity is Afzal et al. 2024 (arXiv:2406.01704) and is a
#: T centre value. The 0.946 readout fidelity is NOT. It is the single-shot
#: QND readout of a single erbium ion (Raha, Chen, Phenicie, Ourari, Dibos &
#: Thompson, Nat. Commun. 11, 1605 (2020)), long misattributed here to
#: Higginbottom et al. 2022, whose preprint (arXiv:2103.07580) reports no
#: readout fidelity at all. Confirmed 2026-08-24. The defensible measured floor
#: is Afzal 2024's nuclear SPAM of 0.87-0.89, and the swap-quality bracket in
#: qrp.hardware.SWEEP_BOUNDS["swap_werner"] replaces this pair. Do not cite
#: 0.946 as a T centre number.
LEGACY_MISATTRIBUTED_GATE_READOUT = {"gate_fidelity": 0.986, "measurement_fidelity": 0.946}
