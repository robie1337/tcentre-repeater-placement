"""The misattributed readout value must not look like a current measurement."""

from __future__ import annotations

from qrp import hardware, legacy


def test_hardware_module_does_not_expose_it():
    assert not hasattr(hardware, "GATE_NOISE_MEASURED")
    for name in dir(hardware):
        value = getattr(hardware, name)
        if isinstance(value, dict):
            assert value.get("measurement_fidelity") != 0.946


def test_legacy_name_says_what_it_is():
    pair = legacy.LEGACY_MISATTRIBUTED_GATE_READOUT
    assert pair["measurement_fidelity"] == 0.946
    assert pair["gate_fidelity"] == 0.986
