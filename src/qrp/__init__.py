"""Hardware-sensitive repeater placement on the CA9 topology.

Adapts the placement MILP of Pouryousef et al. (IEEE TQE 2024,
arXiv:2308.16264v3) to silicon T centre hardware, and sweeps
the four hardware parameters that are not yet pinned down.

No quantum network simulator is involved. Rate and fidelity both have closed
forms in the source paper, so the model computes them directly.

Submodules are imported lazily so that importing ``qrp.physics`` does not
drag in matplotlib.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_SUBMODULES = (
    "figures",
    "hardware",
    "model",
    "paths",
    "physics",
    "sensitivity",
    "solver",
    "sweep",
    "topology",
)

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
