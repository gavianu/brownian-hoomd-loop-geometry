"""Backend abstract numpy/cupy.

Funcția `get_xp(device)` returnează modulul array. Dacă device >= 0
și CuPy este disponibil, folosim cp; altfel numpy.

Intenția este ca restul codului (dynamics, wall_models) să primească
`xp` ca parametru și să nu importe explicit numpy/cupy. Asta ne permite
să rulăm același cod pe CPU și GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def get_xp(device: int = -1) -> Any:
    """device == -1 -> numpy; device >= 0 -> cupy (dacă disponibil)."""
    if device < 0:
        return np
    if not _HAS_CUPY:
        import warnings
        warnings.warn(f"CuPy indisponibil — fallback pe CPU (numpy) pentru device={device}")
        return np
    cp.cuda.Device(device).use()
    return cp


def to_cpu(a: Any) -> np.ndarray:
    """Mută orice array pe CPU (numpy)."""
    if _HAS_CUPY and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def is_gpu(xp: Any) -> bool:
    return _HAS_CUPY and xp is cp
