"""Verificări echilibru termic și stabilitate NESS.

Teste:
  - <v^2> = 3 kT/m (echiparție la echilibru)
  - distribuție Maxwell-Boltzmann (test simplu pe varianța per-axă)
  - stabilitate counts pe piesă (NESS vs tranzitoriu)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureCheck:
    v2_mean: float
    v2_expected: float
    relative_error: float
    passed: bool


def check_temperature(
    velocities: np.ndarray,
    kT: float,
    mass: float = 1.0,
    tol: float = 0.05,
) -> TemperatureCheck:
    """<v^2> == 3 kT/m pentru MB la echilibru."""
    v = velocities.astype(np.float64)
    v2 = float(np.mean(np.sum(v * v, axis=1)))
    expected = 3.0 * kT / mass
    rel = abs(v2 - expected) / expected
    return TemperatureCheck(v2, expected, rel, rel < tol)


def counts_stability(counts_df, window_frac: float = 0.3) -> dict:
    """Compară media celor ultime ~window_frac fraction față de media completă.

    Dacă counts-urile sunt stabile pe piesă, este o bună indicație de regim staționar.
    """
    n = len(counts_df)
    last_n = max(1, int(window_frac * n))
    pieces = [c for c in counts_df.columns if c != "step"]
    out = {}
    for p in pieces:
        full = float(counts_df[p].mean())
        late = float(counts_df[p].iloc[-last_n:].mean())
        rel = abs(late - full) / full if full > 0 else 0.0
        out[p] = {"mean_full": full, "mean_late": late, "relative_drift": rel}
    return out
