"""Mean Square Displacement (MSD) + fit coeficient de difuzie.

MSD pe o traiectorie GSD:
    MSD(t) = <|r(t) - r(0)|^2>
Pentru difuzie browniană 3D:
    MSD(t) = 6 D t   (la timpi lungi)

Returnăm un dict cu rezultate numerice și fit-ul liniar în plaja [t_min, t_max].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MSDResult:
    t: np.ndarray            # (F,) timpi (pași * dt)
    msd_total: np.ndarray    # (F,) MSD 3D
    msd_xyz: np.ndarray      # (F,3) MSD per axă
    D_total: float           # coef. difuzie global
    D_xyz: Tuple[float, float, float]
    fit_range: Tuple[float, float]


def compute_msd_from_gsd(
    gsd_path: str,
    dt: float,
    fit_t_min: float | None = None,
    fit_t_max: float | None = None,
) -> MSDResult:
    import gsd.hoomd  # lazy import

    with gsd.hoomd.open(name=gsd_path, mode="r") as f:
        frames = list(f)

    steps = np.array([fr.configuration.step for fr in frames], dtype=np.float64)
    t = steps * dt

    pos0 = np.asarray(frames[0].particles.position, dtype=np.float64)
    N = pos0.shape[0]
    F = len(frames)

    msd_xyz = np.zeros((F, 3), dtype=np.float64)
    msd_total = np.zeros(F, dtype=np.float64)
    for i, fr in enumerate(frames):
        pos = np.asarray(fr.particles.position, dtype=np.float64)
        d = pos - pos0
        msd_xyz[i] = np.mean(d * d, axis=0)
        msd_total[i] = np.sum(msd_xyz[i])

    # fit liniar
    t_min = fit_t_min if fit_t_min is not None else t[max(1, F // 10)]
    t_max = fit_t_max if fit_t_max is not None else t[-1]
    mask = (t >= t_min) & (t <= t_max)

    def fit_slope(y: np.ndarray) -> float:
        m, _ = np.polyfit(t[mask], y[mask], 1)
        return float(m)

    D_xyz = tuple(fit_slope(msd_xyz[:, k]) / 2.0 for k in range(3))  # type: ignore[assignment]
    D_total = fit_slope(msd_total) / 6.0

    return MSDResult(
        t=t, msd_total=msd_total, msd_xyz=msd_xyz,
        D_total=D_total, D_xyz=D_xyz,  # type: ignore[arg-type]
        fit_range=(float(t_min), float(t_max)),
    )
