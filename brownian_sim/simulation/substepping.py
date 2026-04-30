"""Sub-stepping adaptiv bazat pe CFL local.

Scopul: evităm ca o particulă să străbată peretele într-un singur pas
(ceea ce ar genera un "miss" sau o penetrare mare, care apoi ar cere
snap brutal). Împărțim pasul în nsub sub-pași astfel încât
    max_speed * dt / nsub  <  CFL * dmed
unde dmed e distanța la perete pentru particulele de tracking.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from brownian_sim.geometry.assembly import Assembly


def compute_substeps(
    positions: np.ndarray,      # (N,3)
    velocities: np.ndarray,     # (N,3)
    piece_idx: np.ndarray,      # (N,) int
    dt: float,
    assembly: Assembly,
    sample_ids: np.ndarray,     # (K,) indici pentru track
    cfl: float = 0.5,
    max_substeps: int = 32,
) -> int:
    """Decide câți sub-pași să facem pentru pasul curent.

    Returnează 1..max_substeps.
    """
    if max_substeps <= 1:
        return 1

    speeds = np.linalg.norm(velocities, axis=1)
    if speeds.size == 0:
        return 1
    max_speed = float(np.max(np.where(np.isfinite(speeds), speeds, 0.0)))
    step_len = max_speed * dt + 1e-12

    dists: List[float] = []
    for pid in sample_ids:
        k = int(piece_idx[pid])
        if k < 0:
            continue
        dists.append(assembly.wall_distance(positions[pid], k))
    if not dists:
        return 1
    dmed = float(np.percentile(dists, 20))
    if dmed <= 1e-6:
        return 1

    est = step_len / (cfl * dmed)
    nsub = max(1, int(math.ceil(est)))
    return min(nsub, max_substeps)
