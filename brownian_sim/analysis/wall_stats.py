"""Histograme distanța-la-perete per piesă."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from brownian_sim.geometry.assembly import Assembly


def wall_distance_histograms(
    positions: np.ndarray,
    piece_idx: np.ndarray,
    assembly: Assembly,
    n_bins: int = 40,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Pentru fiecare piesă, histograma distanței la cel mai apropiat perete."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for k, pc in enumerate(assembly.pieces):
        mask = piece_idx == k
        if not np.any(mask):
            continue
        dists = np.array(
            [assembly.wall_distance(positions[i], k) for i in np.where(mask)[0]],
            dtype=np.float64,
        )
        hist, edges = np.histogram(dists, bins=n_bins)
        out[pc.name] = {"hist": hist, "edges": edges, "mean_dist": float(np.mean(dists))}
    return out
