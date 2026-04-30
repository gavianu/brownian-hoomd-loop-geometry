"""Inițializare particule: poziții în uniune + viteze (zero sau MB)."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from brownian_sim.geometry.assembly import Assembly


def sample_positions(
    assembly: Assembly,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Puncte uniform distribuite în uniunea de piese (float32, (N,3))."""
    return assembly.sample_uniform(n_particles, rng)


def sample_velocities(
    n_particles: int,
    mode: Literal["zero", "maxwell_boltzmann"],
    kT_over_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Viteze inițiale.

    - 'zero': sistemul pornește rece și se termalizează prin Langevin/bounce.
    - 'maxwell_boltzmann': sistemul pornește la temperatura țintă.
    """
    if mode == "zero":
        return np.zeros((n_particles, 3), dtype=np.float32)
    if mode == "maxwell_boltzmann":
        sigma = math.sqrt(kT_over_m)
        return (rng.standard_normal((n_particles, 3)) * sigma).astype(np.float32)
    raise ValueError(f"velocity mode necunoscut: {mode}")
