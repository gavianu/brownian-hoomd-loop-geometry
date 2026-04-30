"""Integrator Langevin în volum — decuplat complet de coliziunea cu peretele.

Integrare Euler-Maruyama pe viteză:
    dv = -(gamma/m) v dt + sqrt(2 gamma kT / m^2) dW
        = -(gamma/m) v dt + (sqrt(2 gamma kT)/m) sqrt(dt) xi

Pasul pe poziție se face explicit în engine:
    p_new = p_old + v * dt

Separarea asta ne permite să folosim sub-stepping CFL pe pasul de poziție
fără a rebroda pasul Langevin pe viteză.
"""
from __future__ import annotations

import math
from typing import Any


class LangevinIntegrator:
    """Implementează doar pasul de viteză. Pasul de poziție e controlat de engine."""

    def __init__(self, mass: float, gamma: float, kT: float, dt: float) -> None:
        if mass <= 0 or gamma <= 0 or kT <= 0 or dt <= 0:
            raise ValueError("mass, gamma, kT, dt trebuie > 0")
        self.mass = float(mass)
        self.gamma = float(gamma)
        self.kT = float(kT)
        self.dt = float(dt)

        # precalcule
        self._damping = self.gamma / self.mass
        self._noise_sigma = math.sqrt(2.0 * self.gamma * self.kT) * math.sqrt(self.dt) / self.mass

    @property
    def kT_over_m(self) -> float:
        return self.kT / self.mass

    def step_velocity(self, v: Any, xi: Any) -> Any:
        """Un pas Euler-Maruyama pe viteză.

        v:  array (N,3)
        xi: array (N,3) gaussian standard pregătit de caller (pentru control RNG)
        """
        return v + (-self._damping) * v * self.dt + self._noise_sigma * xi
