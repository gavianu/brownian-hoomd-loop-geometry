"""Material properties pentru frontiere — independente de modelul de bounce.

Un `WallMaterial` transportă parametrii locali (e_n, beta_t) care intră în
toate modelele de coliziune. Modelele (Elastic, Damped, OU) îi consumă uniform.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WallMaterial:
    """Coeficient de restituție normal și amortizare tangențială.

    e_n = 1, beta_t = 1  -> reflexie perfect elastică
    e_n < 1              -> pierdere pe componenta normală
    beta_t < 1           -> frecare tangențială
    """
    e_n: float
    beta_t: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.e_n <= 1.0):
            raise ValueError(f"e_n trebuie în [0,1], primit {self.e_n}")
        if not (0.0 <= self.beta_t <= 1.0):
            raise ValueError(f"beta_t trebuie în [0,1], primit {self.beta_t}")
