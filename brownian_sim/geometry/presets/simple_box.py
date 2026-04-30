"""Preset `simple_box`: o singură cutie cu un singur material.

Cazul de referință pentru validare: într-o cutie cu pereți termalizanți OU,
distribuția vitezelor la echilibru trebuie să fie Maxwell-Boltzmann și <v²> = 3 kT/m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from brownian_sim.geometry.primitives import Box
from brownian_sim.geometry.piece import Piece
from brownian_sim.geometry.assembly import Assembly
from brownian_sim.materials.wall_material import WallMaterial


@dataclass
class SimpleBoxParams:
    size: Tuple[float, float, float] = (100.0, 100.0, 100.0)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: WallMaterial = WallMaterial(e_n=0.95, beta_t=0.95)


def build(params: SimpleBoxParams | None = None) -> Assembly:
    p = params or SimpleBoxParams()
    return Assembly([Piece("BOX", Box(center=p.center, size=p.size), p.material)])
