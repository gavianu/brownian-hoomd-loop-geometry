"""Piece = primitivă + material + nume.

Asocierea material-geometrie se face la acest nivel, nu în primitivă —
astfel aceeași formă geometrică poate apărea cu materiale diferite,
fără duplicare de cod.
"""
from __future__ import annotations

from dataclasses import dataclass

from brownian_sim.geometry.primitives import Primitive
from brownian_sim.materials.wall_material import WallMaterial


@dataclass
class Piece:
    name: str
    shape: Primitive
    material: WallMaterial
