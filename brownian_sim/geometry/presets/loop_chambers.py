"""Preset `loop_chambers`: geometria originală cu 2 camere, funnel 6-segmente, canal retur.

Portat din `legacy/sim_scripts/analytic_langevin_termal_collission.py::make_geometry`.
Parametrii numerici sunt preluați 1:1 pentru a putea valida refactor-ul vs legacy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from brownian_sim.geometry.primitives import Box, CylX
from brownian_sim.geometry.piece import Piece
from brownian_sim.geometry.assembly import Assembly
from brownian_sim.materials.wall_material import WallMaterial


@dataclass
class LoopChambersParams:
    """Parametri geometrici și materiali pentru configurația loop-chambers.

    Valorile default sunt cele din scriptul original (baseline 052a205).
    """

    # dimensiuni camere
    cube_size: Tuple[float, float, float] = (80.0, 80.0, 80.0)
    cub1_center: Tuple[float, float, float] = (-120.0, 70.0, 0.0)
    cub2_center: Tuple[float, float, float] = (120.0, 70.0, 0.0)

    # funnel
    funnel_y: float = 90.0
    funnel_z: float = 0.0
    funnel_radii: Tuple[float, ...] = (8.0, 17.0, 25.0, 15.0, 12.0, 10.0)
    funnel_pad: float = 2.0
    seal_overlap: float = 6.0

    # canal retur
    loop_y: float = 45.0
    ret_R: float = 12.0
    ret_extra: float = 10.0

    # materiale
    mat_cube_L: WallMaterial = field(default_factory=lambda: WallMaterial(e_n=0.9, beta_t=0.7))
    mat_cube_R: WallMaterial = field(default_factory=lambda: WallMaterial(e_n=0.3, beta_t=0.55))
    mat_funnel: Tuple[WallMaterial, ...] = field(
        default_factory=lambda: (
            WallMaterial(e_n=0.85, beta_t=0.85),
            WallMaterial(e_n=0.98, beta_t=0.30),
            WallMaterial(e_n=0.98, beta_t=0.30),
            WallMaterial(e_n=0.98, beta_t=0.30),
            WallMaterial(e_n=0.98, beta_t=0.30),
            WallMaterial(e_n=0.98, beta_t=0.30),
        )
    )
    mat_ret: WallMaterial = field(default_factory=lambda: WallMaterial(e_n=0.98, beta_t=0.02))


def build(params: LoopChambersParams | None = None) -> Assembly:
    p = params or LoopChambersParams()
    pieces: List[Piece] = []

    # camere
    pieces.append(Piece("CUBE_L", Box(center=p.cub1_center, size=p.cube_size), p.mat_cube_L))
    pieces.append(Piece("CUBE_R", Box(center=p.cub2_center, size=p.cube_size), p.mat_cube_R))

    # funnel: 6 segmente de cilindri pe OX între fețele interioare ale cuburilor
    x1 = p.cub1_center[0] + p.cube_size[0] / 2
    x2 = p.cub2_center[0] - p.cube_size[0] / 2
    dist = x2 - x1
    n_seg = len(p.funnel_radii)
    seg = dist / n_seg
    # lungimi individuale (preluate din legacy: extremele mai lungi, centrele mai scurte)
    segi = [seg + 20, seg - 10, seg - 10, seg - 10, seg - 10, seg + 20]

    for i, R in enumerate(p.funnel_radii):
        cx = x1 + (i + 0.5) * seg
        L = segi[i] + 2 * p.funnel_pad + 2 * p.seal_overlap
        pieces.append(
            Piece(
                f"FUN_{i+1}",
                CylX(cx=cx, cy=p.funnel_y, cz=p.funnel_z, R=R, L=L),
                p.mat_funnel[i],
            )
        )

    # canal retur orizontal
    xR0 = x1 - p.ret_extra
    xR1 = x2 + p.ret_extra
    L_ret = xR1 - xR0
    cxr = 0.5 * (xR0 + xR1)
    pieces.append(
        Piece("RET", CylX(cx=cxr, cy=p.loop_y, cz=0.0, R=p.ret_R, L=L_ret), p.mat_ret)
    )

    return Assembly(pieces)
