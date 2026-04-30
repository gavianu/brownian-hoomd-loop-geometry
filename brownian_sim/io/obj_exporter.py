"""Export geometrie ca .obj/.mtl pentru vizualizare în OVITO/Blender.

Fiecare piesă devine un mesh translucid cu material distinct. Utilizat pentru
context vizual: particulele (din GSD) sunt suprapuse peste geometria statică.
"""
from __future__ import annotations

import math
import os
from typing import List

from brownian_sim.geometry.assembly import Assembly
from brownian_sim.geometry.primitives import Box, _CylinderAxisAligned


def export_assembly_obj(
    assembly: Assembly,
    out_dir: str,
    alpha: float = 0.35,
    cyl_sides: int = 96,
) -> tuple[str, str]:
    """Scrie assembly ca .obj + .mtl în out_dir. Returnează (obj_path, mtl_path)."""
    os.makedirs(out_dir, exist_ok=True)
    obj_path = os.path.join(out_dir, "geometry.obj")
    mtl_path = os.path.join(out_dir, "geometry.mtl")

    with open(mtl_path, "w") as m:
        # culori per tip (box vs cyl)
        m.write(f"newmtl box_mat\nKd 0.70 0.70 0.95\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl cylx_mat\nKd 0.70 0.95 0.70\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl cyly_mat\nKd 0.95 0.95 0.70\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl ret_mat\nKd 0.95 0.70 0.70\nd {alpha}\nillum 2\n\n")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        base = 0
        for pc in assembly.pieces:
            mat = _material_for(pc.name)
            f.write(f"o {pc.name}\nusemtl {mat}\n")
            if isinstance(pc.shape, Box):
                base = _write_box(f, base, pc.shape)
            elif isinstance(pc.shape, _CylinderAxisAligned):
                base = _write_cyl(f, base, pc.shape, cyl_sides)
    return obj_path, mtl_path


def _material_for(name: str) -> str:
    if name.startswith("CUBE") or name.startswith("BOX"):
        return "box_mat"
    if name.startswith("RET"):
        return "ret_mat"
    if name.startswith("VERT"):
        return "cyly_mat"
    return "cylx_mat"


def _write_box(f, base: int, box: Box) -> int:
    cx, cy, cz = box._c
    sx, sy, sz = box._s
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    V = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    F = [
        (0, 1, 2), (0, 2, 3),   # -z
        (4, 6, 5), (4, 7, 6),   # +z
        (0, 4, 5), (0, 5, 1),   # -y
        (1, 5, 6), (1, 6, 2),   # +x
        (2, 6, 7), (2, 7, 3),   # +y
        (3, 7, 4), (3, 4, 0),   # -x
    ]
    for v in V:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for a, b, c in F:
        f.write(f"f {a+1+base} {b+1+base} {c+1+base}\n")
    return base + len(V)


def _write_cyl(f, base: int, cyl, sides: int) -> int:
    ax = cyl.AXIS_IDX
    c = cyl._c.copy()
    R = cyl.R
    L = cyl.L
    h = L / 2

    # endpoint coordinates along axis
    v0 = c.copy()
    v1 = c.copy()
    v0[ax] = c[ax] - h
    v1[ax] = c[ax] + h

    # radial axes
    r_axes = [i for i in (0, 1, 2) if i != ax]

    # generate rings
    ring0, ring1 = [], []
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        dx = R * math.cos(theta)
        dy = R * math.sin(theta)
        p0 = v0.copy()
        p1 = v1.copy()
        p0[r_axes[0]] = c[r_axes[0]] + dx
        p0[r_axes[1]] = c[r_axes[1]] + dy
        p1[r_axes[0]] = c[r_axes[0]] + dx
        p1[r_axes[1]] = c[r_axes[1]] + dy
        ring0.append(tuple(p0))
        ring1.append(tuple(p1))

    V = ring0 + ring1
    for v in V:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    # mantle triangles
    for i in range(sides):
        a = base + i
        e = base + ((i + 1) % sides)
        d = base + sides + i
        cc = base + sides + ((i + 1) % sides)
        f.write(f"f {a+1} {e+1} {cc+1}\n")
        f.write(f"f {a+1} {cc+1} {d+1}\n")
    return base + len(V)
