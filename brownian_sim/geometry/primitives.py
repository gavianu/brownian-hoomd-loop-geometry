"""Geometric primitives: axis-aligned box, cylinders along X or Y.

Fiecare primitivă expune:
  - inside(P):        mask boolean pe punctele din interior
  - wall_distance(p): distanța minimă la perete (scalar)
  - snap_and_normal(p_new): snap la frontieră + normală spre interior
                            (utilizat când p_new a ieșit din primitivă)

Interfața este deliberat minimală pentru ca orice primitivă nouă
(ex. sferă, cilindru oblic) să se adauge fără să schimbe restul codului.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


class Primitive(ABC):
    """Interfață comună pentru toate primitivele geometrice."""

    # identificator numeric pentru reprezentări vectorizate
    TYPE_ID: int = -1

    @abstractmethod
    def inside(self, P: np.ndarray) -> np.ndarray:
        """(N,3) -> (N,) bool; True dacă punctul este în interiorul primitivei."""

    @abstractmethod
    def wall_distance(self, p: np.ndarray) -> float:
        """Distanța punctului p (3,) la cel mai apropiat perete."""

    @abstractmethod
    def snap_and_normal(self, p_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dacă p_new este afară, returnează (p_snap pe frontieră, n unitară spre interior).
        Dacă e înăuntru, returnează (p_new, 0-vector) — caller-ul verifică norma.
        """

    @abstractmethod
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returnează (min_corner, max_corner) pentru sampling uniform."""


@dataclass
class Box(Primitive):
    """Cutie aliniată cu axele, definită prin centru și dimensiuni (sx, sy, sz)."""

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    TYPE_ID = 0

    def __post_init__(self) -> None:
        self._c = np.asarray(self.center, dtype=np.float64)
        self._s = np.asarray(self.size, dtype=np.float64)
        self._half = self._s * 0.5

    def inside(self, P: np.ndarray) -> np.ndarray:
        d = np.abs(P - self._c)
        return np.all(d <= self._half, axis=-1)

    def wall_distance(self, p: np.ndarray) -> float:
        d = self._half - np.abs(p - self._c)
        return float(max(0.0, float(np.min(d))))

    def snap_and_normal(self, p_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = p_new - self._c
        pen = np.abs(d) - self._half
        if np.all(pen <= 1e-12):
            return p_new.copy(), np.zeros(3, dtype=np.float64)
        axis = int(np.argmax(pen))
        p_snap = p_new.copy().astype(np.float64)
        n = np.zeros(3, dtype=np.float64)
        sign = math.copysign(1.0, d[axis]) if d[axis] != 0 else 1.0
        p_snap[axis] = self._c[axis] + sign * self._half[axis]
        n[axis] = sign
        return p_snap, n

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._c - self._half, self._c + self._half


@dataclass
class _CylinderAxisAligned(Primitive):
    """Bază pentru cilindri aliniați cu o axă. Subclase: CylX, CylY."""

    cx: float
    cy: float
    cz: float
    R: float
    L: float

    # axis_idx: 0 pentru X, 1 pentru Y — setat în subclasă
    AXIS_IDX: int = -1

    def __post_init__(self) -> None:
        self._c = np.array([self.cx, self.cy, self.cz], dtype=np.float64)
        self._half_L = 0.5 * self.L

    def _radial_axes(self) -> Tuple[int, int]:
        """Indicii celor două axe radiale (nu axiala)."""
        ax = self.AXIS_IDX
        axes = [i for i in (0, 1, 2) if i != ax]
        return axes[0], axes[1]

    def inside(self, P: np.ndarray) -> np.ndarray:
        ax = self.AXIS_IDX
        r1, r2 = self._radial_axes()
        axial = np.abs(P[..., ax] - self._c[ax]) <= self._half_L
        radial2 = (P[..., r1] - self._c[r1]) ** 2 + (P[..., r2] - self._c[r2]) ** 2
        return axial & (radial2 <= self.R * self.R)

    def wall_distance(self, p: np.ndarray) -> float:
        ax = self.AXIS_IDX
        r1, r2 = self._radial_axes()
        d_axial = self._half_L - abs(p[ax] - self._c[ax])
        r = math.hypot(p[r1] - self._c[r1], p[r2] - self._c[r2])
        d_radial = self.R - r
        return float(max(0.0, min(d_axial, d_radial)))

    def snap_and_normal(self, p_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ax = self.AXIS_IDX
        r1, r2 = self._radial_axes()
        p_snap = p_new.copy().astype(np.float64)
        n = np.zeros(3, dtype=np.float64)

        # capăt (cap) mai întâi
        if abs(p_new[ax] - self._c[ax]) > self._half_L:
            sign = math.copysign(1.0, p_new[ax] - self._c[ax])
            p_snap[ax] = self._c[ax] + sign * self._half_L
            n[ax] = sign
            return p_snap, n

        # manta
        d1 = p_new[r1] - self._c[r1]
        d2 = p_new[r2] - self._c[r2]
        r = math.hypot(d1, d2)
        if r > self.R:
            # snap pe cerc + normală radială (spre interior)
            p_snap[r1] = self._c[r1] + self.R * d1 / r
            p_snap[r2] = self._c[r2] + self.R * d2 / r
            # normala spre exterior în primă fază; caller-ul o folosește așa
            n[r1] = d1 / r
            n[r2] = d2 / r
            return p_snap, n

        return p_new.copy(), np.zeros(3, dtype=np.float64)

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        ax = self.AXIS_IDX
        mn = self._c.copy() - self.R
        mx = self._c.copy() + self.R
        mn[ax] = self._c[ax] - self._half_L
        mx[ax] = self._c[ax] + self._half_L
        return mn, mx


@dataclass
class CylX(_CylinderAxisAligned):
    """Cilindru cu axa paralelă cu OX."""
    AXIS_IDX = 0
    TYPE_ID = 1


@dataclass
class CylY(_CylinderAxisAligned):
    """Cilindru cu axa paralelă cu OY."""
    AXIS_IDX = 1
    TYPE_ID = 2
