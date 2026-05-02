"""Modele de coliziune particulă-perete.

Toate modelele au aceeași interfață:
    bounce(v_in, n, material) -> v_out

unde:
    v_in   : (3,) viteza incidentă
    n      : (3,) normala unitară spre interior
    material : WallMaterial (e_n, beta_t)

Toate modelele respectă convenția: după bounce, componenta normală a
vitezei este orientată spre interior (v_out · n > 0).

Modele disponibile:
  - ElasticBounce: reflexie speculară (e_n = beta_t = 1 efectiv, ignoră materialul)
  - DampedBounce:  v' = -e_n * v_n + beta_t * v_t  (disipativ, fără zgomot termic)
  - OUBounce:      v' cu termen stocastic calibrat FDT (termalizant la T)
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from brownian_sim.materials.wall_material import WallMaterial


class WallModel(ABC):
    """Interfață comună pentru modele de coliziune."""

    @abstractmethod
    def bounce_single(
        self, v_in: Any, n: Any, material: WallMaterial, rng: Any
    ) -> Any:
        """Bounce pe o singură particulă (CPU path, pentru claritate/teste).

        v_in: shape (3,), n: shape (3,) unit, material: WallMaterial
        Returns v_out: shape (3,)
        """

    @abstractmethod
    def bounce_batch(
        self,
        v_in: Any,          # (M,3)
        n: Any,             # (M,3) unit
        e_n: Any,           # (M,)
        beta_t: Any,        # (M,)
        xp: Any,
        rng: Any = None,
    ) -> Any:
        """Bounce vectorizat pe M particule. Returnează (M,3)."""

    # proprietate utilă pentru modelele care au nevoie de kT/m
    def needs_temperature(self) -> bool:
        return False


# ---------------- ElasticBounce ----------------

class ElasticBounce(WallModel):
    """Reflexie speculară — ignoră complet materialul.

    v' = v - 2(v·n)n
    Conservă energia cinetică exact. Util pentru validare și cazul idealizat
    din lucrarea de licență (Secțiunea 2.3).
    """

    def bounce_single(self, v_in, n, material, rng):
        vn_scalar = float(v_in.dot(n))
        return v_in - 2.0 * vn_scalar * n

    def bounce_batch(self, v_in, n, e_n, beta_t, xp, rng=None):
        vn = xp.sum(v_in * n, axis=1, keepdims=True)  # (M,1)
        return v_in - 2.0 * vn * n


# ---------------- DampedBounce ----------------

class DampedBounce(WallModel):
    """Reflexie disipativă — consumă materialul.

    v_n' = -e_n * v_n
    v_t' =  beta_t * v_t
    Corespunde modelului din Secțiunea 2.7 a lucrării. Disipativ, fără
    zgomot termic — la limita lungă fără OU bounce, sistemul se răcește.
    """

    def bounce_single(self, v_in, n, material, rng):
        vn_scalar = float(v_in.dot(n))
        vn = vn_scalar * n
        vt = v_in - vn
        return -material.e_n * vn + material.beta_t * vt

    def bounce_batch(self, v_in, n, e_n, beta_t, xp, rng=None):
        vn = xp.sum(v_in * n, axis=1, keepdims=True)  # (M,1)
        v_normal = vn * n
        v_tangent = v_in - v_normal
        e = e_n[:, None]
        b = beta_t[:, None]
        return -e * v_normal + b * v_tangent


# ---------------- OUBounce ----------------

class OUBounce(WallModel):
    """Ornstein-Uhlenbeck bounce: disipare + kick termic calibrat FDT.

    v_t' = beta_t * v_t + sqrt(1 - beta_t^2) * s * xi_t
    v_n' = e_n * |v_n_in| + sqrt(1 - e_n^2) * s * xi_n    (forțat v_n' > 0)

    unde s = sqrt(kT/m) și xi sunt gaussieni standard. Factorii
    sqrt(1 - e^2) și sqrt(1 - beta^2) calibrează zgomotul astfel încât
    distribuția staționară a vitezelor post-bounce să fie Maxwell-Boltzmann
    la temperatura T — relația fluctuație-disipație aplicată local pe perete.

    Aceasta este cea mai completă condiție la limită din lucrare
    (Secțiunea 2.8).
    """

    def __init__(self, kT_over_m: float) -> None:
        self.kT_over_m = float(kT_over_m)
        self.s = math.sqrt(self.kT_over_m)

    def needs_temperature(self) -> bool:
        return True

    def bounce_single(self, v_in, n, material, rng):
        import numpy as np
        vn_scalar = float(v_in.dot(n))
        vn_in = abs(vn_scalar)
        v_tangent = v_in - vn_scalar * n

        # tangent kick: proiectează gaussian 3D pe planul tangent
        xi3 = rng.standard_normal(3)
        xi_t = xi3 - float(xi3.dot(n)) * n
        st = math.sqrt(max(0.0, 1.0 - material.beta_t ** 2)) * self.s
        vt_p = material.beta_t * v_tangent + st * xi_t

        # normal: re-draw până iese pozitiv (max 4 încercări)
        sn = math.sqrt(max(0.0, 1.0 - material.e_n ** 2)) * self.s
        vn_p = material.e_n * vn_in + sn * rng.standard_normal()
        for _ in range(3):
            if vn_p > 0:
                break
            vn_p = material.e_n * vn_in + sn * rng.standard_normal()
        vn_p = max(vn_p, 1e-12)

        return vt_p + vn_p * n

    def bounce_batch(self, v_in, n, e_n, beta_t, xp, rng=None):
        # v_in, n: (M,3); e_n, beta_t: (M,)
        M = v_in.shape[0]
        s = xp.asarray(self.s, dtype=v_in.dtype)

        vn_scalar = xp.sum(v_in * n, axis=1)                # (M,)
        vn_in = xp.abs(vn_scalar)                           # (M,)
        v_tangent = v_in - vn_scalar[:, None] * n           # (M,3)

        # tangent gaussian proiectat pe planul tangent
        xi3 = _standard_normal(xp, v_in.shape, v_in.dtype, rng)
        xi_t = xi3 - xp.sum(xi3 * n, axis=1, keepdims=True) * n

        st = xp.sqrt(xp.maximum(xp.asarray(0.0, dtype=v_in.dtype),
                                1.0 - beta_t ** 2)) * s
        sn = xp.sqrt(xp.maximum(xp.asarray(0.0, dtype=v_in.dtype),
                                1.0 - e_n ** 2)) * s

        vt_p = beta_t[:, None] * v_tangent + st[:, None] * xi_t

        # normal: inițial + 3 re-draw pe cele negative
        xi_n = _standard_normal(xp, (M,), v_in.dtype, rng)
        vn_p = e_n * vn_in + sn * xi_n
        for _ in range(3):
            bad = vn_p <= 0
            if not bool(bad.any()):
                break
            xi_n = _standard_normal(xp, (M,), v_in.dtype, rng)
            vn_prop = e_n * vn_in + sn * xi_n
            vn_p = xp.where(bad, vn_prop, vn_p)
        vn_p = xp.maximum(vn_p, xp.asarray(1e-12, dtype=v_in.dtype))

        return vt_p + vn_p[:, None] * n


# ---------------- MaxwellDiffuse ----------------

class MaxwellDiffuse(WallModel):
    """Reflexie difuza Maxwell — perete fizic real la temperatura T uniforma.

    Particula 'uita' complet viteza de intrare:
      v_t' ~ Gaussian 2D izotrop in planul tangent, sigma = sqrt(kT/m)
      v_n' ~ Rayleigh(sigma),  P(v) propto v * exp(-v^2 / 2*sigma^2), v > 0

    Distributia Rayleigh (nu semi-Gaussian) apare din ponderea cu fluxul:
    particulele care ies mai repede strabat suprafata mai des, deci distributia
    de viteze la iesire e ponderata cu v_n => Rayleigh, nu semi-Gaussian.

    Aceasta e singura distributie care satisface balanta detaliata la nivel
    de coliziune individuala => J_loop = 0 la echilibru, indiferent de
    geometrie sau de heterogenitatea materialelor. Modelul de referinta
    pentru 'perete fizic real la T ambient'.
    """

    def __init__(self, kT_over_m: float) -> None:
        self.kT_over_m = float(kT_over_m)
        self.s = math.sqrt(self.kT_over_m)

    def needs_temperature(self) -> bool:
        return True

    def bounce_single(self, v_in, n, material, rng):
        import numpy as np
        s = self.s
        # tangent: 2 componente gaussiene proiectate pe planul tangent
        xi3 = rng.standard_normal(3)
        xi_t = xi3 - float(xi3.dot(n)) * n
        vt_p = s * xi_t

        # normal: Rayleigh via transform: v = s * sqrt(-2 * log(U)), U ~ Uniform(0,1)
        # echivalent cu sqrt al sumei a 2 gaussiene patrate (chi cu 2 grade)
        u = rng.random()
        vn_p = s * math.sqrt(-2.0 * math.log(max(u, 1e-300)))

        return vt_p + vn_p * n

    def bounce_batch(self, v_in, n, e_n, beta_t, xp, rng=None):
        M = v_in.shape[0]
        s = float(self.s)

        # tangent gaussian proiectat
        xi3 = _standard_normal(xp, v_in.shape, v_in.dtype, rng)
        xi_t = xi3 - xp.sum(xi3 * n, axis=1, keepdims=True) * n
        vt_p = s * xi_t

        # normal Rayleigh: sqrt(g1^2 + g2^2) * s
        g1 = _standard_normal(xp, (M,), v_in.dtype, rng)
        g2 = _standard_normal(xp, (M,), v_in.dtype, rng)
        vn_p = s * xp.sqrt(g1**2 + g2**2)

        return vt_p + vn_p[:, None] * n


def _standard_normal(xp, shape, dtype, rng):
    """Helper: gaussian standard cross-backend, cu RNG opțional (numpy)."""
    if rng is not None and hasattr(rng, "standard_normal"):
        # numpy Generator
        return xp.asarray(rng.standard_normal(size=shape), dtype=dtype)
    return xp.random.standard_normal(size=shape).astype(dtype)


# ---------------- factory ----------------

def make_wall_model(name: str, kT_over_m: float | None = None) -> WallModel:
    """Factory pentru folosire din config YAML."""
    name = name.lower()
    if name == "elastic":
        return ElasticBounce()
    if name == "damped":
        return DampedBounce()
    if name == "ou":
        if kT_over_m is None:
            raise ValueError("OUBounce necesită kT_over_m")
        return OUBounce(kT_over_m)
    if name == "maxwell":
        if kT_over_m is None:
            raise ValueError("MaxwellDiffuse necesită kT_over_m")
        return MaxwellDiffuse(kT_over_m)
    raise ValueError(f"wall_model necunoscut: {name}. Opțiuni: elastic, damped, ou, maxwell.")
