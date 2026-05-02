"""Assembly: colecție de Piece + operații vectorizate necesare simulării.

Responsabilități:
  - `locate(P)`  -> (N,) int; index-ul piesei care conține fiecare punct (-1 dacă OUT)
  - `inside_any(P)` -> (N,) bool
  - `snap_and_normal(p_old, p_new, piece_idx)` -> per-point reflecție
  - `wall_distance(p, k)` -> scalar
  - `sample_uniform(N, rng)` -> N puncte random uniform distribuite în uniune

Importatnt: Assembly **nu** face bounce de viteze — asta e rolul WallModel.
Assembly furnizează doar geometria (snap + normală).
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from brownian_sim.geometry.piece import Piece


class Assembly:
    def __init__(self, pieces: List[Piece]) -> None:
        if not pieces:
            raise ValueError("Assembly gol — e nevoie de cel puțin o piesă")
        self.pieces: List[Piece] = list(pieces)

    # ---- metadata ----

    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.pieces]

    def material(self, k: int):
        return self.pieces[k].material

    # ---- queries vectorizate pe numpy (CPU; analog xp pt. GPU) ----

    def inside_any(self, P: np.ndarray) -> np.ndarray:
        m = np.zeros(P.shape[0], dtype=bool)
        for pc in self.pieces:
            m |= pc.shape.inside(P)
        return m

    def locate(self, P: np.ndarray) -> np.ndarray:
        """Primul piece care conține punctul (ordine din lista pieces).

        Dezambiguarea în caz de suprapuneri (seal-uri, cap-uri lipite):
        se returnează primul match. Pentru tie-break aleator, vezi
        `locate_random_tie_break`.
        """
        N = P.shape[0]
        idx = np.full(N, -1, dtype=np.int32)
        assigned = np.zeros(N, dtype=bool)
        for k, pc in enumerate(self.pieces):
            m = pc.shape.inside(P)
            set_here = (~assigned) & m
            idx[set_here] = k
            assigned |= m
        return idx

    def locate_random_tie_break(self, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Ca `locate`, dar cu tie-break aleator în zonele de suprapunere."""
        N = P.shape[0]
        K = len(self.pieces)
        masks = np.zeros((K, N), dtype=bool)
        for k, pc in enumerate(self.pieces):
            masks[k] = pc.shape.inside(P)
        hits = masks.sum(axis=0)
        idx = np.full(N, -1, dtype=np.int32)

        single = hits == 1
        if single.any():
            idx[single] = np.argmax(masks[:, single], axis=0).astype(np.int32)

        multi = hits >= 2
        if multi.any():
            multi_ids = np.where(multi)[0]
            for j in multi_ids:
                cand = np.where(masks[:, j])[0]
                idx[j] = int(rng.choice(cand))
        return idx

    # ---- reflecție geometrică ----

    def snap_and_normal(
        self, p_new: np.ndarray, piece_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returnează (p_snap, n) pentru un singur punct care a ieșit din piece_idx."""
        return self.pieces[piece_idx].shape.snap_and_normal(p_new)

    def snap_and_normal_batch(
        self,
        p_new: Any,
        prev_piece_idx: Any,
        xp: Any,
    ) -> Tuple[Any, Any]:
        """Versiunea vectorizată: reflectă toate particulele ieșite simultan.

        p_new:          (N, 3) — pozițiile propuse (unele ieșite)
        prev_piece_idx: (N,) int — piesa din care a plecat fiecare particulă
        xp:             modulul array (numpy sau cupy)

        Returnează (p_snap, n) ambele (N, 3).
        Particulele care nu au ieșit primesc n = 0 (caller verifică norma).
        """
        p_snap = p_new.copy()
        n_out = xp.zeros_like(p_new)

        for k, pc in enumerate(self.pieces):
            mask = prev_piece_idx == k
            if not xp.any(mask):
                continue
            ps, ns = pc.shape.snap_and_normal_batch(p_new[mask], xp)
            p_snap[mask] = ps
            n_out[mask] = ns
        return p_snap, n_out

    def wall_distance(self, p: np.ndarray, piece_idx: int) -> float:
        return self.pieces[piece_idx].shape.wall_distance(p)

    # ---- sampling ----

    def sample_uniform(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection sampling în bbox-ul uniunii.

        Pentru geometrii cu densități foarte asimetrice (uniune subțire într-o
        bbox mare), se poate face sampling per-piesă proporțional cu volumul,
        dar pentru cazurile curente bbox-ul global e suficient.
        """
        mn, mx = self.bbox()
        extent = mx - mn
        out = np.empty((N, 3), dtype=np.float64)
        filled = 0
        # batch rejection pentru eficiență
        batch = max(1024, 4 * N)
        while filled < N:
            cand = mn + rng.random((batch, 3)) * extent
            ok = self.inside_any(cand)
            taken = cand[ok]
            need = N - filled
            take = taken[:need]
            out[filled : filled + take.shape[0]] = take
            filled += take.shape[0]
        return out.astype(np.float32)

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        mns, mxs = [], []
        for pc in self.pieces:
            a, b = pc.shape.bbox()
            mns.append(a)
            mxs.append(b)
        mn = np.min(np.stack(mns, axis=0), axis=0)
        mx = np.max(np.stack(mxs, axis=0), axis=0)
        return mn, mx
