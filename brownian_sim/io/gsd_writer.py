"""Writer GSD pentru traiectorii — format binar, citit de OVITO."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import gsd.hoomd  # type: ignore[import]
    _HAS_GSD = True
except Exception:
    _HAS_GSD = False


class GSDWriter:
    def __init__(
        self,
        path: str,
        box: tuple[float, float, float] = (520.0, 320.0, 260.0),
        subset: int = 15_000,
        track_ids: Optional[np.ndarray] = None,
    ) -> None:
        if not _HAS_GSD:
            raise RuntimeError("gsd nu este instalat — nu pot scrie GSD")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        self.path = path
        self.box = box
        self.subset = subset
        self.track_ids = track_ids if track_ids is not None else np.empty(0, dtype=np.int64)
        self._f = gsd.hoomd.open(name=path, mode="w")

    def write_frame(self, step, positions, velocities, piece_idx, assembly):
        N = positions.shape[0]
        subset = min(self.subset if self.subset > 0 else N, N)
        sel = np.arange(subset, dtype=np.int64)

        frame = gsd.hoomd.Frame()
        frame.configuration.step = int(step)
        frame.configuration.box = [self.box[0], self.box[1], self.box[2], 0, 0, 0]
        frame.particles.N = subset
        frame.particles.position = positions[sel].astype(np.float32)
        frame.particles.velocity = velocities[sel].astype(np.float32)
        frame.particles.types = ["He", "HeSel"]
        typeid = np.zeros(subset, dtype=np.int32)
        typeid[np.intersect1d(sel, self.track_ids, assume_unique=False)] = 1
        frame.particles.typeid = typeid
        frame.particles.diameter = np.full(subset, 1.0, dtype=np.float32)
        self._f.append(frame)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
