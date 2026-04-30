"""Simulation engine: orchestrează Langevin + reflect + I/O.

Path curent: **CPU-only**, clar și corect, cu reflecții robuste per-particulă.
GPU-ul se adaugă ulterior ca o altă implementare a reflect-ului (vezi
backend.py pentru abstracția xp).

Design:
  - dependency injection: assembly, wall_model, integrator, writers, sampler
  - engine-ul nu știe de CSV/GSD direct — acceptă writers ca listă
  - timpul de simulare e controlat de SimulationConfig (nu hardcodat)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from brownian_sim.geometry.assembly import Assembly
from brownian_sim.physics.dynamics import LangevinIntegrator
from brownian_sim.physics.wall_models import WallModel
from brownian_sim.simulation.sampler import sample_positions, sample_velocities
from brownian_sim.simulation.substepping import compute_substeps


@dataclass
class SimulationConfig:
    # dinamică
    n_particles: int = 30_000
    steps: int = 20_000
    dt: float = 0.001
    mass: float = 1.0
    gamma: float = 1.0
    kT: float = 1.0
    seed: int = 42

    # init
    velocity_init: str = "zero"  # "zero" | "maxwell_boltzmann"

    # sub-stepping
    cfl: float = 0.5
    max_substeps: int = 32
    adapt_every: int = 1
    track_k: int = 200

    # output
    write_every: int = 1000
    log_every: int = 200
    quiet: bool = False

    # I/O writers fields populated by engine caller
    writers: List = field(default_factory=list)


class Simulation:
    """Orchestrator pentru rularea unei simulări complete."""

    def __init__(
        self,
        assembly: Assembly,
        wall_model: WallModel,
        config: SimulationConfig,
    ) -> None:
        self.assembly = assembly
        self.wall_model = wall_model
        self.config = config
        self.integrator = LangevinIntegrator(
            mass=config.mass, gamma=config.gamma, kT=config.kT, dt=config.dt
        )
        self.rng = np.random.default_rng(config.seed)

        # state (alocat în reset())
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.piece_idx: Optional[np.ndarray] = None
        self.track_ids: Optional[np.ndarray] = None

    # ---------- initialization ----------

    def reset(self) -> None:
        cfg = self.config
        self.positions = sample_positions(self.assembly, cfg.n_particles, self.rng)
        self.velocities = sample_velocities(
            cfg.n_particles, cfg.velocity_init, self.integrator.kT_over_m, self.rng
        )
        self.piece_idx = self.assembly.locate(self.positions)
        track_k = min(cfg.track_k, cfg.n_particles)
        self.track_ids = np.arange(track_k, dtype=np.int32)

    # ---------- main loop ----------

    def run(self) -> None:
        if self.positions is None:
            self.reset()
        cfg = self.config
        assert self.positions is not None
        assert self.velocities is not None
        assert self.piece_idx is not None
        assert self.track_ids is not None

        # write frame 0
        self._write_frame(step=0)

        t_start = time.time()
        last_nsub = 1
        for step in range(1, cfg.steps + 1):
            # --- Langevin pe viteze ---
            xi = self.rng.standard_normal(size=self.positions.shape).astype(np.float32)
            self.velocities = self.integrator.step_velocity(self.velocities, xi)

            # --- decide nsub ---
            if cfg.adapt_every <= 1 or (step % cfg.adapt_every) == 0:
                nsub = compute_substeps(
                    self.positions,
                    self.velocities,
                    self.piece_idx,
                    cfg.dt,
                    self.assembly,
                    self.track_ids,
                    cfl=cfg.cfl,
                    max_substeps=cfg.max_substeps,
                )
                last_nsub = nsub
            else:
                nsub = last_nsub

            sub_dt = cfg.dt / nsub
            for _ in range(nsub):
                self._step_positions(sub_dt, step)

            # --- output + log ---
            if step % cfg.write_every == 0:
                self._write_frame(step=step)
            if (step % cfg.log_every == 0 or step == cfg.steps) and not cfg.quiet:
                self._log_progress(step, t_start, nsub)

        if not cfg.quiet:
            print(f"\n[OK] done in {time.time() - t_start:.1f}s.")

    # ---------- low-level step ----------

    def _step_positions(self, sub_dt: float, step: int) -> None:
        """Un sub-pas pe poziții cu reflecție per-particulă.

        Implementare CPU robustă: detectăm particulele ieșite din piesa
        curentă; fac snap + normal + bounce; dacă după bounce tot sunt
        afară (ex. geometrie locală complicată), revin la p_old (eveniment
        rar, log warning).
        """
        p_old = self.positions
        p_new = p_old + self.velocities * sub_dt
        prev_idx = self.piece_idx.copy()

        # check care sunt încă în uniune
        in_union_after = self.assembly.inside_any(p_new)

        # particulele care au ieșit trebuie reflectate
        need = np.where(~in_union_after)[0]
        if need.size > 0:
            for i in need:
                k = int(prev_idx[i])
                if k < 0:
                    # particula era deja OUT înainte — anulează pasul
                    p_new[i] = p_old[i]
                    continue
                p_snap, n = self.assembly.snap_and_normal(p_new[i], k)
                if np.linalg.norm(n) < 1e-9:
                    # piesa zice că e înăuntru, dar uniunea zice că e afară:
                    # probabil a trecut în altă piesă prin seal; lăsăm piece_idx
                    # să fie reasignat mai jos
                    continue
                # asigură n unitar
                n = n / np.linalg.norm(n)
                v_out = self.wall_model.bounce_single(
                    v_in=self.velocities[i].astype(np.float64),
                    n=n,
                    material=self.assembly.material(k),
                    rng=self.rng,
                )
                # micro-offset în interior pentru evitarea re-hit-ului imediat
                p_new[i] = p_snap - 1e-6 * n
                self.velocities[i] = v_out.astype(np.float32)

        # re-check: dacă încă e afară, revino la p_old (rar)
        still_out = ~self.assembly.inside_any(p_new)
        if still_out.any():
            p_new[still_out] = p_old[still_out]

        self.positions = p_new.astype(np.float32)

        # update piece index (detect tranziții între piese)
        new_idx = self.assembly.locate(self.positions)
        # pentru particulele care tocmai au făcut bounce, fixăm la piesa veche
        # (bounce-ul înseamnă că sunt tot în piesa originală, nu au trecut)
        if need.size > 0:
            fixed = need[self.piece_idx[need] != -1 if False else np.ones(need.size, dtype=bool)]
            new_idx[need] = prev_idx[need]

        # log tranziții (doar particulele care n-au făcut bounce)
        crossed_mask = new_idx != prev_idx
        if crossed_mask.any():
            ids = np.where(crossed_mask)[0]
            self._on_transitions(step, ids, prev_idx, new_idx)

        self.piece_idx = new_idx

    # ---------- hooks pentru I/O ----------

    def _write_frame(self, step: int) -> None:
        for w in self.config.writers:
            if hasattr(w, "write_frame"):
                w.write_frame(step=step, positions=self.positions,
                              velocities=self.velocities, piece_idx=self.piece_idx,
                              assembly=self.assembly)

    def _on_transitions(self, step, ids, prev_idx, new_idx) -> None:
        for w in self.config.writers:
            if hasattr(w, "write_transitions"):
                w.write_transitions(step=step, ids=ids, prev_idx=prev_idx,
                                    new_idx=new_idx, names=self.assembly.names)

    # ---------- logging ----------

    def _log_progress(self, step: int, t_start: float, nsub: int) -> None:
        elapsed = time.time() - t_start
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (self.config.steps - step) / rate if rate > 0 else float("inf")
        eta_str = self._fmt_time(eta)
        print(
            f"[{step}/{self.config.steps}] {rate:6.1f} steps/s  "
            f"elapsed {self._fmt_time(elapsed)}  ETA {eta_str}  nsub={nsub}",
            flush=True,
        )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if not math.isfinite(seconds):
            return "—"
        s = max(0.0, float(seconds))
        m, s = divmod(int(round(s)), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m}m{s}s"
        if m:
            return f"{m}m{s}s"
        return f"{s}s"
