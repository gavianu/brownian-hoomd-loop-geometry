"""Simulation engine: orchestrează Langevin + reflect + I/O.

Implementare CPU (numpy) optimizată cu reflecție batch per pas.
GPU: kernelele de fizică (Langevin, bounce_batch) sunt xp-agnostice și
gata pentru GPU, dar geometria SDF (inside_any, locate) rămâne pe CPU.
Extensia full-GPU necesită portarea geometriei — temă deschisă.

Design:
  - dependency injection: assembly, wall_model, writers
  - engine-ul nu știe de CSV/GSD direct — acceptă writers ca listă
  - SimulationConfig.device rezervat pentru extensia GPU
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
    velocity_init: str = "zero"

    # sub-stepping
    cfl: float = 0.5
    max_substeps: int = 32
    adapt_every: int = 1
    track_k: int = 200

    # output
    write_every: int = 1000
    log_every: int = 200
    quiet: bool = False

    # rezervat extensiei GPU (ignorat in implementarea curenta CPU)
    device: int = -1

    # writers injectati de caller
    writers: List = field(default_factory=list)


class Simulation:
    """Orchestrator CPU pentru rularea unei simulări complete."""

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

        self._write_frame(step=0)

        t_start = time.time()
        last_nsub = 1
        for step in range(1, cfg.steps + 1):
            # Langevin pe viteze
            xi = self.rng.standard_normal(size=self.positions.shape).astype(np.float32)
            self.velocities = self.integrator.step_velocity(self.velocities, xi)

            # decide nsub
            if cfg.adapt_every <= 1 or (step % cfg.adapt_every) == 0:
                nsub = compute_substeps(
                    self.positions, self.velocities, self.piece_idx,
                    cfg.dt, self.assembly, self.track_ids,
                    cfl=cfg.cfl, max_substeps=cfg.max_substeps,
                )
                last_nsub = nsub
            else:
                nsub = last_nsub

            sub_dt = cfg.dt / nsub
            for _ in range(nsub):
                self._step_positions(sub_dt, step)

            if step % cfg.write_every == 0:
                self._write_frame(step=step)
            if (step % cfg.log_every == 0 or step == cfg.steps) and not cfg.quiet:
                self._log_progress(step, t_start, nsub)

        if not cfg.quiet:
            print(f"\n[OK] done in {time.time() - t_start:.1f}s.")

    # ---------- low-level step ----------

    def _step_positions(self, sub_dt: float, step: int) -> None:
        """Un sub-pas pe poziții cu reflecție batch (numpy, CPU)."""
        p_old = self.positions
        p_new = p_old + self.velocities * sub_dt
        prev_idx = self.piece_idx.copy()

        # detectare particule ieșite
        in_union = self.assembly.inside_any(p_new)
        need = np.where(~in_union)[0]

        if need.size > 0:
            need_prev = prev_idx[need]

            # particule OUT (piece_idx < 0) — anulăm pasul
            invalid = need_prev < 0
            if invalid.any():
                p_new[need[invalid]] = p_old[need[invalid]]

            # particule valide — snap + bounce batch
            valid = ~invalid
            if valid.any():
                ids = need[valid]
                prev_k = need_prev[valid]

                p_snap, n_raw = self.assembly.snap_and_normal_batch(p_new[ids], prev_k, np)

                n_norm = np.sqrt((n_raw * n_raw).sum(axis=1, keepdims=True))
                has_n = n_norm[:, 0] > 1e-9
                n_safe = np.where(n_norm > 1e-9, n_norm, np.ones_like(n_norm))
                n_unit = np.where(has_n[:, None], n_raw / n_safe, np.zeros_like(n_raw))

                if has_n.any():
                    ids_hit = ids[has_n]
                    n_hit = n_unit[has_n]
                    e_n_arr = np.array(
                        [self.assembly.pieces[int(k)].material.e_n for k in prev_k[has_n]],
                        dtype=np.float32,
                    )
                    bt_arr = np.array(
                        [self.assembly.pieces[int(k)].material.beta_t for k in prev_k[has_n]],
                        dtype=np.float32,
                    )
                    v_out = self.wall_model.bounce_batch(
                        self.velocities[ids_hit], n_hit,
                        e_n=e_n_arr, beta_t=bt_arr, xp=np, rng=self.rng,
                    )
                    p_new[ids_hit] = p_snap[has_n] - 1e-6 * n_hit
                    self.velocities[ids_hit] = v_out

                # particule fara normala valida — lăsăm p_new neschimbat
                # (vor fi prinse de re-check sau de locate mai jos)

        # re-check: dacă încă e afară, revert la p_old (rar — colțuri/muchii)
        still_out = ~self.assembly.inside_any(p_new)
        if still_out.any():
            p_new[still_out] = p_old[still_out]

        self.positions = p_new

        # update piece index
        new_idx = self.assembly.locate(self.positions)
        # particulele care au bounced rămân în piesa veche
        if need.size > 0:
            new_idx[need] = prev_idx[need]

        # tranzitii
        crossed = new_idx != prev_idx
        if crossed.any():
            ids = np.where(crossed)[0]
            self._on_transitions(step, ids, prev_idx, new_idx)

        self.piece_idx = new_idx

    # ---------- hooks I/O ----------

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
        print(
            f"[{step}/{self.config.steps}] {rate:6.1f} steps/s  "
            f"elapsed {self._fmt_time(elapsed)}  ETA {self._fmt_time(eta)}  nsub={nsub}",
            flush=True,
        )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if not math.isfinite(seconds):
            return "-"
        s = max(0.0, float(seconds))
        m, s = divmod(int(round(s)), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m}m{s}s"
        if m:
            return f"{m}m{s}s"
        return f"{s}s"
