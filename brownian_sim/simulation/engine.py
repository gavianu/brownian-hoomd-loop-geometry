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
from brownian_sim.physics.backend import get_xp, to_cpu
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

    # device: -1 = CPU (numpy), 0+ = GPU index (cupy)
    device: int = -1

    # I/O writers fields populated by engine caller
    writers: List = field(default_factory=list)


class Simulation:
    """Orchestrator pentru rularea unei simulări complete (CPU sau GPU)."""

    def __init__(
        self,
        assembly: Assembly,
        wall_model: WallModel,
        config: SimulationConfig,
    ) -> None:
        self.assembly = assembly
        self.wall_model = wall_model
        self.config = config
        self.xp = get_xp(config.device)
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
        xp = self.xp
        pos_cpu = sample_positions(self.assembly, cfg.n_particles, self.rng)
        vel_cpu = sample_velocities(
            cfg.n_particles, cfg.velocity_init, self.integrator.kT_over_m, self.rng
        )
        idx_cpu = self.assembly.locate(pos_cpu)
        self.positions = xp.asarray(pos_cpu)
        self.velocities = xp.asarray(vel_cpu)
        self.piece_idx = xp.asarray(idx_cpu)
        track_k = min(cfg.track_k, cfg.n_particles)
        self.track_ids = xp.arange(track_k, dtype=xp.int32)

    # ---------- main loop ----------

    def run(self) -> None:
        if self.positions is None:
            self.reset()
        cfg = self.config
        xp = self.xp

        self._write_frame(step=0)

        t_start = time.time()
        last_nsub = 1
        for step in range(1, cfg.steps + 1):
            # Langevin pe viteze (xp-agnostic)
            xi = xp.asarray(
                self.rng.standard_normal(size=self.positions.shape).astype(np.float32)
            )
            self.velocities = self.integrator.step_velocity(self.velocities, xi)

            # decide nsub (pe CPU — folosim to_cpu pentru substepping care e tot CPU)
            if cfg.adapt_every <= 1 or (step % cfg.adapt_every) == 0:
                nsub = compute_substeps(
                    to_cpu(self.positions),
                    to_cpu(self.velocities),
                    to_cpu(self.piece_idx),
                    cfg.dt,
                    self.assembly,
                    to_cpu(self.track_ids),
                    cfl=cfg.cfl,
                    max_substeps=cfg.max_substeps,
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
        """Un sub-pas pe poziții cu reflecție batch (xp-agnostic, CPU sau GPU)."""
        xp = self.xp
        p_old = self.positions
        p_new = p_old + self.velocities * sub_dt
        prev_idx = self.piece_idx.copy()

        # detectare ieșiri (pe CPU — inside_any folosește numpy)
        p_new_cpu = to_cpu(p_new)
        in_union = self.assembly.inside_any(p_new_cpu)
        need_cpu = np.where(~in_union)[0]

        if need_cpu.size > 0:
            # snap + normal batch per piesă (xp-agnostic)
            prev_idx_cpu = to_cpu(prev_idx)
            need_prev = prev_idx_cpu[need_cpu]

            # particulele cu piece_idx < 0 (deja OUT) — anulăm pasul
            valid = need_prev >= 0
            if valid.any():
                ids_valid = need_cpu[valid]
                p_need = xp.asarray(p_new_cpu[ids_valid])
                prev_need = xp.asarray(need_prev[valid])

                p_snap, n_raw = self.assembly.snap_and_normal_batch(p_need, prev_need, xp)

                # normalizare + filtrare normale valide
                n_norm = xp.sqrt(xp.sum(n_raw * n_raw, axis=1, keepdims=True))
                has_normal = (n_norm[:, 0] > 1e-9)

                if xp.any(has_normal):
                    n_unit = xp.where(
                        has_normal[:, None],
                        n_raw / xp.where(n_norm > 1e-9, n_norm, xp.ones_like(n_norm)),
                        xp.zeros_like(n_raw),
                    )
                    # bounce batch — material per piesă
                    e_n_arr = xp.asarray(np.array(
                        [self.assembly.pieces[int(k)].material.e_n for k in to_cpu(prev_need[valid])],
                        dtype=np.float32,
                    ))
                    bt_arr = xp.asarray(np.array(
                        [self.assembly.pieces[int(k)].material.beta_t for k in to_cpu(prev_need[valid])],
                        dtype=np.float32,
                    ))
                    v_need = self.velocities[xp.asarray(ids_valid)]
                    v_out = self.wall_model.bounce_batch(
                        v_need, n_unit, e_n=e_n_arr, beta_t=bt_arr, xp=xp, rng=self.rng
                    )
                    # micro-offset pentru a evita re-hit imediat
                    p_snap_offset = p_snap - xp.asarray(1e-6, dtype=p_snap.dtype) * n_unit
                    p_new = p_new.copy()
                    ids_valid_xp = xp.asarray(ids_valid)
                    p_new[ids_valid_xp] = xp.where(
                        has_normal[:, None], p_snap_offset, p_new[ids_valid_xp]
                    )
                    self.velocities = self.velocities.copy()
                    self.velocities[ids_valid_xp] = xp.where(
                        has_normal[:, None], v_out, self.velocities[ids_valid_xp]
                    )

            # particulele OUT (k<0) — anulăm pasul
            invalid = ~valid
            if invalid.any():
                ids_inv_xp = xp.asarray(need_cpu[invalid])
                p_new[ids_inv_xp] = p_old[ids_inv_xp]

        # re-check final: dacă încă e afară, revert (rar — geometrie complexă)
        still_out = ~self.assembly.inside_any(to_cpu(p_new))
        if still_out.any():
            so_xp = xp.asarray(np.where(still_out)[0])
            p_new[so_xp] = p_old[so_xp]

        self.positions = p_new

        # update piece index pe CPU, convertit înapoi
        new_idx_cpu = self.assembly.locate(to_cpu(self.positions))
        # particulele care au bounced rămân în piesa veche
        if need_cpu.size > 0:
            prev_idx_cpu = to_cpu(prev_idx)
            new_idx_cpu[need_cpu] = prev_idx_cpu[need_cpu]
        new_idx = xp.asarray(new_idx_cpu)

        # tranzitii
        crossed_mask_cpu = new_idx_cpu != to_cpu(prev_idx)
        if crossed_mask_cpu.any():
            ids = np.where(crossed_mask_cpu)[0]
            self._on_transitions(step, ids, to_cpu(prev_idx), new_idx_cpu)

        self.piece_idx = new_idx

    # ---------- hooks pentru I/O ----------

    def _write_frame(self, step: int) -> None:
        for w in self.config.writers:
            if hasattr(w, "write_frame"):
                w.write_frame(step=step, positions=to_cpu(self.positions),
                              velocities=to_cpu(self.velocities),
                              piece_idx=to_cpu(self.piece_idx),
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
