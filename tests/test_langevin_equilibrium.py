"""Test: Langevin + OU bounce în simple_box ajunge la <v^2> = 3 kT/m."""
import numpy as np
import pytest

from brownian_sim.geometry.presets.simple_box import build as build_box
from brownian_sim.physics import make_wall_model
from brownian_sim.simulation import Simulation, SimulationConfig


def test_simple_box_ou_maintains_equilibrium():
    """Pornim de la MB, verificăm că <v^2> rămâne ~3 kT/m după relaxare."""
    assembly = build_box()
    cfg = SimulationConfig(
        n_particles=1000, steps=1000, dt=0.01,
        mass=1.0, gamma=1.0, kT=1.0, seed=7,
        velocity_init="maxwell_boltzmann",
        write_every=10_000, log_every=10_000, track_k=50,
        quiet=True,
    )
    wall = make_wall_model("ou", kT_over_m=cfg.kT / cfg.mass)
    sim = Simulation(assembly, wall, cfg)
    sim.run()

    v = sim.velocities.astype(np.float64)
    v2 = float(np.mean(np.sum(v * v, axis=1)))
    expected = 3.0 * cfg.kT / cfg.mass
    rel = abs(v2 - expected) / expected
    assert rel < 0.1, f"<v^2>={v2:.3f} expected {expected:.3f} (rel_err={rel:.3f})"


def test_simple_box_ou_relaxes_from_zero():
    """Pornim rece (v=0), verificăm că Langevin + OU încălzesc la kT."""
    assembly = build_box()
    cfg = SimulationConfig(
        n_particles=1000, steps=3000, dt=0.01,
        mass=1.0, gamma=1.0, kT=1.0, seed=11,
        velocity_init="zero",
        write_every=10_000, log_every=10_000, track_k=50,
        quiet=True,
    )
    wall = make_wall_model("ou", kT_over_m=cfg.kT / cfg.mass)
    sim = Simulation(assembly, wall, cfg)
    sim.run()

    v = sim.velocities.astype(np.float64)
    v2 = float(np.mean(np.sum(v * v, axis=1)))
    expected = 3.0
    rel = abs(v2 - expected) / expected
    assert rel < 0.15, f"<v^2>={v2:.3f} expected {expected:.3f}"


def test_no_particles_escape_loop_chambers():
    """Niciun OUT după 500 pași în loop_chambers (reflect robust)."""
    from brownian_sim.geometry.presets.loop_chambers import build as build_loop
    assembly = build_loop()
    cfg = SimulationConfig(
        n_particles=500, steps=500, dt=0.01,
        mass=1.0, gamma=1.0, kT=1.0, seed=13,
        velocity_init="maxwell_boltzmann",
        write_every=10_000, log_every=10_000, track_k=50,
        quiet=True,
    )
    wall = make_wall_model("ou", kT_over_m=cfg.kT / cfg.mass)
    sim = Simulation(assembly, wall, cfg)
    sim.run()

    # niciun OUT în piece_idx
    assert (sim.piece_idx >= 0).all(), \
        f"particule pierdute: {int((sim.piece_idx == -1).sum())}"
