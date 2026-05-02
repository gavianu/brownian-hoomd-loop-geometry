"""Validare rapida engine batch: <v2> si timp per pas."""
import sys, time, yaml, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from brownian_sim.simulation.engine import Simulation, SimulationConfig
from brownian_sim.physics import make_wall_model
from brownian_sim.physics.backend import to_cpu
from brownian_sim.scripts.run_sim import _build_assembly

with open("configs/loop_maxwell_hetero_short.yaml") as f:
    cfg_dict = yaml.safe_load(f)

cfg_dict["simulation"]["steps"] = 500
cfg_dict["simulation"]["quiet"] = True
cfg_dict["output"]["gsd"] = False
cfg_dict["output"]["csv"] = False

assembly = _build_assembly(cfg_dict["geometry"])
sim_cfg = SimulationConfig(**cfg_dict["simulation"])
wall = make_wall_model(cfg_dict["wall_model"], kT_over_m=sim_cfg.kT / sim_cfg.mass)
sim = Simulation(assembly, wall, sim_cfg)

t0 = time.perf_counter()
sim.run()
dt = time.perf_counter() - t0

v_cpu = to_cpu(sim.velocities)
v2 = float((v_cpu ** 2).sum(axis=1).mean())
print(f"500 pasi in {dt:.1f}s = {dt/500*1000:.1f} ms/pas")
print(f"<v2> = {v2:.3f}  (tinta 3.000)")
print("OK" if 2.5 < v2 < 3.5 else "FAIL — v2 in afara intervalului")
