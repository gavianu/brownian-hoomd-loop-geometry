"""Benchmark: timp per pas pentru N=30000 particule in loop_chambers."""
import time, sys, platform
import numpy as np
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from brownian_sim.simulation.engine import Simulation, SimulationConfig
from brownian_sim.physics import make_wall_model
from brownian_sim.scripts.run_sim import _build_assembly

# --- sistem ---
print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
print(f"Platform: {platform.processor()}")
print()

# --- 1. Langevin pur (fara geometrie, benchmark lower bound) ---
N = 30000
v = np.random.randn(N, 3)
r = np.random.rand(N, 3) * 60
gamma, kT, m, dt = 1.0, 1.0, 1.0, 0.001
noise_amp = (2 * gamma * kT / m * dt) ** 0.5
STEPS = 300
t0 = time.perf_counter()
for _ in range(STEPS):
    v += (-gamma/m * dt) * v + noise_amp * np.random.randn(N, 3)
    r += v * dt
t1 = time.perf_counter()
ms_langevin = (t1 - t0) / STEPS * 1000
print(f"1. Langevin pur (fara geometrie):   {ms_langevin:.2f} ms/pas  (N={N})")

# --- 2. Simulare completa cu geometrie loop_chambers ---
with open("configs/loop_ou_hetero_long.yaml") as f:
    cfg_dict = yaml.safe_load(f)

BENCH_STEPS = 100
cfg_dict["simulation"]["steps"] = BENCH_STEPS
cfg_dict["simulation"]["write_every"] = 99999
cfg_dict["simulation"]["log_every"] = 99999
cfg_dict["simulation"]["quiet"] = True
cfg_dict["output"]["dir"] = "sim_out/_benchmark_tmp"
cfg_dict["output"]["gsd"] = False
cfg_dict["output"]["csv"] = False

assembly = _build_assembly(cfg_dict["geometry"])
sim_cfg_raw = cfg_dict["simulation"].copy()
sim_cfg = SimulationConfig(**sim_cfg_raw)
wall = make_wall_model(cfg_dict["wall_model"], kT_over_m=sim_cfg.kT / sim_cfg.mass)
track_ids = np.arange(min(sim_cfg.track_k, sim_cfg.n_particles), dtype=np.int64)
sim_cfg.writers = []

sim = Simulation(assembly, wall, sim_cfg)
t0 = time.perf_counter()
sim.run()
t1 = time.perf_counter()
ms_full = (t1 - t0) / BENCH_STEPS * 1000

print(f"2. Simulare completa (loop_chambers): {ms_full:.1f} ms/pas  (N={N})")
print()
print(f"   Overhead geometrie + coliziuni:  {ms_full / ms_langevin:.1f}x fata de Langevin pur")
print(f"   Estimat 300k pasi (N=30k):       {ms_full * 300000 / 3600000:.1f} ore pe acest sistem")
print(f"   Estimat 300k pasi (N=10k):       {ms_full * 300000 / 3600000 / 3:.1f} ore (3x mai putine particule)")
print()
print("Scalare:")
print(f"   GPU (CuPy, ~50x NumPy):          ~{ms_full * 300000 / 3600000 / 50:.2f} ore estimate")
print(f"   N=100k (statistici mai bune):    ~{ms_full * 300000 / 3600000 * (100/30):.1f} ore (liniar in N)")
