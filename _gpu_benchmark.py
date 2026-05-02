"""Benchmark CPU vs GPU pentru kernelele vectorizate + simulare completa.

Salveaza rezultatele in results/benchmark_<hostname>_<date>.txt
pentru a putea fi commitate si comparate intre masini.
"""
import sys, time, platform, socket, datetime
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from brownian_sim.physics.dynamics import LangevinIntegrator
from brownian_sim.physics.wall_models import OUBounce, MaxwellDiffuse
from brownian_sim.physics.backend import get_xp, to_cpu

def fmt(ms): return f"{ms:.2f} ms"
def speedup(cpu, gpu): return f"{cpu/gpu:.1f}x"

N = 100_000
STEPS = 500
kT, m, gamma, dt = 1.0, 1.0, 1.0, 0.001

lines = []
def out(s=""):
    print(s)
    lines.append(s)

out("=" * 60)
out("GPU Benchmark -- brownian_sim")
out(f"Host:    {socket.gethostname()}")
out(f"Date:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
out(f"Python:  {sys.version.split()[0]}")
out(f"NumPy:   {np.__version__}")
out(f"CPU:     {platform.processor()}")
out(f"N={N:,} particule, {STEPS} pasi per kernel")
out("=" * 60)
out()

# -- verifica CuPy -----------------------------------------------
try:
    import cupy as cp
    dev = cp.cuda.Device(0)
    dev.use()
    mem = dev.mem_info
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    out(f"GPU:     {gpu_name}")
    out(f"         VRAM libera: {mem[0]/1e9:.1f} GB / {mem[1]/1e9:.1f} GB")
    out(f"CuPy:    {cp.__version__}")
    HAS_GPU = True
except Exception as e:
    out(f"CuPy: indisponibil ({e})")
    out("Ruleaza doar benchmark CPU.")
    HAS_GPU = False
out()

xp_cpu = get_xp(-1)
xp_gpu = get_xp(0) if HAS_GPU else None

integrator = LangevinIntegrator(mass=m, gamma=gamma, kT=kT, dt=dt)
wall_ou = OUBounce(kT_over_m=kT/m)
wall_md = MaxwellDiffuse(kT_over_m=kT/m)
n_wall = np.array([0., 0., 1.], dtype=np.float32)

# -- 1. Langevin step --------------------------------------------
out("-- 1. LangevinIntegrator.step_velocity --")
v_cpu = np.random.randn(N, 3).astype(np.float32)
for _ in range(5):
    v_cpu = integrator.step_velocity(v_cpu, np.random.randn(N, 3).astype(np.float32))
t0 = time.perf_counter()
for _ in range(STEPS):
    v_cpu = integrator.step_velocity(v_cpu, np.random.randn(N, 3).astype(np.float32))
ms_langevin_cpu = (time.perf_counter() - t0) / STEPS * 1000
out(f"  CPU: {fmt(ms_langevin_cpu)}")

ms_langevin_gpu = None
if HAS_GPU:
    v_gpu = cp.array(v_cpu)
    for _ in range(5):
        v_gpu = integrator.step_velocity(v_gpu, cp.random.standard_normal((N, 3), dtype=cp.float32))
    cp.cuda.stream.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        v_gpu = integrator.step_velocity(v_gpu, cp.random.standard_normal((N, 3), dtype=cp.float32))
    cp.cuda.stream.get_current_stream().synchronize()
    ms_langevin_gpu = (time.perf_counter() - t0) / STEPS * 1000
    out(f"  GPU: {fmt(ms_langevin_gpu)}  ->  speedup {speedup(ms_langevin_cpu, ms_langevin_gpu)}")

# -- 2. OUBounce batch -------------------------------------------
out()
out("-- 2. OUBounce.bounce_batch --")
v_in = np.random.randn(N, 3).astype(np.float32)
v_in[:, 2] = -np.abs(v_in[:, 2])
n_batch = np.tile(n_wall, (N, 1)).astype(np.float32)
e_n_arr = np.full(N, 0.9, dtype=np.float32)
bt_arr  = np.full(N, 0.8, dtype=np.float32)
rng = np.random.default_rng(42)
for _ in range(3):
    wall_ou.bounce_batch(v_in, n_batch, e_n=e_n_arr, beta_t=bt_arr, xp=xp_cpu, rng=rng)
t0 = time.perf_counter()
for _ in range(STEPS):
    wall_ou.bounce_batch(v_in, n_batch, e_n=e_n_arr, beta_t=bt_arr, xp=xp_cpu, rng=rng)
ms_ou_cpu = (time.perf_counter() - t0) / STEPS * 1000
out(f"  CPU: {fmt(ms_ou_cpu)}")

ms_ou_gpu = None
if HAS_GPU:
    v_in_g = cp.array(v_in); n_g = cp.array(n_batch)
    e_n_g = cp.array(e_n_arr); bt_g = cp.array(bt_arr)
    for _ in range(3):
        wall_ou.bounce_batch(v_in_g, n_g, e_n=e_n_g, beta_t=bt_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        wall_ou.bounce_batch(v_in_g, n_g, e_n=e_n_g, beta_t=bt_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    ms_ou_gpu = (time.perf_counter() - t0) / STEPS * 1000
    out(f"  GPU: {fmt(ms_ou_gpu)}  ->  speedup {speedup(ms_ou_cpu, ms_ou_gpu)}")

# -- 3. MaxwellDiffuse batch -------------------------------------
out()
out("-- 3. MaxwellDiffuse.bounce_batch --")
e_n_md = np.ones(N, dtype=np.float32)
bt_md  = np.ones(N, dtype=np.float32)
for _ in range(3):
    wall_md.bounce_batch(v_in, n_batch, e_n=e_n_md, beta_t=bt_md, xp=xp_cpu, rng=rng)
t0 = time.perf_counter()
for _ in range(STEPS):
    wall_md.bounce_batch(v_in, n_batch, e_n=e_n_md, beta_t=bt_md, xp=xp_cpu, rng=rng)
ms_md_cpu = (time.perf_counter() - t0) / STEPS * 1000
out(f"  CPU: {fmt(ms_md_cpu)}")

ms_md_gpu = None
if HAS_GPU:
    e_n_md_g = cp.array(e_n_md); bt_md_g = cp.array(bt_md)
    for _ in range(3):
        wall_md.bounce_batch(v_in_g, n_g, e_n=e_n_md_g, beta_t=bt_md_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        wall_md.bounce_batch(v_in_g, n_g, e_n=e_n_md_g, beta_t=bt_md_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    ms_md_gpu = (time.perf_counter() - t0) / STEPS * 1000
    out(f"  GPU: {fmt(ms_md_gpu)}  ->  speedup {speedup(ms_md_cpu, ms_md_gpu)}")

# -- 4. Simulare completa (engine batch) -------------------------
out()
out("-- 4. Simulare completa loop_chambers (engine batch) --")
import yaml as _yaml
from brownian_sim.simulation.engine import Simulation, SimulationConfig
from brownian_sim.physics import make_wall_model
from brownian_sim.scripts.run_sim import _build_assembly

BENCH_STEPS = 200
for dev_id, label in [(-1, "CPU")]:  # GPU adaugat automat mai jos
    with open("configs/loop_ou_hetero_long.yaml") as f:
        cd = _yaml.safe_load(f)
    cd["simulation"]["steps"] = BENCH_STEPS
    cd["simulation"]["write_every"] = 99999
    cd["simulation"]["log_every"] = 99999
    cd["simulation"]["quiet"] = True
    cd["simulation"]["device"] = dev_id
    cd["output"]["dir"] = f"sim_out/_bench_{label.lower()}"
    cd["output"]["gsd"] = False
    cd["output"]["csv"] = False
    asm = _build_assembly(cd["geometry"])
    sc = SimulationConfig(**cd["simulation"])
    wl = make_wall_model(cd["wall_model"], kT_over_m=sc.kT / sc.mass)
    sc.writers = []
    sim = Simulation(asm, wl, sc)
    t0 = time.perf_counter()
    sim.run()
    ms_sim = (time.perf_counter() - t0) / BENCH_STEPS * 1000
    est_h = ms_sim * 300000 / 3600000
    out(f"  {label}: {ms_sim:.1f} ms/pas  (300k pasi ~ {est_h:.1f} ore)")

if HAS_GPU:
    with open("configs/loop_ou_hetero_long.yaml") as f:
        cd = _yaml.safe_load(f)
    cd["simulation"]["steps"] = BENCH_STEPS
    cd["simulation"]["write_every"] = 99999
    cd["simulation"]["log_every"] = 99999
    cd["simulation"]["quiet"] = True
    cd["simulation"]["device"] = 0
    cd["output"]["dir"] = "sim_out/_bench_gpu"
    cd["output"]["gsd"] = False
    cd["output"]["csv"] = False
    asm = _build_assembly(cd["geometry"])
    sc = SimulationConfig(**cd["simulation"])
    wl = make_wall_model(cd["wall_model"], kT_over_m=sc.kT / sc.mass)
    sc.writers = []
    sim = Simulation(asm, wl, sc)
    # warmup
    sim.reset()
    t0 = time.perf_counter()
    sim.run()
    ms_sim_gpu = (time.perf_counter() - t0) / BENCH_STEPS * 1000
    est_h_gpu = ms_sim_gpu * 300000 / 3600000
    out(f"  GPU: {ms_sim_gpu:.1f} ms/pas  (300k pasi ~ {est_h_gpu:.1f} ore)")
    out(f"  Speedup simulare completa: {speedup(ms_sim, ms_sim_gpu)}")
    out(f"  Nota: bottleneck ramas = inside_any/locate (CPU), transferuri xp<->np")

# -- Sumar -------------------------------------------------------
out()
out("=" * 60)
out("SUMAR")
out(f"  Langevin step (N=100k):  CPU {fmt(ms_langevin_cpu)}" +
    (f"  GPU {fmt(ms_langevin_gpu)}  ({speedup(ms_langevin_cpu, ms_langevin_gpu)})" if ms_langevin_gpu else ""))
out(f"  OUBounce batch:          CPU {fmt(ms_ou_cpu)}" +
    (f"  GPU {fmt(ms_ou_gpu)}  ({speedup(ms_ou_cpu, ms_ou_gpu)})" if ms_ou_gpu else ""))
out(f"  MaxwellDiffuse batch:    CPU {fmt(ms_md_cpu)}" +
    (f"  GPU {fmt(ms_md_gpu)}  ({speedup(ms_md_cpu, ms_md_gpu)})" if ms_md_gpu else ""))
out()
out("Nota: inside_any si locate raman pe CPU (geometrie SDF).")
out("Urmatorul pas: vectorizare inside_batch cu xp pentru full GPU.")
out("=" * 60)

# -- Salveaza in results/ ----------------------------------------
Path("results").mkdir(exist_ok=True)
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
host = socket.gethostname().replace(" ", "_")
out_path = Path(f"results/benchmark_{host}_{date_str}.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nRaport salvat in: {out_path}")
print("Ruleaza: git add results/ && git commit -m 'Benchmark GPU results' && git push")
