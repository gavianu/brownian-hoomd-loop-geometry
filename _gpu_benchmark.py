"""Benchmark CPU vs GPU pentru kernelele care pot fi vectorizate.

Testează:
  1. Pasul Langevin (step_velocity) — complet vectorizat, candidat ideal GPU
  2. bounce_batch OUBounce — vectorizat pe N particule simultan
  3. bounce_batch MaxwellDiffuse — vectorizat pe N particule simultan

Nota: _step_positions din engine are inca o bucla Python per-particula
pentru snap+normal (geometrie). Acesta NU este testat aici — necesita
vectorizarea snap_and_normal, care e urmatorul pas de optimizare.
"""
import sys, time, platform
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

print(f"{'='*60}")
print(f"GPU Benchmark — brownian_sim")
print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
print(f"CPU: {platform.processor()}")
print(f"N={N:,} particule, {STEPS} pasi")
print(f"{'='*60}\n")

# -- 1. verifica CuPy ------------------------------------------
try:
    import cupy as cp
    dev = cp.cuda.Device(0)
    dev.use()
    mem = dev.mem_info
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"     VRAM libera: {mem[0]/1e9:.1f} GB / {mem[1]/1e9:.1f} GB\n")
    HAS_GPU = True
except Exception as e:
    print(f"CuPy indisponibil: {e}")
    print("Ruleaza doar benchmark CPU.\n")
    HAS_GPU = False

xp_cpu = get_xp(-1)
xp_gpu = get_xp(0) if HAS_GPU else None

integrator = LangevinIntegrator(mass=m, gamma=gamma, kT=kT, dt=dt)
wall_ou = OUBounce(kT_over_m=kT/m)
wall_md = MaxwellDiffuse(kT_over_m=kT/m)
n_wall = np.array([0., 0., 1.], dtype=np.float32)  # normala spre z+

# -- 2. Benchmark Langevin step --------------------------------
print("-- 1. LangevinIntegrator.step_velocity --")
v_cpu = np.random.randn(N, 3).astype(np.float32)

# warmup
for _ in range(5):
    xi = np.random.randn(N, 3).astype(np.float32)
    v_cpu = integrator.step_velocity(v_cpu, xi)

t0 = time.perf_counter()
for _ in range(STEPS):
    xi = np.random.randn(N, 3).astype(np.float32)
    v_cpu = integrator.step_velocity(v_cpu, xi)
ms_langevin_cpu = (time.perf_counter() - t0) / STEPS * 1000
print(f"  CPU: {fmt(ms_langevin_cpu)}")

if HAS_GPU:
    v_gpu = cp.array(v_cpu)
    for _ in range(5):
        xi_g = cp.random.standard_normal((N, 3), dtype=cp.float32)
        v_gpu = integrator.step_velocity(v_gpu, xi_g)
    cp.cuda.stream.get_current_stream().synchronize()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        xi_g = cp.random.standard_normal((N, 3), dtype=cp.float32)
        v_gpu = integrator.step_velocity(v_gpu, xi_g)
    cp.cuda.stream.get_current_stream().synchronize()
    ms_langevin_gpu = (time.perf_counter() - t0) / STEPS * 1000
    print(f"  GPU: {fmt(ms_langevin_gpu)}  →  speedup {speedup(ms_langevin_cpu, ms_langevin_gpu)}")

# -- 3. Benchmark OUBounce batch ------------------------------─
print("\n-- 2. OUBounce.bounce_batch --")
v_in = np.random.randn(N, 3).astype(np.float32)
v_in[:, 2] = -np.abs(v_in[:, 2])  # viteza spre perete (vn < 0)
n_batch = np.tile(n_wall, (N, 1)).astype(np.float32)
e_n_arr = np.full(N, 0.9, dtype=np.float32)
bt_arr  = np.full(N, 0.8, dtype=np.float32)
rng = np.random.default_rng(42)

t0 = time.perf_counter()
for _ in range(STEPS):
    _ = wall_ou.bounce_batch(v_in, n_batch, e_n=e_n_arr, beta_t=bt_arr, xp=xp_cpu, rng=rng)
ms_ou_cpu = (time.perf_counter() - t0) / STEPS * 1000
print(f"  CPU: {fmt(ms_ou_cpu)}")

if HAS_GPU:
    v_in_g = cp.array(v_in)
    n_batch_g = cp.array(n_batch)
    e_n_g = cp.array(e_n_arr)
    bt_g  = cp.array(bt_arr)
    for _ in range(3):
        _ = wall_ou.bounce_batch(v_in_g, n_batch_g, e_n=e_n_g, beta_t=bt_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        _ = wall_ou.bounce_batch(v_in_g, n_batch_g, e_n=e_n_g, beta_t=bt_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    ms_ou_gpu = (time.perf_counter() - t0) / STEPS * 1000
    print(f"  GPU: {fmt(ms_ou_gpu)}  ->  speedup {speedup(ms_ou_cpu, ms_ou_gpu)}")

# -- 4. Benchmark MaxwellDiffuse batch ------------------------─
print("\n-- 3. MaxwellDiffuse.bounce_batch --")
e_n_md = np.ones(N, dtype=np.float32)
bt_md  = np.ones(N, dtype=np.float32)

t0 = time.perf_counter()
for _ in range(STEPS):
    _ = wall_md.bounce_batch(v_in, n_batch, e_n=e_n_md, beta_t=bt_md, xp=xp_cpu, rng=rng)
ms_md_cpu = (time.perf_counter() - t0) / STEPS * 1000
print(f"  CPU: {fmt(ms_md_cpu)}")

if HAS_GPU:
    e_n_md_g = cp.array(e_n_md)
    bt_md_g  = cp.array(bt_md)
    for _ in range(3):
        _ = wall_md.bounce_batch(v_in_g, n_batch_g, e_n=e_n_md_g, beta_t=bt_md_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        _ = wall_md.bounce_batch(v_in_g, n_batch_g, e_n=e_n_md_g, beta_t=bt_md_g, xp=xp_gpu)
    cp.cuda.stream.get_current_stream().synchronize()
    ms_md_gpu = (time.perf_counter() - t0) / STEPS * 1000
    print(f"  GPU: {fmt(ms_md_gpu)}  ->  speedup {speedup(ms_md_cpu, ms_md_gpu)}")

# -- 5. Sumar --------------------------------------------------
print(f"\n{'='*60}")
print("Sumar kernele vectorizate:")
print(f"  Langevin step:      CPU {fmt(ms_langevin_cpu)}", end="")
if HAS_GPU: print(f"  GPU {fmt(ms_langevin_gpu)}  ({speedup(ms_langevin_cpu, ms_langevin_gpu)})")
else: print()
print(f"  OUBounce batch:     CPU {fmt(ms_ou_cpu)}", end="")
if HAS_GPU: print(f"  GPU {fmt(ms_ou_gpu)}  ({speedup(ms_ou_cpu, ms_ou_gpu)})")
else: print()
print(f"  MaxwellDiffuse:     CPU {fmt(ms_md_cpu)}", end="")
if HAS_GPU: print(f"  GPU {fmt(ms_md_gpu)}  ({speedup(ms_md_cpu, ms_md_gpu)})")
else: print()
print(f"\nNota: snap_and_normal (detectia coliziunilor cu geometria) necesita")
print(f"vectorizare suplimentara pentru a beneficia de GPU — urmatorul pas.")
