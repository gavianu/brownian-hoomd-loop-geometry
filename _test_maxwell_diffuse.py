"""Test Maxwell diffuse: verifica MB si bilant detaliat microscopic."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from brownian_sim.physics.wall_models import MaxwellDiffuse, OUBounce
from brownian_sim.materials.wall_material import WallMaterial

rng = np.random.default_rng(42)
kT, m = 1.0, 1.0
s = np.sqrt(kT / m)
mat = WallMaterial(e_n=0.9, beta_t=0.8)
n = np.array([0.0, 0.0, 1.0])  # normala pe z

N = 100_000

# --- 1. Distributia de iesire e Maxwell-Boltzmann? ---
print("=== 1. Distributie MB la iesire ===")
for Model, label in [(MaxwellDiffuse(kT/m), "MaxwellDiffuse"),
                     (OUBounce(kT/m),       "OUBounce     ")]:
    v_out = []
    for _ in range(N):
        v_in = rng.standard_normal(3) * s
        v_in[2] = -abs(v_in[2])  # vine din exterior (v_n < 0)
        vo = Model.bounce_single(v_in, n, mat, rng)
        v_out.append(vo)
    v_out = np.array(v_out)
    vx_std = float(v_out[:, 0].std())
    vy_std = float(v_out[:, 1].std())
    vn_mean = float(v_out[:, 2].mean())
    v2_mean = float(np.mean(np.sum(v_out**2, axis=1)))
    print(f"  {label}: sigma_vx={vx_std:.4f} sigma_vy={vy_std:.4f} "
          f"<vn>={vn_mean:.4f} <v2>={v2_mean:.4f} (teoretic: sigma=1.0, <v2>=3.0)")

# --- 2. Bilant detaliat microscopic ---
# P(v_in -> v_out) * f_eq(v_in) = P(v_out_time_reversed -> v_in_time_reversed) * f_eq(v_out)
# Test proxy: pentru Maxwell, distributia conditionata de iesire NU depinde de v_in
# Verifica: <v_out> independent de |v_in|
print("\n=== 2. Independenta v_out de v_in (test bilant detaliat) ===")
md = MaxwellDiffuse(kT/m)
ou = OUBounce(kT/m)

for Model, label in [(md, "MaxwellDiffuse"), (ou, "OUBounce     ")]:
    # grup dupa viteza normala incidenta: mica vs mare
    vn_out_slow, vn_out_fast = [], []
    for _ in range(N):
        speed = rng.exponential(s) + 0.1
        v_in = rng.standard_normal(3) * 0.1
        v_in[2] = -speed
        vo = Model.bounce_single(v_in, n, mat, rng)
        vn_out_slow.append(vo[2])

        v_in2 = rng.standard_normal(3) * 0.1
        v_in2[2] = -3 * speed
        vo2 = Model.bounce_single(v_in2, n, mat, rng)
        vn_out_fast.append(vo2[2])

    print(f"  {label}: <vn_out|v_in_slow>={np.mean(vn_out_slow):.4f}  "
          f"<vn_out|v_in_fast>={np.mean(vn_out_fast):.4f}  "
          f"diff={abs(np.mean(vn_out_fast)-np.mean(vn_out_slow)):.4f} "
          f"({'INDEPENDENT' if abs(np.mean(vn_out_fast)-np.mean(vn_out_slow)) < 0.05 else 'DEPENDENT — rupe DB'})")

print("\n=== 3. <v2> output vs teoretic ===")
print(f"  Teoretic Maxwell 3D: <v2> = 3*kT/m = {3*kT/m:.3f}")
print(f"  Teoretic v_n Rayleigh: <v_n> = sqrt(pi/2)*s = {np.sqrt(np.pi/2)*s:.4f}")
