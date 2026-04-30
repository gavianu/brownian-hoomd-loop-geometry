"""
- Estimates local drift <Δr/Δt> on a sliding window and bins it per voxel
Outputs:
- out/msd.csv (time, MSD)
- out/density_voxels.npz (density, x_edges, y_edges, z_edges)
- out/drift_voxels.npz (drift[... ,3], counts, x_edges, y_edges, z_edges)
"""
import os
import numpy as np
import pandas as pd
from gsd import hoomd


OUT = os.path.join(os.path.dirname(__file__), 'out')
TRJ = os.path.join(OUT, 'traj.gsd')


# --------- Load trajectory ---------
with hoomd.open(TRJ, 'rb') as t:
frames = [f for f in t]


if not frames:
raise RuntimeError('No frames found in traj.gsd. Did the sim run?')


# Assume all particles are tracers (type 0). If MPCD particles are present in file,
# you can filter by type name here.
# Extract box for histogram bounds
box = frames[0].configuration.box # [Lx, Ly, Lz, xy, xz, yz]
Lx, Ly, Lz = box[:3]


# --------- MSD ---------
# Positions over time (T, N, 3)
pos_list = [f.particles.position.copy() for f in frames]
P0 = pos_list[0]
pos_arr = np.stack(pos_list, axis=0)
# unwrap not needed for nonperiodic; if periodic, do simple unwrap here


# Displacements from t0
D = pos_arr - P0[None, :, :]
MSD = np.mean(np.sum(D**2, axis=2), axis=1)
# Time axis: here we infer dt from frame interval stored in log if available; else use step index
# GSD doesn’t store time by default, so we write step index as proxy
steps = np.arange(len(frames), dtype=float)
msd_df = pd.DataFrame({'step': steps, 'MSD': MSD})
msd_df.to_csv(os.path.join(OUT, 'msd.csv'), index=False)


# --------- Density histogram (last frame) ---------
# You can also average over frames if desired
Nbx, Nby, Nbz = 40, 20, 20
edges_x = np.linspace(-Lx/2, Lx/2, Nbx+1)
edges_y = np.linspace(-Ly/2, Ly/2, Nby+1)
edges_z = np.linspace(-Lz/2, Lz/2, Nbz+1)


H, _ = np.histogramdd(frames[-1].particles.position,
bins=(edges_x, edges_y, edges_z))
np.savez_compressed(os.path.join(OUT, 'density_voxels.npz'),
density=H, x_edges=edges_x, y_edges=edges_y, z_edges=edges_z)


# --------- Drift estimation ---------
# Compute per-particle instantaneous velocity over a window and bin it
window = 5 # frames
vel_bins = np.zeros((Nbx, Nby, Nbz, 3), dtype=float)
counts = np.zeros((Nbx, Nby, Nbz), dtype=int)


for k in range(window, len(frames)):
r_now = frames[k].particles.position
r_prev = frames[k - window].particles.position
v_est = (r_now - r_prev) / max(1, window) # dt=1 frame; scale if you know real dt


ix = np.clip(np.digitize(r_now[:, 0], edges_x) - 1, 0, Nbx - 1)
iy = np.clip(np.digitize(r_now[:, 1], edges_y) - 1, 0, Nby - 1)
iz = np.clip(np.digitize(r_now[:, 2], edges_z) - 1, 0, Nbz - 1)


for p in range(r_now.shape[0]):
vel_bins[ix[p], iy[p], iz[p]] += v_est[p]
counts[ix[p], iy[p], iz[p]] += 1


with np.errstate(invalid='ignore'):
drift = np.where(counts[..., None] > 0, vel_bins / counts[..., None], 0.0)


np.savez_compressed(os.path.join(OUT, 'drift_voxels.npz'),
drift=drift, counts=counts,
x_edges=edges_x, y_edges=edges_y, z_edges=edges_z)


print('Wrote: out/msd.csv, out/density_voxels.npz, out/drift_voxels.npz')