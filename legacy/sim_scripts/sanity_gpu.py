import hoomd
from hoomd import md
import numpy as np

# --- Device (GPU if available, else CPU) ---
try:
    device = hoomd.device.GPU()
    _ = device.devices  # triggers query
except Exception:
    device = hoomd.device.CPU()
print("Using:", device)

# --- Simulation & box ---
sim = hoomd.Simulation(device=device, seed=1)
Lx, Ly, Lz = 100, 100, 100
box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)
snap = hoomd.Snapshot()
snap.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

# --- Particles (traceri doar pentru test) ---
N = 500
snap.particles.N = N
rng = np.random.default_rng(0)
# positions in (-L/2, L/2)
snap.particles.position[:] = (rng.random((N,3)) - 0.5) * np.array([Lx, Ly, Lz])
snap.particles.types = ['A']

sim.create_state_from_snapshot(snap)

# --- Brownian (doar ca sanity check; MPCD vine după) ---
bd = md.methods.Brownian(kT=1.0, filter=hoomd.filter.All())
integrator = md.Integrator(dt=0.01, methods=[bd])
sim.operations.integrator = integrator

# --- Output GSD pentru OVITO ---
import os
os.makedirs('sim/out', exist_ok=True)
gsd = hoomd.write.GSD(filename='sim/out/traj.gsd',
                      trigger=hoomd.trigger.Periodic(200),
                      mode='wb',
                      filter=hoomd.filter.All())
sim.operations.writers.append(gsd)

print("Running sanity check... (this writes sim/out/traj.gsd)")
sim.run(10_000)
print("Done.")
