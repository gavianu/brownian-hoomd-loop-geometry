# sim/run_mpcd.py
import hoomd
from hoomd import mpcd, md
import numpy as np
from loop_geometry import make_geometry  # tu setezi dimensiuni/materiale aici

# ---- device ----
try:
    device = hoomd.device.GPU()  # pe Windows cu NVIDIA
except RuntimeError:
    device = hoomd.device.CPU()  # fallback (macOS sau fără GPU)

sim = hoomd.Simulation(device=device, seed=42)

# ---- BOX ----
# alege o cutie care include tot ansamblul; non-periodic pe toate axele
Lx, Ly, Lz = 200, 120, 120   # unități arbitr.
box = hoomd.Box(Lx=Lx, Ly=Ly, Lz=Lz)
snap = hoomd.Snapshot()
snap.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
snap.particles.N = 0  # tracerii îi adăugăm după
sim.create_state_from_snapshot(snap)

# ---- GEOMETRIE MPCD + PEREȚI ----
# make_geometry returnează o listă de primitive (plane/cylinder) pentru camere/funnels/tub
# și tag-uri de "material" -> no_slip / slip.
geom = make_geometry()  # vezi loop_geometry.py

# Streaming MPCD cu bounce-back la pereți (no-slip ideal pe segmentele marcate no_slip)
stream = mpcd.stream.BounceBack(period=1, geometry=geom)

# Temperatura efectivă (unități reduse) și densitatea de particule MPCD / celulă
methods = mpcd.methods.BounceBack(streaming_method=stream, kT=1.0)
sim.operations.integrator = methods

# ---- Sistemul MPCD (fluid) ----
# “SRD” = Stochastic Rotation Dynamics; dt și unghiul controlează vâscozitatea efectivă
srd = mpcd.integrate.SRD(simulation=sim,
                         seed=1,
                         dt=0.1,
                         rotation_angle=mpcd.integrate.srd_rotation_angle(130.0),
                         filter=hoomd.filter.All())
# Densitate țintă (particule/celulă), dimensiune celulă
mpcd.init.make_random(simulation=sim, number_density=10.0, box=box, cell=(2.0, 2.0, 2.0))

# ---- Traceri (particule MD pasive) ----
N_tr = 2000
snap = sim.state.get_snapshot()
if snap.communicator.rank == 0:
    # plasăm tracerii uniform în fluid (excludem solidele) – funcție helper:
    from loop_geometry import sample_in_fluid
    pos = sample_in_fluid(N_tr, box, geom)
    snap.particles.N = N_tr
    snap.particles.position[:] = pos
    snap.particles.types = ['A']
sim.state.set_snapshot(snap)

# Brownian “pasiv”: masa mică, doar pentru output/urmărire (mișcarea vine din cuplajul cu MPCD)
bd = md.methods.Brownian(kT=1.0, filter=hoomd.filter.All())
integrator = md.Integrator(dt=0.01, methods=[bd], forces=[])
sim.operations.integrator = integrator
# Cuplaj MPCD↔MD (tracerii simt fluidul; fluidul simte tracerii neglijabil)
mpcd.coupling.AT(simulation=sim, seed=2)

# ---- Output pentru OVITO ----
gsd = hoomd.write.GSD(filename='out/traj.gsd',
                      trigger=hoomd.trigger.Periodic(1000),
                      mode='wb',
                      filter=hoomd.filter.All())
sim.operations.writers.append(gsd)

# Probe pentru MSD și poziții (CSV)
logger = hoomd.logging.Logger(categories=['scalar'])
msd = md.compute.MSD(hoomd.filter.All(), variant=hoomd.variant.Constant(0))
sim.operations.computes.append(msd)

table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),
                          logger=logger,
                          output=open('out/msd.csv', 'w'))
logger.add(msd, quantities=['value'])
sim.operations.writers.append(table)

# ---- RUN ----
sim.run(500_000)
