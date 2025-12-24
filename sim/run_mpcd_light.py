import hoomd, numpy as np
from hoomd import mpcd, md
try: dev=hoomd.device.GPU()
except: dev=hoomd.device.CPU()
sim=hoomd.Simulation(device=dev, seed=11)
# box + snapshot
L=1.0; snap=hoomd.Snapshot(); snap.configuration.box=[L,L,L,0,0,0]
# traceri MD (doar ca să-i vedem miscați de MPCD)
Ntr=100
rng=np.random.default_rng(0)
snap.particles.N=Ntr
snap.particles.position[:]=(rng.random((Ntr,3))-0.5)*L
snap.particles.types=['A']
# solvent MPCD prin Snapshot.mpcd
Ns= int((L**3)/(0.2**3))  # ~densitate 10/celulă (celula e fixă 1.0 în v5)
snap.mpcd.N = Ns
snap.mpcd.types = ['S']
snap.mpcd.position[:]=(rng.random((Ns,3))-0.5)*L
snap.mpcd.velocity[:]=0.0
sim.create_state_from_snapshot(snap)
# integrator MPCD (v5): stream bulk + SRD collisions, cuplaj la toți tracerii
md_dt=0.01; period=10
stream = mpcd.stream.Bulk(period=period)
collide = mpcd.collide.StochasticRotationDynamics(period=period, angle=130, kT=1.0, embedded_particles=hoomd.filter.All())
solute_method = md.methods.Brownian(kT=1.0, filter=collide.embedded_particles)
integr = mpcd.Integrator(dt=md_dt, methods=[solute_method], streaming_method=stream, collision_method=collide, mpcd_particle_sorter=mpcd.tune.ParticleSorter(trigger=200))
sim.operations.integrator = integr
# output GSD
gsd=hoomd.write.GSD(filename='sim/out/traj.gsd', trigger=hoomd.trigger.Periodic(500), mode='wb', filter=hoomd.filter.All())
sim.operations.writers.append(gsd)
print('running...'); sim.run(5000); print('done, wrote sim/out/traj.gsd');