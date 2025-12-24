"""
Brownian tracers confined by a complex geometry built from FIXED "wall beads" (particle mesh).
- No solvent/MPCD (Route 1) → simple, robust across HOOMD versions.
- Works in the Docker image you already ran (glotzerlab/software:latest).
- Exports traj for OVITO and CSV with MSD + simple per-particle zone transits.

Geometry (coarse but effective):
  - Three chambers (cubes): LEFT, MID, RIGHT, arranged along X (top row).
  - Two top connectors ("funnels"): straight ducts with optional narrow middle.
  - One bottom tube connecting LEFT ↔ RIGHT below the chambers.
  - Different "materials": emulate no-slip vs slip by changing wall bead roughness
    and interaction strength (epsilon) → stronger + rougher ≈ "stickier" (no-slip-like),
    weaker + sparser ≈ more slip.

Tunable parameters are grouped in CFG below.
"""

import os
import math
import numpy as np
import hoomd
from hoomd import md

# ---------------- CFG (edit freely) ----------------
class CFG:
    # Box (must contain everything)
    Lx, Ly, Lz = 220.0, 140.0, 120.0

    # Chambers (cubes) size
    CHAM = (60.0, 60.0, 60.0)  # (sx, sy, sz)
    C_LEFT  = (-70.0,  30.0, 0.0)
    C_MID   = (  0.0,  30.0, 0.0)
    C_RIGHT = ( 70.0,  30.0, 0.0)

    # Top ducts (rectangular) and bottom tube
    DUCT_Y, DUCT_Z = 60.0, 0.0
    DUCT_SY, DUCT_SZ = 18.0, 18.0
    NARROW_SY, NARROW_LEN = 10.0, 10.0
    BOT_Y, BOT_SY, BOT_SZ = -40.0, 30.0, 30.0

    # Wall bead lattice (coarser = faster, fewer overlaps)
    WALL_SPACING = 3.5
    WALL_THICK   = 3.5

    # Tracers
    N_TR = 1500
    TR_RADIUS = 1.0

    # Dynamics
    DT_RELAX = 1e-4     # tiny dt for initial relaxation
    STEPS_RELAX = 20000 # relax with kT=0 and soft walls
    DT_RUN   = 0.001    # production dt (safe)
    STEPS    = 200000
    WRITE_EVERY = 20000

    # LJ parameters (will be ramped: EPS_SOFT -> EPS_HARD)
    SIGMA = 1.5
    EPS_SOFT = 0.5
    EPS_HARD = 1.5

    # Analysis
    TRACK_IDS = list(range(20))  # first N tracers tracked

    # Materials via WCA-like parameters (epsilon scales "stickiness")
    MAT_LEFT   = dict(epsilon=2.0)
    MAT_MID    = dict(epsilon=1.5)
    MAT_RIGHT  = dict(epsilon=2.0)
    MAT_DUCT   = dict(epsilon=2.5)   # funnels
    MAT_BOT    = dict(epsilon=2.0)   # bottom tube


    # Analysis
    TRACK_IDS = list(range(20))  # first N tracers tracked

# --------------------------------------------------

# Utility: build wall beads for a rectangular box shell with an opening mask

def rect_shell(center, size, spacing, thick, open_masks=()):
    """Generate wall bead positions on the surfaces of an axis-aligned box.
    open_masks: list of callables f(x,y,z)->bool; points where any mask==True are SKIPPED (openings).
    """
    cx, cy, cz = center
    sx, sy, sz = size
    # extents
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    pts = []
    # ranges for grids
    xr = np.arange(x0, x1+1e-6, spacing)
    yr = np.arange(y0, y1+1e-6, spacing)
    zr = np.arange(z0, z1+1e-6, spacing)

    def keep(p):
        return not any(mask(*p) for mask in open_masks)

    # six faces; draw two layers (thickness)
    offs = np.linspace(0, thick, 2)
    # x-faces
    for sgn, xx in [(-1, x0), (1, x1)]:
        for o in offs:
            # move INWARD, not outward
            x = xx - sgn*o
            Y, Z = np.meshgrid(yr, zr, indexing='ij')
            face = np.column_stack([np.full(Y.size, x), Y.ravel(), Z.ravel()])
            for p in face:
                if keep(p):
                    pts.append(p)
    # y-faces
    for sgn, yy in [(-1, y0), (1, y1)]:
        for o in offs:
            y = yy - sgn*o
            X, Z = np.meshgrid(xr, zr, indexing='ij')
            face = np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()])
            for p in face:
                if keep(p):
                    pts.append(p)
    # z-faces
    for sgn, zz in [( -1, z0), (1, z1)]:
        for o in offs:
            z = zz - sgn*o
            X, Y = np.meshgrid(xr, yr, indexing='ij')
            face = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
            for p in face:
                if keep(p):
                    pts.append(p)
    return np.asarray(pts, dtype=np.float32)

# Utility: rectangular duct along X between two x's

def duct_shell(x0, x1, y_center, z_center, sy, sz, spacing, thick):
    cx = 0.5*(x0+x1)
    size = (abs(x1-x0), sy, sz)
    return rect_shell((cx, y_center, z_center), size, spacing, thick, open_masks=())

# Masks to carve openings in chamber faces (top connectors)

def hole_top_mask(x0, x1, y_top, zc, sy, sz):
    def _mask(x,y,z):
        return (abs(y - y_top) < 1.0) and (x0 <= x <= x1) and (abs(z - zc) <= sz/2)
    return _mask

# Build full geometry

def build_geometry_points(cfg: CFG):
    W = cfg
    pts = []

    # LEFT chamber with top opening to duct
    half = np.array(cfg.CHAM)/2
    left_open = hole_top_mask(W.C_LEFT[0]-half[0], W.C_LEFT[0]+half[0], W.C_LEFT[1]+half[1], W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ)
    pts.append(rect_shell(W.C_LEFT, cfg.CHAM, W.WALL_SPACING, W.WALL_THICK, open_masks=[left_open]))

    # MID chamber with two top openings
    mid_open = hole_top_mask(W.C_MID[0]-half[0], W.C_MID[0]+half[0], W.C_MID[1]+half[1], W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ)
    pts.append(rect_shell(W.C_MID, cfg.CHAM, W.WALL_SPACING, W.WALL_THICK, open_masks=[mid_open]))

    # RIGHT chamber with top opening
    right_open = hole_top_mask(W.C_RIGHT[0]-half[0], W.C_RIGHT[0]+half[0], W.C_RIGHT[1]+half[1], W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ)
    pts.append(rect_shell(W.C_RIGHT, cfg.CHAM, W.WALL_SPACING, W.WALL_THICK, open_masks=[right_open]))

    # TOP ducts LEFT↔MID and MID↔RIGHT
    xLM0 = W.C_LEFT[0] + cfg.CHAM[0]/2
    xLM1 = W.C_MID[0]  - cfg.CHAM[0]/2
    xMR0 = W.C_MID[0]  + cfg.CHAM[0]/2
    xMR1 = W.C_RIGHT[0]- cfg.CHAM[0]/2

    def split_with_narrow(x0, x1):
        xm = 0.5*(x0+x1)
        return (x0, xm - W.NARROW_LEN/2, xm + W.NARROW_LEN/2, x1)

    # LEFT-MID duct
    a,b,c,d = split_with_narrow(xLM0, xLM1)
    pts.append(duct_shell(a, b, W.DUCT_Y, W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))
    pts.append(duct_shell(b, c, W.DUCT_Y, W.DUCT_Z, W.NARROW_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))
    pts.append(duct_shell(c, d, W.DUCT_Y, W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))

    # MID-RIGHT duct
    a,b,c,d = split_with_narrow(xMR0, xMR1)
    pts.append(duct_shell(a, b, W.DUCT_Y, W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))
    pts.append(duct_shell(b, c, W.DUCT_Y, W.DUCT_Z, W.NARROW_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))
    pts.append(duct_shell(c, d, W.DUCT_Y, W.DUCT_Z, W.DUCT_SY, W.DUCT_SZ, W.WALL_SPACING, W.WALL_THICK))

    # BOTTOM tube LEFT↔RIGHT (straight)
    xb0 = W.C_LEFT[0] + cfg.CHAM[0]/2
    xb1 = W.C_RIGHT[0] - cfg.CHAM[0]/2
    pts.append(duct_shell(xb0, xb1, W.BOT_Y, W.DUCT_Z, W.BOT_SY, W.BOT_SZ, W.WALL_SPACING, W.WALL_THICK))

    # OUTER container walls to confine everything inside the simulation box
    pts.append(rect_shell((0.0, 0.0, 0.0), (W.Lx-2.0, W.Ly-2.0, W.Lz-2.0), W.WALL_SPACING, W.WALL_THICK, open_masks=()))

    P = np.vstack(pts).astype(np.float32)
    return P

# Sample tracer positions inside fluid (rejection against wall beads)

def sample_tracers(N, box, wall_pos, min_dist=2.0, seed=0):
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz = box
    pos = np.empty((N,3), dtype=np.float32)
    i = 0
    wall_tree = None
    try:
        # optional speed-up if scipy available in container
        from scipy.spatial import cKDTree
        wall_tree = cKDTree(wall_pos)
    except Exception:
        wall_tree = None

    while i < N:
        p = (rng.random(3)-0.5)*np.array([Lx, Ly, Lz])
        if wall_tree is not None:
            d = wall_tree.query(p, k=1)[0]
            if d < min_dist:
                continue
        else:
            # brute force small subset
            if np.min(np.linalg.norm(wall_pos[::20] - p, axis=1)) < min_dist:
                continue
        pos[i] = p
        i += 1
    return pos

# ---------------- Build system ----------------

def main():
    os.makedirs('sim/out1', exist_ok=True)

    # try:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #     device = hoomd.device.GPU()
    #     print(f"✓ Using GPU: {device.device}")
    # except Exception as e:
    #     print(f"⚠ GPU unavailable ({e}), falling back to CPU")
    device = hoomd.device.CPU()
    
    sim = hoomd.Simulation(device=device, seed=123)

    # Box & empty snapshot
    snap = hoomd.Snapshot()
    snap.configuration.box = [CFG.Lx, CFG.Ly, CFG.Lz, 0, 0, 0]

    # Build wall beads
    wall_pos = build_geometry_points(CFG)
    print(f"Generated {len(wall_pos)} wall beads")

    n_wall = wall_pos.shape[0]

    # Particles: wall beads (type 'W') + tracers (type 'A')
    Ntr = CFG.N_TR
    snap.particles.N = n_wall + Ntr
    snap.particles.types = ['A', 'W']

    # Assign types and positions
    snap.particles.typeid[:] = 0
    snap.particles.position[:] = 0

    # Walls first (frozen later)
    snap.particles.typeid[:n_wall] = 1
    snap.particles.position[:n_wall] = wall_pos

    # Traceri
    tr_pos = sample_tracers(Ntr, (CFG.Lx, CFG.Ly, CFG.Lz), wall_pos, min_dist=2.8, seed=7)
    snap.particles.position[n_wall:] = tr_pos

    sim.create_state_from_snapshot(snap)

    # Freeze walls (set massive + zero DOF)
    rigid_filter = hoomd.filter.Tags([i for i in range(n_wall)])
    mobile_filter = hoomd.filter.Tags([i for i in range(n_wall, n_wall+Ntr)])

    # Brownian/Langevin for traceri
    mobile_filter = hoomd.filter.Tags([i for i in range(n_wall, n_wall+Ntr)])
    brown = md.methods.Brownian(kT=0.0, filter=mobile_filter)  # start with kT=0 for relaxation
    integrator = md.Integrator(dt=CFG.DT_RELAX, methods=[brown])

    # Neighbor list + LJ pair
    nl = md.nlist.Cell(buffer=0.4)
    lj = md.pair.LJ(nlist=nl)
    sigma = CFG.SIGMA
    rcut = 2**(1/6)*sigma
    # Define ALL pairs
    lj.params[('A','A')] = dict(epsilon=0.05, sigma=sigma)
    lj.r_cut[('A','A')] = rcut
    lj.params[('A','W')] = dict(epsilon=CFG.EPS_SOFT, sigma=sigma)
    lj.r_cut[('A','W')] = rcut
    lj.params[('W','A')] = dict(epsilon=CFG.EPS_SOFT, sigma=sigma)
    lj.r_cut[('W','A')] = rcut
    lj.params[('W','W')] = dict(epsilon=0.0, sigma=1.0)
    lj.r_cut[('W','W')] = 0.0

    integrator.forces.append(lj)
    sim.operations.integrator = integrator

    # Constrain wall beads: zero-temperature Langevin (pinning)
    rigid_filter = hoomd.filter.Tags([i for i in range(n_wall)])
    thermostat_walls = md.methods.Langevin(kT=0.0, filter=rigid_filter)
    integrator.methods.append(thermostat_walls)

    # Writers
    gsd = hoomd.write.GSD(filename='sim/out1/traj.gsd',
                          mode='wb',
                          trigger=hoomd.trigger.Periodic(CFG.WRITE_EVERY),
                          filter=hoomd.filter.All())
    sim.operations.writers.append(gsd)

    print(f"Walls: {n_wall} beads | Tracers: {Ntr} | Box=({CFG.Lx},{CFG.Ly},{CFG.Lz})")
    print("Stage 1: relax (kT=0, soft walls)…")
    sim.run(CFG.STEPS_RELAX)

    # Ramp to production: harder walls and Brownian thermal noise, larger dt
    print("Stage 2: production (kT=1, harder walls)…")
    brown.kT = 1.0
    integrator.dt = CFG.DT_RUN
    lj.params[('A','W')] = dict(epsilon=CFG.EPS_HARD, sigma=sigma)
    lj.params[('W','A')] = dict(epsilon=CFG.EPS_HARD, sigma=sigma)

    # Writer
    gsd = hoomd.write.GSD(filename='sim/out1/traj.gsd',
                          mode='wb',
                          trigger=hoomd.trigger.Periodic(CFG.WRITE_EVERY),
                          filter=hoomd.filter.All())
    sim.operations.writers.append(gsd)

    print("Running… writing sim/out1/traj.gsd")
    sim.run(CFG.STEPS)
    print("Done.")

if __name__ == '__main__':
    main()
