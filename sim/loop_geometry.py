# sim/loop_geometry.py  (schelet)
import numpy as np
from hoomd import mpcd

# Dimensiuni de exemplu (schimbă liber):
CUBE = dict(size=(60, 60, 60))   # IN, MID, OUT
PIPE_TOP = dict(radii=[6, 10, 8], lengths=[20, 10, 20])  # funnel1, mid constriction, funnel2
PIPE_BOTTOM = dict(radius=20, length=120)

# Materiale (exemplu): sticlă= no-slip, PTFE= slip parțial (modelat ca specular)
MATS = {
    'cube_in':  {'no_slip': True},
    'pipe_f1':  {'no_slip': True},
    'pipe_mid': {'no_slip': False},   # “alunecare” (PTFE)
    'pipe_f2':  {'no_slip': True},
    'cube_mid': {'no_slip': True},
    'cube_out': {'no_slip': True},
    'pipe_bot': {'no_slip': True},
    'risers':   {'no_slip': True},
}

def make_geometry():
    """Construiește o listă de primitive pentru BounceBack.
       Pentru MPCD folosim un compus de plăci și cilindri care conturează camerele + tuburile.
       (HOOMD tratează coliziile la suprafețe în timpul streamingului MPCD.)
    """
    geoms = []

    # Exemplu: IN cube = “cutie închisă” cu o gură circulară (riser) – se descrie cu plane + cilindru
    # (Pentru claritate păstrăm aici doar câteva primitive; în repo poți adăuga toate muchiile.)
    # HOOMD are utilitare mpcd.geometry.* (Plane, Cylinder, Sphere). Le instanțiezi cu normală și poziție.

    # … construiește aici pereții pentru cele 3 cuburi și conectările cu cilindri (sus) și tubul gros (jos).
    # Setează atributul no_slip din MATS pentru fiecare componentă când creezi obiectul geometric.

    # PSEUDOCOD (în repo vei avea implementarea completă):
    # geoms.append(mpcd.geometry.Plane(origin=(x0, y0, z0), normal=(1,0,0), no_slip=MATS['cube_in']['no_slip']))
    # geoms.append(mpcd.geometry.Cylinder(center=..., axis=..., radius=..., length=..., no_slip=MATS['pipe_mid']['no_slip']))
    # etc.

    return mpcd.geometry.List(geoms)

def sample_in_fluid(N, box, geom):
    """Eșantionează poziții în volum excluzând solidele (rejection sampling).
       geom.point_inside(x) -> True dacă x e în fluid.
    """
    Lx, Ly, Lz = box.Lx, box.Ly, box.Lz
    out = np.empty((N,3))
    i = 0
    while i < N:
        p = (np.random.rand(3) - 0.5) * np.array([Lx, Ly, Lz])
        if geom.is_inside(p):   # funcție utilitară a compoziției de geometrii
            out[i] = p
            i += 1
    return out
