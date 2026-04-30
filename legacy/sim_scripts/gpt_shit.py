# sim/geometry_beads_gpu.py
# HOOMD 4.x — geometrie conectată cu „wall beads” (fără dinamică încă).
# Scrie:
#   sim/out/geom_beads.gsd  (beads tip W, pereți perforați unde trebuie)
#   sim/out/geom.obj/.mtl   (vizual, transparent, pentru OVITO)
#
# Ideea: generăm puncte (beads) pe suprafața cuburilor/țevilor și
# SĂRIM punctele din zonele „găurilor” (trecerile). Așa rezultă un volum conectat.

import os, math
import numpy as np
import hoomd
from hoomd import write, trigger

os.makedirs("sim/out", exist_ok=True)

class CFG:
    # box (doar pentru GSD)
    Lx, Ly, Lz = 500.0, 320.0, 260.0

    # cuburi
    CUBE  = (80.0, 80.0, 80.0)
    CUB1  = (-120.0,  70.0, 0.0)
    CUB2  = ( 120.0,  70.0, 0.0)

    # conector sus (5 segmente pe X)
    FUNNEL_Y, FUNNEL_Z =  90.0, 0.0
    FUNNEL_SEG = 5
    FUNNEL_RAD = [12.0, 10.0, 8.0, 10.0, 12.0]
    FUNNEL_PAD =  2.0    # cât intră peste fețele cuburilor în OBJ

    # verticale + retur (retur mai jos, mai gros)
    LOOP_Y   = -90.0
    VERT_R   = 12.0
    RET_R    = 26.0
    RET_EXTRA= 30.0      # retur mai lung la capete
    RET_PORT_R = 12.0    # raza „găurii” în retur (≈ VERT_R)

    # sampling pereți
    WALL_SP  = 3.5       # pasul rețelei pentru beads (mai mic => mai dens)
    WALL_T   = 3.5       # grosime (număr de straturi = 2, spațiate cu WALL_T/1)
    # (2 straturi e OK pentru etanș)

    # OBJ
    SIDES = 96
    ALPHA = 0.55

# ---------------- utilitare geometrie (sampling beads) ----------------

def grid_1d(a, b, h):
    n = max(1, int(round((b - a) / h)))
    return np.linspace(a, b, n+1)

def sample_box_shell(center, size, spacing, thick,
                     hole_disks=()):
    """
    Beads pe „coaja” unei cutii axate pe axe.
    hole_disks: listă de „găuri circulare” descrise ca
      dict(face='x+'|'x-'|'y+'|'y-'|'z+'|'z-', cy=?, cz=?, r=?)
      — beads din acea zonă sunt sărite (deschidere reală).
    """
    cx, cy, cz = center; sx, sy, sz = size
    x0,x1 = cx - sx/2, cx + sx/2
    y0,y1 = cy - sy/2, cy + sy/2
    z0,z1 = cz - sz/2, cz + sz/2

    # mapăm găurile pe fețe
    holes = {'x+':[], 'x-':[], 'y+':[], 'y-':[], 'z+':[], 'z-':[]}
    for h in hole_disks:
        holes[h['face']].append(h)

    pts = []
    # două straturi spre interior
    offs = np.linspace(0, thick, 2)

    # selecție „dacă p pe față e în gaură”
    def in_disk(face, y, z):
        for h in holes[face]:
            if (y - h['cy'])**2 + (z - h['cz'])**2 <= h['r']**2:
                return True
        return False

    # X-faces
    yr = grid_1d(y0, y1, spacing)
    zr = grid_1d(z0, z1, spacing)
    for sgn, xx, face in [(-1, x0, 'x-'), (1, x1, 'x+')]:
        for o in offs:
            x = xx - sgn*o    # intrăm spre interior
            Y, Z = np.meshgrid(yr, zr, indexing='ij')
            face_pts = np.column_stack([np.full(Y.size, x), Y.ravel(), Z.ravel()])
            for p in face_pts:
                if not in_disk(face, p[1], p[2]):
                    pts.append(p)

    # Y-faces
    xr = grid_1d(x0, x1, spacing)
    zr = grid_1d(z0, z1, spacing)
    for sgn, yy, face in [(-1, y0, 'y-'), (1, y1, 'y+')]:
        for o in offs:
            y = yy - sgn*o
            X, Z = np.meshgrid(xr, zr, indexing='ij')
            pts.extend(np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()]))

    # Z-faces
    xr = grid_1d(x0, x1, spacing)
    yr = grid_1d(y0, y1, spacing)
    for sgn, zz, face in [(-1, z0, 'z-'), (1, z1, 'z+')]:
        for o in offs:
            z = zz - sgn*o
            X, Y = np.meshgrid(xr, yr, indexing='ij')
            pts.extend(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)]))

    return np.asarray(pts, dtype=np.float32)

def sample_cyl_shell_x(cx, cy, cz, R, L, spacing, thick, skip_top_windows=()):
    """
    Beads pe manta unui cilindru pe X (cu capace opționale – aici lăsăm capetele deschise).
    skip_top_windows: listă cu dict(xc=?, half_dx=?, r=?)
       – „ferestre” pe coroana de sus (în jurul unghiului pi/2) pe unde NU punem beads,
         ca să reprezinte GAURA (ex: conexiunea verticală în retur).
    """
    # manta: param (x,theta)
    x0, x1 = cx - L/2, cx + L/2
    xs = grid_1d(x0, x1, spacing)
    angs = grid_1d(0.0, 2*math.pi, (spacing / max(R,1e-6)))
    offs = np.linspace(0, thick, 2)  # două straturi

    def in_window(x, th):
        # sus ~ pi/2, +/-; permitem gaura dacă x e în [xc-half_dx, xc+half_dx]
        for w in skip_top_windows:
            if (w['xc'] - w['half_dx'] <= x <= w['xc'] + w['half_dx']):
                d = ((th - math.pi/2 + math.pi) % (2*math.pi)) - math.pi
                if abs(d) <= math.asin(min(0.999, w['r']/R)):
                    return True
        return False

    pts = []
    for o in offs:
        r = R - o
        for x in xs:
            for th in angs:
                if skip_top_windows and in_window(x, th):
                    continue
                y = cy + r * math.cos(th)
                z = cz + r * math.sin(th)
                pts.append((x,y,z))
    return np.asarray(pts, dtype=np.float32)

def sample_cyl_shell_y(cx, cy, cz, R, L, spacing, thick):
    # manta cilindru pe Y
    y0, y1 = cy - L/2, cy + L/2
    ys = grid_1d(y0, y1, spacing)
    angs = grid_1d(0.0, 2*math.pi, (spacing / max(R,1e-6)))
    offs = np.linspace(0, thick, 2)
    pts=[]
    for o in offs:
        r = R - o
        for y in ys:
            for th in angs:
                x = cx + r * math.cos(th)
                z = cz + r * math.sin(th)
                pts.append((x,y,z))
    return np.asarray(pts, dtype=np.float32)

# ---------------- OBJ (vizual) — indicii 1-based corecți ----------------

def write_obj_mtl(obj_path, mtl_path, items):
    with open(mtl_path, "w") as m:
        for nm, col in [("mat_cube",(0.7,0.7,0.95)),
                        ("mat_fun",(0.7,0.95,0.7)),
                        ("mat_vert",(0.95,0.95,0.7)),
                        ("mat_ret",(0.95,0.7,0.7))]:
            m.write(f"newmtl {nm}\nKd {col[0]} {col[1]} {col[2]}\nd {CFG.ALPHA}\nillum 2\n\n")
    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        vbase=0
        for it in items:
            f.write(f"o {it['name']}\nusemtl {it['mat']}\n")
            for x,y,z in it["V"]:
                f.write(f"v {x} {y} {z}\n")
            for a,b,c in it["F"]:
                f.write(f"f {a+1+vbase} {b+1+vbase} {c+1+vbase}\n")
            vbase += len(it["V"])

def solid_cyl_x_mesh(cx,cy,cz,R,L,sides):
    V,F=[],[]
    x0,x1 = cx-L/2, cx+L/2
    def ring(Rr,x):
        return [(x, cy+Rr*math.cos(2*math.pi*i/sides), cz+Rr*math.sin(2*math.pi*i/sides)) for i in range(sides)]
    r0, r1 = ring(R,x0), ring(R,x1)
    base=len(V); V.extend(r0+r1)
    def t(a,b,c): F.append((a,b,c))
    for i in range(sides):
        a=base+i; b=base+((i+1)%sides); c=base+sides+((i+1)%sides); d=base+sides+i
        t(a,b,c); t(a,c,d)
    # capace
    c0=len(V); V.append((x0,cy,cz))
    for i in range(sides): a=base+i; b=base+((i+1)%sides); t(c0,b,a)
    c1=len(V); V.append((x1,cy,cz))
    for i in range(sides): a=base+sides+i; b=base+sides+((i+1)%sides); t(c1,a,b)
    return V,F

def solid_cyl_y_mesh(cx,cy,cz,R,L,sides):
    V,F=[],[]
    y0,y1 = cy-L/2, cy+L/2
    def ring(Rr,y):
        return [(cx+Rr*math.cos(2*math.pi*i/sides), y, cz+Rr*math.sin(2*math.pi*i/sides)) for i in range(sides)]
    r0, r1 = ring(R,y0), ring(R,y1)
    base=len(V); V.extend(r0+r1)
    def t(a,b,c): F.append((a,b,c))
    for i in range(sides):
        a=base+i; b=base+((i+1)%sides); c=base+sides+((i+1)%sides); d=base+sides+i
        t(a,c,b); t(a,d,c)
    # capace
    c0=len(V); V.append((cx,y0,cz))
    for i in range(sides): a=base+i; b=base+((i+1)%sides); t(c0,a,b)
    c1=len(V); V.append((cx,y1,cz))
    for i in range(sides): a=base+sides+i; b=base+sides+((i+1)%sides); t(c1,b,a)
    return V,F

def cube_mesh(cx,cy,cz,sx,sy,sz):
    V,F=[],[]
    x0,x1 = cx-sx/2, cx+sx/2
    y0,y1 = cy-sy/2, cy+sy/2
    z0,z1 = cz-sz/2, cz+sz/2
    V.extend([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
              (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
    def t(a,b,c): F.append((a,b,c))
    t(0,1,2);t(0,2,3);t(4,6,5);t(4,7,6)
    t(0,4,5);t(0,5,1);t(1,5,6);t(1,6,2)
    t(2,6,7);t(2,7,3);t(3,7,4);t(3,4,0)
    return V,F

# ---------------- build (beads + obj) ----------------

def build_all():
    W = CFG
    # fețele interioare ale cuburilor (x ±) unde vin funnels & verticale
    x1_face = W.CUB1[0] + W.CUBE[0]/2
    x2_face = W.CUB2[0] - W.CUBE[0]/2

    # === BEADS ===
    beads = []

    # cub stânga/dreapta cu găuri în fețele X
    holesL = [ dict(face='x+',
                    cy=W.FUNNEL_Y, cz=W.FUNNEL_Z, r=max(W.FUNNEL_RAD)),
               dict(face='x+',
                    cy=W.CUB1[1]-W.CUBE[1]/2, cz=0.0, r=W.VERT_R) ]
    holesR = [ dict(face='x-',
                    cy=W.FUNNEL_Y, cz=W.FUNNEL_Z, r=max(W.FUNNEL_RAD)),
               dict(face='x-',
                    cy=W.CUB2[1]-W.CUBE[1]/2, cz=0.0, r=W.VERT_R) ]

    beads.append( sample_box_shell(W.CUB1, W.CUBE, W.WALL_SP, W.WALL_T, hole_disks=holesL) )
    beads.append( sample_box_shell(W.CUB2, W.CUBE, W.WALL_SP, W.WALL_T, hole_disks=holesR) )

    # funnels (5 segmente pe X) – cilindri, cap la cap
    dist = x2_face - x1_face
    seg  = dist / W.FUNNEL_SEG
    for i, R in enumerate(W.FUNNEL_RAD):
        cx = x1_face + (i+0.5)*seg
        L  = seg + 2*W.FUNNEL_PAD   # puțin peste fețele cuburilor (doar vizual)
        beads.append( sample_cyl_shell_x(cx, W.FUNNEL_Y, W.FUNNEL_Z, R, L, W.WALL_SP, W.WALL_T, skip_top_windows=[]) )

    # verticale (y) din fiecare cub până la LOOP_Y
    yb1 = W.CUB1[1]-W.CUBE[1]/2
    yb2 = W.CUB2[1]-W.CUBE[1]/2
    L1  = abs(yb1 - W.LOOP_Y)
    L2  = abs(yb2 - W.LOOP_Y)
    cy1 = 0.5*(yb1 + W.LOOP_Y)
    cy2 = 0.5*(yb2 + W.LOOP_Y)
    beads.append( sample_cyl_shell_y(W.CUB1[0], cy1, 0.0, W.VERT_R, L1, W.WALL_SP, W.WALL_T) )
    beads.append( sample_cyl_shell_y(W.CUB2[0], cy2, 0.0, W.VERT_R, L2, W.WALL_SP, W.WALL_T) )

    # retur (cilindru pe X) — GAURI în manta la x = x1_face și x2_face
    Lret = (x2_face - x1_face) + 2*W.RET_EXTRA
    cxr  = 0.5*(x1_face + x2_face)
    windows = [ dict(xc=x1_face, half_dx=W.RET_PORT_R, r=W.RET_PORT_R),
                dict(xc=x2_face, half_dx=W.RET_PORT_R, r=W.RET_PORT_R) ]
    beads.append( sample_cyl_shell_x(cxr, W.LOOP_Y, 0.0, W.RET_R, Lret, W.WALL_SP, W.WALL_T,
                                     skip_top_windows=windows) )

    BEADS = np.vstack(beads).astype(np.float32)

    # === OBJ (vizual simplu) ===
    items = []
    V,F = cube_mesh(*W.CUB1, *W.CUBE); items.append(dict(name="CUBE_L", mat="mat_cube", V=V, F=F))
    V,F = cube_mesh(*W.CUB2, *W.CUBE); items.append(dict(name="CUBE_R", mat="mat_cube", V=V, F=F))
    for i,R in enumerate(W.FUNNEL_RAD):
        cx = x1_face + (i+0.5)*seg
        V,F = solid_cyl_x_mesh(cx, W.FUNNEL_Y, W.FUNNEL_Z, R, seg+2*W.FUNNEL_PAD, CFG.SIDES)
        items.append(dict(name=f"FUN_{i+1}", mat="mat_fun", V=V, F=F))
    V,F = solid_cyl_y_mesh(W.CUB1[0], cy1, 0.0, W.VERT_R, L1, CFG.SIDES); items.append(dict(name="VERT_L", mat="mat_vert", V=V, F=F))
    V,F = solid_cyl_y_mesh(W.CUB2[0], cy2, 0.0, W.VERT_R, L2, CFG.SIDES); items.append(dict(name="VERT_R", mat="mat_vert", V=V, F=F))
    V,F = solid_cyl_x_mesh(cxr, W.LOOP_Y, 0.0, W.RET_R, Lret, CFG.SIDES); items.append(dict(name="RET", mat="mat_ret", V=V, F=F))

    return BEADS, items

def main():
    BEADS, items = build_all()

    # --- GSD cu beads (tip W) pentru OVITO ---
    dev = hoomd.device.CPU()   # doar scriem un GSD; device-ul nu contează
    sim = hoomd.Simulation(device=dev, seed=1)
    snap = hoomd.Snapshot()
    snap.configuration.box = [CFG.Lx, CFG.Ly, CFG.Lz, 0,0,0]
    snap.particles.N = BEADS.shape[0]
    snap.particles.types = ["W"]
    snap.particles.position[:] = BEADS
    snap.particles.typeid[:] = 0
    sim.create_state_from_snapshot(snap)
    gsd = write.GSD(filename="sim/out11/geom_beads.gsd", mode="wb",
                    trigger=trigger.Periodic(1), filter=hoomd.filter.All())
    sim.operations.writers.append(gsd)
    sim.run(0, write_at_start=True)

    # --- OBJ/MTL pentru vizualizare ---
    write_obj_mtl("sim/out11/geom.obj", "sim/out11/geom.mtl", items)

    print(f"[OK] Pereți (beads): {BEADS.shape[0]} puncte")
    print("     - sim/out11/geom_beads.gsd")
    print("     - sim/out11/geom.obj / geom.mtl")

if __name__ == "__main__":
    main()
