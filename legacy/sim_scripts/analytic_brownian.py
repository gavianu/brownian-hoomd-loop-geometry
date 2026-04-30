# sim/analytic_brownian.py
# Langevin (inerțial) într-o geometrie ANALITICĂ (fără “beads”)
# Pereți: reflexie speculară cu coeficienți de material (e_n, e_t) per piesă.
# Export: XYZ (traj), CSV (tranzit), OBJ (preview volum).

import os, math, csv, argparse
import numpy as np

# -------------------- CONFIG --------------------
class CFG:
    # Cutie de vizualizare / limită numerică (NU e "container" de perete)
    Lx, Ly, Lz = 520.0, 320.0, 260.0

    # Geometrie (ca în setup-ul tău)
    CUBE = (80.0, 80.0, 80.0)
    CUB1 = (-120.0,  70.0, 0.0)
    CUB2 = ( 120.0,  70.0, 0.0)

    FUNNEL_Y, FUNNEL_Z = 70.0, 0.0
    FUNNEL_SEG = 6
    FUNNEL_RAD = [8.0, 35.0, 40.0, 30.0, 22.0, 10.0]
    FUNNEL_PAD = 2.0    # mic overlap în cuburi
    SEAL_OVERLAP = 6.0  # ușor overlap între segmente

    LOOP_Y     = -55.0
    VERT_R     = 12.0
    RET_R      = 42.0
    RET_EXTRA  = 65.0

    # Materiale (coeficienți) pe suprafețe
    # e_n: 1.0 elastic, <1.0 plastic (pierdere pe normală)
    # e_t: 1.0 fără frecare, <1.0 "damping" tangential
    MAT = {
        "cube":    dict(e_n=0.95, e_t=1.00),
        "funnel":  dict(e_n=0.98, e_t=1.00),
        "vert":    dict(e_n=0.98, e_t=0.98),
        "ret":     dict(e_n=0.98, e_t=0.95),
        "cap":     dict(e_n=0.90, e_t=0.95),  # capacele returului
    }

    # Simulare
    N        = 15000      # particule
    MASS     = 1.0
    GAMMA    = 1.0
    TEMP     = 1.0
    DT       = 1e-3
    STEPS    = 10000
    WRITE_EVERY = 100

    RNG_SEED = 1234

    # Inițializare: distribuție uniformă în volum + v ~ N(0, v0^2)
    V0_STD   = 0.2

    # Reflexie: număr maxim de bounc-uri într-un pas (siguranță numerică)
    MAX_BOUNCES = 4

# -------------------- GEOMETRIE ANALITICĂ --------------------
# Piese: box (cub), cyl-x, cyl-y, + capace pentru retur
def in_box(p, center, size):
    x,y,z = p; cx,cy,cz = center; sx,sy,sz = size
    return (abs(x-cx) <= sx*0.5) and (abs(y-cy) <= sy*0.5) and (abs(z-cz) <= sz*0.5)

def box_closest_normal(p, center, size):
    # întoarce normală unit pe fața cea mai apropiată + distanță semnată (negativ = în interior)
    x,y,z = p; cx,cy,cz = center; sx,sy,sz = size
    dx = (x - cx); dy = (y - cy); dz = (z - cz)
    # dist la fețe
    px = sx*0.5 - abs(dx)
    py = sy*0.5 - abs(dy)
    pz = sz*0.5 - abs(dz)
    # cea mai mică "pernă" -> fața cea mai apropiată
    m = min(px, py, pz)
    if m == px:
        nx = +1.0 if dx > 0 else -1.0
        return np.array([nx,0.0,0.0],dtype=np.float32), -(m)
    elif m == py:
        ny = +1.0 if dy > 0 else -1.0
        return np.array([0.0,ny,0.0],dtype=np.float32), -(m)
    else:
        nz = +1.0 if dz > 0 else -1.0
        return np.array([0.0,0.0,nz],dtype=np.float32), -(m)

def in_cylx(p, cx, cy, cz, R, L):
    x,y,z = p
    if abs(x - cx) > L*0.5: return False
    rr = (y - cy)**2 + (z - cz)**2
    return rr <= R*R

def cylx_closest_normal(p, cx, cy, cz, R, L):
    # dacă e în interior: normală radială către exterior (fața laterală) sau către capace
    x,y,z = p
    dx = x - cx
    r  = math.hypot(y - cy, z - cz)
    # distanțe la "capace" vs lateral
    px = L*0.5 - abs(dx)  # pernă la capace
    pr = R - r            # pernă la lateral
    if px < pr:
        nx = +1.0 if dx > 0 else -1.0
        return np.array([nx,0.0,0.0],dtype=np.float32), -(px), "cap"
    else:
        if r == 0:  # direcție arbitrară
            return np.array([0.0,1.0,0.0],dtype=np.float32), -(pr), "lat"
        ny = (y - cy)/r; nz = (z - cz)/r
        return np.array([0.0,ny,nz],dtype=np.float32), -(pr), "lat"

def in_cyly(p, cx, cy, cz, R, L):
    x,y,z = p
    if abs(y - cy) > L*0.5: return False
    rr = (x - cx)**2 + (z - cz)**2
    return rr <= R*R

def cyly_closest_normal(p, cx, cy, cz, R, L):
    x,y,z = p
    dy = y - cy
    r  = math.hypot(x - cx, z - cz)
    py = L*0.5 - abs(dy)
    pr = R - r
    if py < pr:
        ny = +1.0 if dy > 0 else -1.0
        return np.array([0.0,ny,0.0],dtype=np.float32), -(py), "cap"
    else:
        if r == 0:
            return np.array([1.0,0.0,0.0],dtype=np.float32), -(pr), "lat"
        nx = (x - cx)/r; nz = (z - cz)/r
        return np.array([nx,0.0,nz],dtype=np.float32), -(pr), "lat"

# definim piese și "în ce volum e punctul"
def build_geometry():
    W = CFG
    x1 = W.CUB1[0] + W.CUBE[0]/2
    x2 = W.CUB2[0] - W.CUBE[0]/2
    dist = x2 - x1
    seg  = dist / W.FUNNEL_SEG

    pieces = []

    # cuburi
    pieces.append(dict(kind="box", name="cube1", center=W.CUB1, size=W.CUBE, mat="cube"))
    pieces.append(dict(kind="box", name="cube2", center=W.CUB2, size=W.CUBE, mat="cube"))

    # funnels
    for i,R in enumerate(W.FUNNEL_RAD):
        cx = x1 + (i+0.5)*seg
        L  = seg + 2*W.FUNNEL_PAD + 2*W.SEAL_OVERLAP
        pieces.append(dict(kind="cylx", name=f"funnel{i+1}", cx=cx, cy=W.FUNNEL_Y, cz=0.0, R=R, L=L, mat="funnel"))

    # verticale
    yb1=W.CUB1[1]-W.CUBE[1]/2; yb2=W.CUB2[1]-W.CUBE[1]/2
    y_top1 = yb1 + W.SEAL_OVERLAP
    y_top2 = yb2 + W.SEAL_OVERLAP
    y_bot  = W.LOOP_Y - W.SEAL_OVERLAP
    L1 = abs(y_top1 - y_bot); cy1 = 0.5*(y_top1 + y_bot)
    L2 = abs(y_top2 - y_bot); cy2 = 0.5*(y_top2 + y_bot)
    pieces.append(dict(kind="cyly", name="vertL", cx=W.CUB1[0], cy=cy1, cz=0.0, R=W.VERT_R, L=L1, mat="vert"))
    pieces.append(dict(kind="cyly", name="vertR", cx=W.CUB2[0], cy=cy2, cz=0.0, R=W.VERT_R, L=L2, mat="vert"))

    # retur
    xR0 = x1 - W.RET_EXTRA
    xR1 = x2 + W.RET_EXTRA
    Lret = xR1 - xR0
    cxr  = 0.5*(xR0 + xR1)
    pieces.append(dict(kind="cylx", name="ret", cx=cxr, cy=W.LOOP_Y, cz=0.0, R=W.RET_R, L=Lret, mat="ret"))

    return pieces

def point_in_union(p, pieces):
    for S in pieces:
        if S["kind"]=="box":
            if in_box(p, S["center"], S["size"]): return True
        elif S["kind"]=="cylx":
            if in_cylx(p, S["cx"], S["cy"], S["cz"], S["R"], S["L"]): return True
        elif S["kind"]=="cyly":
            if in_cyly(p, S["cx"], S["cy"], S["cz"], S["R"], S["L"]): return True
    return False

def nearest_boundary_normal(p, pieces):
    # întoarce normală (unit), cât de adânc e în interior (negativ), numele piesei și tip (cap/lat) pt. material
    best = None
    for S in pieces:
        if S["kind"]=="box":
            n, d = box_closest_normal(p, S["center"], S["size"])
            side = "lat"
        elif S["kind"]=="cylx":
            n, d, side = cylx_closest_normal(p, S["cx"], S["cy"], S["cz"], S["R"], S["L"])
        else:
            n, d, side = cyly_closest_normal(p, S["cx"], S["cy"], S["cz"], S["R"], S["L"])
        # d<0 (inside), cu |d| perna. Căutăm perna minimă (|d| mic => aproape de perete).
        depth = -d
        if best is None or depth < best[1]:
            best = (n, depth, S["name"], S["mat"], side)
    return best  # (n, depth, part_name, mat_name, side)

def region_name(p, pieces):
    hits = []
    for S in pieces:
        inside = (in_box(p,S["center"],S["size"]) if S["kind"]=="box"
                  else (in_cylx(p,S["cx"],S["cy"],S["cz"],S["R"],S["L"]) if S["kind"]=="cylx"
                        else in_cyly(p,S["cx"],S["cy"],S["cz"],S["R"],S["L"])))
        if inside:
            hits.append(S["name"])
    # dacă e în multiple piese (intersecție), alegi un label preferențial
    return hits[0] if hits else "outside"

# -------------------- SIM LANGEVIN + REFLEXII --------------------
def simulate(pieces, out_xyz, out_csv, steps=CFG.STEPS, write_every=CFG.WRITE_EVERY, N=CFG.N,
             dt=CFG.DT, mass=CFG.MASS, gamma=CFG.GAMMA, T=CFG.TEMP, v0_std=CFG.V0_STD, seed=CFG.RNG_SEED):

    os.makedirs(os.path.dirname(out_xyz), exist_ok=True)

    rng = np.random.default_rng(seed)
    # inițializează uniform în volum (accept-reject)
    pos = np.empty((N,3), dtype=np.float32)
    i=0
    while i<N:
        p = (rng.random(3)-0.5)*np.array([CFG.Lx, CFG.Ly, CFG.Lz], dtype=np.float32)
        if point_in_union(p, pieces):
            pos[i]=p; i+=1
    vel = rng.normal(0.0, v0_std, size=(N,3)).astype(np.float32)

    # Langevin coef
    # v_{t+dt} = a * v_t + b * eta  (Euler-Maruyama)
    a = 1.0 - (gamma/mass)*dt
    b = math.sqrt( (2.0*gamma*1.0*T)/(mass*mass) * dt )  # k_B=1

    # tranzit log: ultima regiune și timpi
    last_region = np.array([region_name(pos[k], pieces) for k in range(N)], dtype=object)
    last_time   = np.zeros(N, dtype=np.int64)
    transitions = []  # (step, pid, from, to)

    # XYZ writer (multi-frame, extXYZ simplu)
    xzf = open(out_xyz, "w")
    def write_xyz_frame(step, subset=None):
        ids = subset if subset is not None else range(N)
        M = len(list(ids))
        xzf.write(f"{M}\n")
        xzf.write(f"step={step}\n")
        for k in ids:
            x,y,z = pos[k]
            xzf.write(f"He {x:.6f} {y:.6f} {z:.6f}\n")

    # CSV tranzit
    cf = open(out_csv, "w", newline="")
    cw = csv.writer(cf)
    cw.writerow(["step","id","from","to","dsteps"])

    # prim frame
    write_xyz_frame(0)

    for step in range(1, steps+1):
        # Langevin kick
        vel = a*vel + b*rng.normal(0.0, 1.0, size=vel.shape).astype(np.float32)
        # drift
        dp = vel * dt
        pos_new = pos + dp

        # coliziuni: dacă e afară din volum, reflectă (max CFG.MAX_BOUNCES)
        for k in range(N):
            p0 = pos[k].copy()
            p1 = pos_new[k].copy()
            bounces = 0
            while not point_in_union(p1, pieces) and bounces < CFG.MAX_BOUNCES:
                # ia normală la "cel mai apropiat perete" în p0 (e robust pentru pași mici)
                n, depth, part_name, mat_name, side = nearest_boundary_normal(p0, pieces)
                # reflectă: descompune viteza
                v = vel[k]
                vn = np.dot(v, n) * n            # componenta normală
                vt = v - vn                      # componenta tangentială
                e_n = CFG.MAT[("cap" if side=="cap" else mat_name)]["e_n"]
                e_t = CFG.MAT[("cap" if side=="cap" else mat_name)]["e_t"]
                v_ref = (-e_n)*vn + (e_t)*vt
                vel[k] = v_ref
                # repoziționează: mută punctul din nou dinspre p0 puțin în interior (evită blocaj)
                p1 = p0 + vel[k]*dt*0.5
                bounces += 1
            pos[k] = p1

        # log tranzit de volum
        if (step % write_every) == 0:
            # o mică optimizare: label-uri doar pentru subset (sau pentru toți)
            for k in range(N):
                rn = region_name(pos[k], pieces)
                if rn != last_region[k]:
                    dsteps = step - last_time[k]
                    transitions.append((step, k, last_region[k], rn, dsteps))
                    last_region[k] = rn
                    last_time[k]   = step
            write_xyz_frame(step)

    # scrie tranzit
    for st,pid,fr,to,ds in transitions:
        cw.writerow([st,pid,fr,to,ds])
    xzf.close(); cf.close()

# -------------------- OBJ preview (translucid) --------------------
def write_obj(objp, mtlp, pieces, alpha=0.4, sides=96):
    # doar preview simplu: cilindri cu capace, cuburi cu fețe
    with open(mtlp,"w") as m:
        m.write(f"newmtl cube\nKd 0.7 0.7 0.95\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl funnel\nKd 0.7 0.95 0.7\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl vert\nKd 0.95 0.95 0.7\nd {alpha}\nillum 2\n\n")
        m.write(f"newmtl ret\nKd 0.95 0.7 0.7\nd {alpha}\nillum 2\n\n")
    def add_box(f, base, center, size, mat, name):
        cx,cy,cz=center; sx,sy,sz=size
        x0,x1=cx-sx/2,cx+sx/2; y0,y1=cy-sy/2,cy+sy/2; z0,z1=cz-sz/2,cz+sz/2
        V=[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
           (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        F=[(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),
           (1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)]
        f.write(f"o {name}\nusemtl {mat}\n")
        for v in V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a,b,c in F: f.write(f"f {a+1+base} {b+1+base} {c+1+base}\n")
        return base+len(V)
    def add_cylx(f, base, cx,cy,cz,R,L, mat, name):
        x0,x1=cx-L/2,cx+L/2
        # inele
        r0=[(x0, cy+R*math.cos(2*math.pi*i/sides), cz+R*math.sin(2*math.pi*i/sides)) for i in range(sides)]
        r1=[(x1, cy+R*math.cos(2*math.pi*i/sides), cz+R*math.sin(2*math.pi*i/sides)) for i in range(sides)]
        V=r0+r1
        f.write(f"o {name}\nusemtl {mat}\n")
        for v in V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        b=base
        for i in range(sides):
            a=b+i; c=b+sides+((i+1)%sides); d=b+sides+i; e=b+((i+1)%sides)
            f.write(f"f {a+1} {e+1} {c+1}\n"); f.write(f"f {a+1} {c+1} {d+1}\n")
        return base+len(V)
    def add_cyly(f, base, cx,cy,cz,R,L, mat, name):
        y0,y1=cy-L/2,cy+L/2
        r0=[(cx+R*math.cos(2*math.pi*i/sides), y0, cz+R*math.sin(2*math.pi*i/sides)) for i in range(sides)]
        r1=[(cx+R*math.cos(2*math.pi*i/sides), y1, cz+R*math.sin(2*math.pi*i/sides)) for i in range(sides)]
        V=r0+r1
        f.write(f"o {name}\nusemtl {mat}\n")
        for v in V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        b=base
        for i in range(sides):
            a=b+i; c=b+sides+((i+1)%sides); d=b+sides+i; e=b+((i+1)%sides)
            f.write(f"f {a+1} {c+1} {e+1}\n"); f.write(f"f {a+1} {d+1} {c+1}\n")
        return base+len(V)
    with open(objp,"w") as f:
        f.write(f"mtllib {os.path.basename(mtlp)}\n")
        base=0
        for S in pieces:
            if S["kind"]=="box":
                base=add_box(f, base, S["center"], S["size"], S["mat"], S["name"])
            elif S["kind"]=="cylx":
                base=add_cylx(f, base, S["cx"],S["cy"],S["cz"],S["R"],S["L"], S["mat"], S["name"])
            else:
                base=add_cyly(f, base, S["cx"],S["cy"],S["cz"],S["R"],S["L"], S["mat"], S["name"])

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CFG.STEPS)
    ap.add_argument("--n", type=int, default=CFG.N)
    ap.add_argument("--dt", type=float, default=CFG.DT)
    ap.add_argument("--gamma", type=float, default=CFG.GAMMA)
    ap.add_argument("--temp", type=float, default=CFG.TEMP)
    ap.add_argument("--write-every", type=int, default=CFG.WRITE_EVERY)
    ap.add_argument("--seed", type=int, default=CFG.RNG_SEED)
    ap.add_argument("--out-dir", type=str, default="sim/out_analytic")
    args = ap.parse_args()

    pieces = build_geometry()
    os.makedirs(args.out_dir, exist_ok=True)

    # OBJ preview
    write_obj(os.path.join(args.out_dir,"geom.obj"),
              os.path.join(args.out_dir,"geom.mtl"),
              pieces, alpha=0.45)

    # Simulare
    xyz = os.path.join(args.out_dir, "traj.xyz")
    csvp= os.path.join(args.out_dir, "transitions.csv")
    simulate(
        pieces,
        out_xyz=xyz,
        out_csv=csvp,
        steps=args.steps,
        write_every=args.write_every,
        N=args.n,
        dt=args.dt,
        gamma=args.gamma,
        T=args.temp,
        seed=args.seed
    )
    print(f"[OK] scris:\n  - {xyz}\n  - {csvp}\n  - {args.out_dir}/geom.obj(.mtl)")

if __name__ == "__main__":
    main()
