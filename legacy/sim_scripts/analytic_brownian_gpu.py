# sim/analytic_brownian_gpu.py
# Langevin (inerțial) analitic cu reflexii speculare pe GPU (CuPy) sau CPU (NumPy fallback).
# Geometrie = uniune de primitive (box/cyl-x/cyl-y). Materiale per piesă (e_n, e_t).
# Export: XYZ (traj), CSV (tranzit), OBJ (preview).

import os, math, csv, argparse
import numpy as np
try:
    import cupy as cp
    xp = cp  # GPU
    ON_GPU = True
except Exception:
    xp = np  # CPU fallback
    ON_GPU = False

# -------------------- CONFIG --------------------
class CFG:
    Lx, Ly, Lz = 520.0, 320.0, 260.0

    CUBE = (80.0, 80.0, 80.0)
    CUB1 = (-120.0,  70.0, 0.0)
    CUB2 = ( 120.0,  70.0, 0.0)

    FUNNEL_Y, FUNNEL_Z = 70.0, 0.0
    FUNNEL_SEG = 6
    FUNNEL_RAD = [8.0, 35.0, 40.0, 30.0, 22.0, 10.0]
    FUNNEL_PAD = 2.0
    SEAL_OVERLAP = 6.0

    LOOP_Y     = -55.0
    VERT_R     = 12.0
    RET_R      = 42.0
    RET_EXTRA  = 65.0

    # materiale
    MAT = {
        "cube":    dict(e_n=0.95, e_t=1.00),
        "funnel":  dict(e_n=0.98, e_t=1.00),
        "vert":    dict(e_n=0.98, e_t=0.98),
        "ret":     dict(e_n=0.98, e_t=0.95),
        "cap":     dict(e_n=0.90, e_t=0.95),
    }

    # sim
    N        = 150000
    MASS     = 1.0
    GAMMA    = 1.0
    TEMP     = 1.0
    DT       = 1e-3
    STEPS    = 20000
    WRITE_EVERY = 1000
    WRITE_SUBSET = 20000  # câte particule să scriem în XYZ (reduce IO)

    RNG_SEED = 1234
    V0_STD   = 0.2
    MAX_BOUNCES = 4

# -------------------- GEOMETRY --------------------
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

def point_in_union(pos, pieces):
    # pos: (N,3) on xp backend. return mask (N,) bool
    x,y,z = pos[:,0], pos[:,1], pos[:,2]
    N = pos.shape[0]
    inside = xp.zeros(N, dtype=bool)
    for S in pieces:
        if S["kind"]=="box":
            cx,cy,cz = S["center"]; sx,sy,sz = S["size"]
            m = (xp.abs(x-cx) <= sx*0.5) & (xp.abs(y-cy) <= sy*0.5) & (xp.abs(z-cz) <= sz*0.5)
        elif S["kind"]=="cylx":
            cx,cy,cz,R,L = S["cx"],S["cy"],S["cz"],S["R"],S["L"]
            m = (xp.abs(x-cx) <= L*0.5) & (((y-cy)**2 + (z-cz)**2) <= R*R)
        else:
            cx,cy,cz,R,L = S["cx"],S["cy"],S["cz"],S["R"],S["L"]
            m = (xp.abs(y-cy) <= L*0.5) & (((x-cx)**2 + (z-cz)**2) <= R*R)
        inside |= m
    return inside

def nearest_boundary_normal(pos, pieces):
    # approx: găsim "peretele cel mai apropiat" după pernă minimă; vectorizat pe xp.
    N = pos.shape[0]
    best_depth = xp.full(N, xp.inf, dtype=pos.dtype)
    best_n = xp.zeros_like(pos)
    best_e_n = xp.ones(N, dtype=pos.dtype)
    best_e_t = xp.ones(N, dtype=pos.dtype)

    x,y,z = pos[:,0], pos[:,1], pos[:,2]

    def upd(mask, depth, nvec, mat_key):
        # păstrează acolo unde depth < best
        nonlocal best_depth, best_n, best_e_n, best_e_t
        take = mask & (depth < best_depth)
        if not xp.any(take):
            return
        best_depth = xp.where(take, depth, best_depth)
        for j in range(3):
            best_n[:,j] = xp.where(take, nvec[:,j], best_n[:,j])
        e_n = CFG.MAT[mat_key]["e_n"]
        e_t = CFG.MAT[mat_key]["e_t"]
        best_e_n = xp.where(take, e_n, best_e_n)
        best_e_t = xp.where(take, e_t, best_e_t)

    # box faces
    for S in (p for p in pieces if p["kind"]=="box"):
        cx,cy,cz = S["center"]; sx,sy,sz = S["size"]
        dx,dy,dz = x-cx, y-cy, z-cz
        px = sx*0.5 - xp.abs(dx)  # pernă la fețele x±
        py = sy*0.5 - xp.abs(dy)
        pz = sz*0.5 - xp.abs(dz)
        # x-faces
        nx = xp.where(dx>0, 1.0, -1.0)
        nvec = xp.stack([nx, xp.zeros_like(nx), xp.zeros_like(nx)], axis=1)
        upd(xp.ones(N, dtype=bool), px, nvec, S["mat"])
        # y-faces
        ny = xp.where(dy>0, 1.0, -1.0)
        nvec = xp.stack([xp.zeros_like(ny), ny, xp.zeros_like(ny)], axis=1)
        upd(xp.ones(N, dtype=bool), py, nvec, S["mat"])
        # z-faces
        nz = xp.where(dz>0, 1.0, -1.0)
        nvec = xp.stack([xp.zeros_like(nz), xp.zeros_like(nz), nz], axis=1)
        upd(xp.ones(N, dtype=bool), pz, nvec, S["mat"])

    # cyl-x
    for S in (p for p in pieces if p["kind"]=="cylx"):
        cx,cy,cz,R,L = S["cx"],S["cy"],S["cz"],S["R"],S["L"]
        dx = x - cx
        r  = xp.sqrt((y-cy)**2 + (z-cz)**2)
        # capace vs lateral
        px = L*0.5 - xp.abs(dx)
        pr = R - r
        # cap
        nx = xp.where(dx>0, 1.0, -1.0)
        ncap = xp.stack([nx, xp.zeros_like(nx), xp.zeros_like(nx)], axis=1)
        upd(xp.ones(N, dtype=bool), px, ncap, "cap")
        # lateral
        # atenție la r=0 (normală arbitrară)
        eps = 1e-12
        ny = (y-cy) / xp.maximum(r, eps)
        nz = (z-cz) / xp.maximum(r, eps)
        nlat = xp.stack([xp.zeros_like(ny), ny, nz], axis=1)
        upd(xp.ones(N, dtype=bool), pr, nlat, S["mat"])

    # cyl-y
    for S in (p for p in pieces if p["kind"]=="cyly"):
        cx,cy,cz,R,L = S["cx"],S["cy"],S["cz"],S["R"],S["L"]
        dy = y - cy
        r  = xp.sqrt((x-cx)**2 + (z-cz)**2)
        py = L*0.5 - xp.abs(dy)
        pr = R - r
        # cap
        ny = xp.where(dy>0, 1.0, -1.0)
        ncap = xp.stack([xp.zeros_like(ny), ny, xp.zeros_like(ny)], axis=1)
        upd(xp.ones(N, dtype=bool), py, ncap, "cap")
        # lateral
        eps = 1e-12
        nx = (x-cx) / xp.maximum(r, eps)
        nz = (z-cz) / xp.maximum(r, eps)
        nlat = xp.stack([nx, xp.zeros_like(nx), nz], axis=1)
        upd(xp.ones(N, dtype=bool), pr, nlat, S["mat"])

    return best_n, best_depth, best_e_n, best_e_t

def region_name_scalar(p, pieces):
    # doar pentru log tranzit (CPU side), rareori
    x,y,z = p
    for S in pieces:
        if S["kind"]=="box":
            cx,cy,cz=S["center"]; sx,sy,sz=S["size"]
            if (abs(x-cx)<=sx*0.5 and abs(y-cy)<=sy*0.5 and abs(z-cz)<=sz*0.5): return S["name"]
        elif S["kind"]=="cylx":
            if (abs(x-S["cx"])<=S["L"]*0.5 and (y-S["cy"])**2+(z-S["cz"])**2<=S["R"]**2): return S["name"]
        else:
            if (abs(y-S["cy"])<=S["L"]*0.5 and (x-S["cx"])**2+(z-S["cz"])**2<=S["R"]**2): return S["name"]
    return "outside"

# -------------------- SIM --------------------
def simulate(pieces, out_xyz, out_csv, steps, write_every, N, dt, mass, gamma, T, v0_std, seed):
    os.makedirs(os.path.dirname(out_xyz), exist_ok=True)

    rng = np.random.default_rng(seed)

    # init pos uniform în volum (accept-reject) — pe CPU, apoi trimitem pe GPU
    pos_h = np.empty((N,3), dtype=np.float32)
    i=0
    while i<N:
        p = (rng.random(3)-0.5)*np.array([CFG.Lx, CFG.Ly, CFG.Lz], dtype=np.float32)
        if point_in_union(xp.asarray(p)[None,:], pieces).get() if ON_GPU else point_in_union(p[None,:], pieces):
            pos_h[i]=p; i+=1
    vel_h = rng.normal(0.0, v0_std, size=(N,3)).astype(np.float32)

    pos = xp.asarray(pos_h)
    vel = xp.asarray(vel_h)

    a = 1.0 - (gamma/mass)*dt
    b = math.sqrt( (2.0*gamma*T)/(mass*mass) * dt )

    # tranzit tracking (CPU, subset)
    subset = np.arange(min(N, CFG.WRITE_SUBSET), dtype=np.int64)
    last_region = np.array([region_name_scalar(pos_h[k], pieces) for k in subset], dtype=object)
    last_time   = np.zeros(subset.shape[0], dtype=np.int64)
    transitions=[]
    # XYZ writer
    xzf = open(out_xyz,"w")
    def write_xyz(step):
        # extrage subset pe host
        P = pos.get() if ON_GPU else pos
        ids = subset
        xzf.write(f"{ids.shape[0]}\n")
        xzf.write(f"step={step}\n")
        for k in ids:
            x,y,z = P[k]
            xzf.write(f"He {x:.6f} {y:.6f} {z:.6f}\n")

    write_xyz(0)

    for step in range(1, steps+1):
        # Langevin
        noise = xp.asarray(rng.normal(0.0, 1.0, size=vel.shape).astype(np.float32))
        vel = a*vel + b*noise
        # drift propunere
        p1 = pos + vel*dt

        # măști inside/outside
        inside = point_in_union(p1, pieces)
        # pentru cei outside: reflectă (poate de mai multe ori)
        if xp.any(~inside):
            p_work = xp.where(inside[:,None], p1, pos)  # inside rămân cu propunerea
            v_work = vel.copy()
            mask = ~inside
            for _ in range(CFG.MAX_BOUNCES):
                if not xp.any(mask): break
                n, depth, e_n, e_t = nearest_boundary_normal(p_work[mask], pieces)
                # reflexie: v = (-e_n)*vn + e_t*vt
                v = v_work[mask]
                # proiecții
                vn_mag = xp.sum(v*n, axis=1, keepdims=True)
                vn = vn_mag * n
                vt = v - vn
                v_ref = (-e_n[:,None])*vn + (e_t[:,None])*vt
                v_work[mask] = v_ref
                # re-propunere puțin în interior
                p_work[mask] = pos[mask] + v_ref*dt*0.5
                mask = ~point_in_union(p_work, pieces)
            # finalizează
            pos = p_work
            vel = v_work
        else:
            pos = p1

        if (step % write_every)==0:
            # tranzit CPU pe subset
            P = pos.get() if ON_GPU else pos
            for j,k in enumerate(subset):
                rn = region_name_scalar(P[k], pieces)
                if rn != last_region[j]:
                    dsteps = step - last_time[j]
                    transitions.append((step, int(k), last_region[j], rn, int(dsteps)))
                    last_region[j] = rn
                    last_time[j]   = step
            write_xyz(step)

    # scrie CSV
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["step","id","from","to","dsteps"])
        for row in transitions: w.writerow(row)

    xzf.close()

# -------------------- OBJ (preview) --------------------
def write_obj(objp, mtlp, pieces, alpha=0.45, sides=96):
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
    def add_cylx(f, base, cx,cy,cz,R,L, mat, name, sides=sides):
        x0,x1=cx-L/2,cx+L/2
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
    def add_cyly(f, base, cx,cy,cz,R,L, mat, name, sides=sides):
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
    ap.add_argument("--out-dir", type=str, default="sim/out_analytic_gpu")
    args = ap.parse_args()

    pieces = build_geometry()
    os.makedirs(args.out_dir, exist_ok=True)

    write_obj(os.path.join(args.out_dir,"geom.obj"),
              os.path.join(args.out_dir,"geom.mtl"),
              pieces, alpha=0.45)

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
        mass=CFG.MASS,
        gamma=args.gamma,
        T=args.temp,
        v0_std=CFG.V0_STD,
        seed=args.seed
    )

    backend = "GPU (CuPy)" if ON_GPU else "CPU (NumPy)"
    print(f"[{backend}] OK -> {xyz}, {csvp}, {args.out_dir}/geom.obj(.mtl)")

if __name__ == "__main__":
    main()
