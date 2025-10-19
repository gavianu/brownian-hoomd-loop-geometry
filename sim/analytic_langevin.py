# sim/analytic_langevin.py
# Langevin + pereți analitici (box / cyl-x / cyl-y) cu materiale pe piesă.
# Salvează: GSD (pos+vel, subset), XYZ complet, counts per piesă, transitions (toți),
# piece-id map per write, histograme distanță-la-perete pe piesă.
# CPU by default; GPU opțional (CuPy). Reflexiile se fac pe CPU (robuste).

import os, math, argparse, csv, time
import numpy as np
import sys, time

# ---------- optional GPU (CuPy) ----------
try:
    import cupy as cp
except Exception:
    cp = None

# ---------- optional GSD ----------
try:
    import gsd, gsd.hoomd
    HAVE_GSD = True
except Exception:
    HAVE_GSD = False

# ---------------- CONFIG ----------------
class CFG:
    # Box vizual (nu e container): [-Lx/2, Lx/2] etc. pentru sampling inițial
    Lx, Ly, Lz = 520.0, 320.0, 260.0

    # Geometrie ansamblu
    CUBE = (80.0, 80.0, 80.0)
    CUB1 = (-120.0,  70.0, 0.0)
    CUB2 = ( 120.0,  70.0, 0.0)

    FUNNEL_Y =  70.0
    FUNNEL_Z =   0.0
    FUNNEL_SEG = 6
    FUNNEL_RAD = [8.0, 35.0, 40.0, 30.0, 22.0, 10.0]
    FUNNEL_PAD = 2.0
    SEAL_OVERLAP = 6.0

    LOOP_Y   = -55.0
    VERT_R   = 12.0
    RET_R    = 42.0
    RET_EXTRA= 65.0

    # Materiale pe piesă (e_n: coef normal, beta_t: fricțiune tangențială „sticking”)
    MAT_CUBE_L = dict(e_n=0.65, beta_t=0.8)
    MAT_CUBE_R = dict(e_n=0.05, beta_t=0.15)
    MAT_FUN  = [dict(e_n=0.3, beta_t=0.4),dict(e_n=0.9, beta_t=0.95),dict(e_n=0.9, beta_t=0.95),dict(e_n=0.9, beta_t=0.95),dict(e_n=0.9, beta_t=0.95),dict(e_n=0.9, beta_t=0.95)]
    MAT_VERT = dict(e_n=0.98, beta_t=0.985)
    MAT_RET  = dict(e_n=0.98, beta_t=0.985)

    # Simulare (default; pot fi suprascrise din CLI)
    N          = 30000
    DT         = 0.001
    STEPS      = 20000
    WRITE_EVERY= 1000
    MASS       = 1.0
    GAMMA      = 1.0
    KT         = 1.0
    SEED       = 42

    # I/O
    OUT_DIR  = "sim/out_langevin_3"
    GSD_NAME = "run.gsd"
    XYZ_NAME = "run.xyz"

    # Vizual în GSD: scriem subset pentru performanță
    GSD_SUBSET = 15000   # 0 = scrie toți (mare fișier). Recomandă 10k–50k.

    # Tracking și loguri
    TRACK_K       = 200          # câte particule „selectate” (HeSel)
    LOG_TRANS_EVERY_STEP = True  # log from→to la fiecare pas (mare fișier la N mare)
    WALL_HIST_BINS = 40          # număr de bin-uri pentru histograme distanță-la-perete
    SAVE_WALL_DIST_FOR_TRACK = True  # salvează distanțe brute pt particulele urmărite

    # Reflexii: sub-stepping adaptiv (max deplasare fracțiune din distanța la perete)
    COLLISION_CFL = 0.5          # 0.5 = sigur, 0.9 mai rapid dar risc mare
    MAX_SUBSTEPS  = 8

# ------------- Geometrie & materiale -------------
def make_geometry():
    W = CFG
    pieces = []
    # CUB1 / CUB2
    pieces.append(dict(type='box', center=W.CUB1, size=W.CUBE, **W.MAT_CUBE_L, name='CUBE_L'))
    pieces.append(dict(type='box', center=W.CUB2, size=W.CUBE, **W.MAT_CUBE_R, name='CUBE_R'))

    # Funnels pe X
    x1 = W.CUB1[0] + W.CUBE[0]/2
    x2 = W.CUB2[0] - W.CUBE[0]/2
    dist = x2 - x1
    seg  = dist / W.FUNNEL_SEG
    for i, R in enumerate(W.FUNNEL_RAD):
        cx = x1 + (i+0.5)*seg
        L  = seg + 2*W.FUNNEL_PAD + 2*W.SEAL_OVERLAP
        pieces.append(dict(type='cylx', cx=cx, cy=W.FUNNEL_Y, cz=W.FUNNEL_Z, R=R, L=L,
                           **W.MAT_FUN[i], name=f'FUN_{i+1}'))

    # Verticale pe Y
    yb1 = W.CUB1[1] - W.CUBE[1]/2
    yb2 = W.CUB2[1] - W.CUBE[1]/2
    y_top1 = yb1 + W.SEAL_OVERLAP
    y_bot1 = W.LOOP_Y - W.SEAL_OVERLAP
    L1 = abs(y_top1 - y_bot1); cy1 = 0.5*(y_top1+y_bot1)

    y_top2 = yb2 + W.SEAL_OVERLAP
    y_bot2 = W.LOOP_Y - W.SEAL_OVERLAP
    L2 = abs(y_top2 - y_bot2); cy2 = 0.5*(y_top2+y_bot2)

    pieces.append(dict(type='cyly', cx=W.CUB1[0], cy=cy1, cz=0.0, R=W.VERT_R, L=L1,
                       **W.MAT_VERT, name='VERT_L'))
    pieces.append(dict(type='cyly', cx=W.CUB2[0], cy=cy2, cz=0.0, R=W.VERT_R, L=L2,
                       **W.MAT_VERT, name='VERT_R'))

    # Retur pe X
    xR0 = x1 - W.RET_EXTRA
    xR1 = x2 + W.RET_EXTRA
    Lret= xR1 - xR0
    cxr = 0.5*(xR0+xR1)
    pieces.append(dict(type='cylx', cx=cxr, cy=W.LOOP_Y, cz=0.0, R=W.RET_R, L=Lret,
                       **W.MAT_RET, name='RET'))
    return pieces

def piece_names(pieces):
    return [p['name'] for p in pieces]

# ---------- inside tests (vectorizate pe CPU) ----------
def inside_box(P, center, size):
    x,y,z = P[...,0], P[...,1], P[...,2]
    cx,cy,cz = center; sx,sy,sz = size
    return (np.abs(x-cx) <= sx*0.5) & (np.abs(y-cy) <= sy*0.5) & (np.abs(z-cz) <= sz*0.5)

def inside_cylx(P, cx, cy, cz, R, L):
    x,y,z = P[...,0], P[...,1], P[...,2]
    return (np.abs(x-cx) <= L*0.5) & (((y-cy)**2 + (z-cz)**2) <= R*R)

def inside_cyly(P, cx, cy, cz, R, L):
    x,y,z = P[...,0], P[...,1], P[...,2]
    return (np.abs(y-cy) <= L*0.5) & (((x-cx)**2 + (z-cz)**2) <= R*R)

def point_in_union(P, pieces):
    m = np.zeros(P.shape[0], dtype=bool)
    for S in pieces:
        if S['type']=='box':
            m |= inside_box(P, S['center'], S['size'])
        elif S['type']=='cylx':
            m |= inside_cylx(P, S['cx'], S['cy'], S['cz'], S['R'], S['L'])
        else:
            m |= inside_cyly(P, S['cx'], S['cy'], S['cz'], S['R'], S['L'])
    return m

def locate_points_in_pieces(P, pieces):
    idx = np.full(P.shape[0], -1, dtype=np.int32)
    m_any = np.zeros(P.shape[0], dtype=bool)
    for k, S in enumerate(pieces):
        if S['type']=='box':
            m = inside_box(P, S['center'], S['size'])
        elif S['type']=='cylx':
            m = inside_cylx(P, S['cx'], S['cy'], S['cz'], S['R'], S['L'])
        else:
            m = inside_cyly(P, S['cx'], S['cy'], S['cz'], S['R'], S['L'])
        set_here = (~m_any) & m
        idx[set_here] = k
        m_any |= m
    return idx

# ---------- distanța la cel mai apropiat perete al piesei curente ----------
def wall_distance_one(p, S):
    if S['type']=='box':
        cx,cy,cz = S['center']; sx,sy,sz = S['size']
        dx = sx*0.5 - abs(p[0]-cx)
        dy = sy*0.5 - abs(p[1]-cy)
        dz = sz*0.5 - abs(p[2]-cz)
        return max(0.0, min(dx, dy, dz))
    if S['type']=='cylx':
        cx,cy,cz,R,L = S['cx'],S['cy'],S['cz'],S['R'],S['L']
        rx = L*0.5 - abs(p[0]-cx)
        r = math.hypot(p[1]-cy, p[2]-cz)
        rr = R - r
        return max(0.0, min(rx, rr))
    if S['type']=='cyly':
        cx,cy,cz,R,L = S['cx'],S['cy'],S['cz'],S['R'],S['L']
        ry = L*0.5 - abs(p[1]-cy)
        r = math.hypot(p[0]-cx, p[2]-cz)
        rr = R - r
        return max(0.0, min(ry, rr))
    return 0.0

# ---------- coliziune cu o piesă (proiecție + reflexie) ----------
def collide_one_piece(p_old, p_new, v, S):
    # întoarce (p_corectat, v_corectată, hit:bool)
    e_n, beta_t = S['e_n'], S['beta_t']

    if S['type']=='box':
        cx,cy,cz = S['center']; sx,sy,sz = S['size']
        x,y,z = p_new
        nx=ny=nz=0.0
        if x > cx + sx*0.5: nx = +1.0; x = cx + sx*0.5
        elif x < cx - sx*0.5: nx = -1.0; x = cx - sx*0.5
        elif y > cy + sy*0.5: ny = +1.0; y = cy + sy*0.5
        elif y < cy - sy*0.5: ny = -1.0; y = cy - sy*0.5
        elif z > cz + sz*0.5: nz = +1.0; z = cz + sz*0.5
        elif z < cz - sz*0.5: nz = -1.0; z = cz - sz*0.5
        n = np.array([nx,ny,nz], dtype=np.float32)
        if np.linalg.norm(n)<0.5:
            return p_new, v, False
        n /= np.linalg.norm(n)
        vn = np.dot(v,n)*n
        vt = v - vn
        v_ref = -e_n*vn + beta_t*vt
        return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True

    if S['type']=='cylx':
        cx,cy,cz,R,L = S['cx'],S['cy'],S['cz'],S['R'],S['L']
        x,y,z = p_new
        # capete (X)
        if abs(x-cx) > L*0.5:
            nx = +1.0 if x>cx else -1.0
            n = np.array([nx,0.0,0.0],dtype=np.float32)
            x = cx + math.copysign(L*0.5, x-cx)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
        # manta cilindrică
        ry,rz = y-cy, z-cz
        r = math.hypot(ry,rz)
        if r > R:
            n = np.array([0.0, ry, rz], dtype=np.float32)
            n /= np.linalg.norm(n)
            y = cy + R * (ry / r); z = cz + R * (rz / r)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
        return p_new, v, False

    if S['type']=='cyly':
        cx,cy,cz,R,L = S['cx'],S['cy'],S['cz'],S['R'],S['L']
        x,y,z = p_new
        # capete (Y)
        if abs(y-cy) > L*0.5:
            ny = +1.0 if y>cy else -1.0
            n = np.array([0.0,ny,0.0],dtype=np.float32)
            y = cy + math.copysign(L*0.5, y-cy)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
        # manta cilindrică
        rx,rz = x-cx, z-cz
        r = math.hypot(rx,rz)
        if r > R:
            n = np.array([rx,0.0,rz],dtype=np.float32); n/=np.linalg.norm(n)
            x = cx + R * (rx / r); z = cz + R * (rz / r)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
        return p_new, v, False

    return p_new, v, False

# ---------- sampling inițial uniform în union ----------
def sample_uniform_in_union(N, pieces, seed=0):
    rng = np.random.default_rng(seed)
    Lx,Ly,Lz = CFG.Lx, CFG.Ly, CFG.Lz
    pos = np.empty((N,3), dtype=np.float32)
    i = 0
    while i < N:
        p = (rng.random(3)-0.5)*np.array([Lx,Ly,Lz],dtype=np.float32)
        if point_in_union(p[None,:], pieces)[0]:
            pos[i]=p; i+=1
    return pos

def append_xyz(path, pos):
    N = pos.shape[0]
    with open(path, "a") as f:
        f.write(f"{N}\nframe\n")
        for i in range(N):
            x,y,z = pos[i]
            f.write(f"He {x:.6f} {y:.6f} {z:.6f}\n")


t0 = time.time()
last_print_len = 0

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CFG.STEPS)
    ap.add_argument("--n", type=int, default=CFG.N)
    ap.add_argument("--dt", type=float, default=CFG.DT)
    ap.add_argument("--gamma", type=float, default=CFG.GAMMA)
    ap.add_argument("--kt", type=float, default=CFG.KT)
    ap.add_argument("--write-every", type=int, default=CFG.WRITE_EVERY)
    ap.add_argument("--gpu", type=int, default=-1, help="-1=CPU, 0..=GPU id (CuPy)")
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--gsd-subset", type=int, default=CFG.GSD_SUBSET)
    ap.add_argument("--log-every", type=int, default=200, help="la câți pași log de progres")
    ap.add_argument("--quiet", action="store_true", help="fără progres (doar erori)")

    args = ap.parse_args()


    def human_time(seconds):
        if not math.isfinite(seconds):
            return "—"
        s = max(0.0, float(seconds))
        m, s = divmod(int(round(s)), 60)
        h, m = divmod(m, 60)
        if h:   return f"{h}h {m}m {s}s"
        if m:   return f"{m}m {s}s"
        return f"{s}s"

    def log_progress(step, total, extra=""):
        if args.quiet:
            return
        elapsed = time.time() - t0
        rate = step / elapsed if elapsed > 0 else 0.0
        eta  = (total - step) / rate if (rate > 0 and step > 0) else float("inf")
        msg = f"[{step}/{total}] {rate:7.1f} steps/s  elapsed {human_time(elapsed)}  ETA {human_time(eta)}"
        if extra:
            msg += f"  | {extra}"

        global last_print_len
        pad = " " * max(0, last_print_len - len(msg))
        sys.stdout.write("\r" + msg + pad)
        sys.stdout.flush()
        last_print_len = len(msg)
        if step >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # backend numeric
    xp = np
    if args.gpu >= 0 and cp is not None:
        try:
            cp.cuda.Device(args.gpu).use()
            xp = cp
            print(f"[GPU] CuPy on device {args.gpu}")
        except Exception as e:
            print(f"[WARN] CuPy device {args.gpu} unavailable ({e}); falling back to CPU.")
            xp = np
    elif args.gpu >= 0 and cp is None:
        print("[WARN] CuPy not installed; running on CPU.")

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    xyz_path = os.path.join(CFG.OUT_DIR, CFG.XYZ_NAME)
    if os.path.exists(xyz_path): os.remove(xyz_path)

    pieces = make_geometry()
    names  = piece_names(pieces)

    # init
    pos0 = sample_uniform_in_union(args.n, pieces, seed=args.seed)
    vel0 = np.zeros_like(pos0, dtype=np.float32)

    # piece index (CPU)
    piece_idx = locate_points_in_pieces(pos0, pieces)

    # tracking
    track_k = min(CFG.TRACK_K, args.n)
    track_ids = np.arange(track_k, dtype=np.int32)

    # RNG pentru Langevin
    rng = np.random.default_rng(args.seed)

    # mută pe GPU dacă e cazul
    pos = xp.asarray(pos0)
    vel = xp.asarray(vel0)

    m  = CFG.MASS
    dt = args.dt
    gamma = args.gamma
    kt = args.kt
    noise_sigma = math.sqrt(2.0*gamma*kt/m) * math.sqrt(dt)

    # GSD
    gsd_file = None
    if HAVE_GSD:
        gsd_path = os.path.join(CFG.OUT_DIR, CFG.GSD_NAME)
        gsd_file = gsd.hoomd.open(name=gsd_path, mode='w')
        print(f"[GSD] writing to {gsd_path}")
    else:
        print("[WARN] GSD not available; only XYZ/CSV will be written.")

    # CSV loguri
    trans_path  = os.path.join(CFG.OUT_DIR, "transitions.csv")
    counts_path = os.path.join(CFG.OUT_DIR, "piece_counts.csv")
    with open(trans_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["step","particle_id","from","to"])
    with open(counts_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["step"]+names)

    # helper I/O
    def to_cpu(a):
        if cp is not None and isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
        return np.asarray(a)

    def max_speed_cpu(vel_xp):
        v = to_cpu(vel_xp)                     # CuPy -> NumPy dacă e cazul
        if v.size == 0:
            return 0.0
        v = np.where(np.isfinite(v), v, 0.0)   # curăță NaN/Inf
        # norma pe fiecare particulă
        speeds = np.linalg.norm(v, axis=1)
        # max cu identitate (evită eroarea pe array gol)
        return float(np.max(speeds)) if speeds.size else 0.0

    def write_gsd_frame(step, pos_cpu, vel_cpu):
        if gsd_file is None: return
        N = pos_cpu.shape[0]
        subset = min(args.gsd_subset if args.gsd_subset>0 else N, N)
        sel = np.arange(subset, dtype=np.int64)
        frame = gsd.hoomd.Frame()
        frame.configuration.step = int(step)
        frame.configuration.box  = [CFG.Lx, CFG.Ly, CFG.Lz, 0, 0, 0]
        frame.particles.N = subset
        frame.particles.position = pos_cpu[sel].astype(np.float32)
        frame.particles.velocity = vel_cpu[sel].astype(np.float32)
        frame.particles.types = ["He","HeSel"]
        typeid = np.zeros(subset, dtype=np.int32)
        typeid[typeid < 0] = 0
        typeid[np.intersect1d(sel, track_ids, assume_unique=False)] = 1
        frame.particles.typeid = typeid
        frame.particles.diameter = np.full(subset, 1.0, dtype=np.float32)
        gsd_file.append(frame)

    def write_piece_map(step, pos_cpu):
        idx = locate_points_in_pieces(pos_cpu, pieces)
        outp = os.path.join(CFG.OUT_DIR, f"piece_map_step{step:06d}.csv")
        with open(outp,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["id","piece"])
            for i in range(idx.shape[0]):
                nm = names[idx[i]] if idx[i]>=0 else "OUT"
                w.writerow([i,nm])
        return idx

    def wall_histograms(step, pos_cpu, idx_cpu):
        # histograme pe piesă + (opțional) distanțele brute ale particulelor urmărite
        edges = None
        rows  = [["piece","bin_left","bin_right","count"]]
        track_rows = [["id","piece","dist"]]
        for k,S in enumerate(pieces):
            mask = (idx_cpu==k)
            if not np.any(mask): continue
            dists = np.empty(mask.sum(), dtype=np.float32)
            ii=0
            for i,flag in enumerate(mask):
                if flag:
                    dists[ii] = wall_distance_one(pos_cpu[i], S); ii+=1
            if edges is None:
                # praguri în funcție de mărimea caracteristică a piesei
                if S['type']=='box':
                    sx,sy,sz = S['size']; dmax = min(sx,sy,sz)*0.5
                elif S['type']=='cylx':
                    dmax = min(S['R'], S['L']*0.5)
                else:
                    dmax = min(S['R'], S['L']*0.5)
                edges = np.linspace(0.0, dmax, CFG.WALL_HIST_BINS+1)
            hist,_ = np.histogram(dists, bins=edges)
            for b in range(len(edges)-1):
                rows.append([names[k], f"{edges[b]:.6f}", f"{edges[b+1]:.6f}", int(hist[b])])
        with open(os.path.join(CFG.OUT_DIR, f"wall_hist_step{step:06d}.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerows(rows)

        if CFG.SAVE_WALL_DIST_FOR_TRACK and track_ids.size>0:
            for pid in track_ids:
                kk = idx_cpu[pid]
                nm = names[kk] if kk>=0 else "OUT"
                dist = 0.0 if kk<0 else wall_distance_one(pos_cpu[pid], pieces[kk])
                track_rows.append([int(pid), nm, f"{dist:.6f}"])
            with open(os.path.join(CFG.OUT_DIR, f"wall_dist_track_step{step:06d}.csv"),"w",newline="") as f:
                w=csv.writer(f); w.writerows(track_rows)

    # scrie frame inițial
    pos_cpu = to_cpu(pos); vel_cpu = to_cpu(vel)
    append_xyz(xyz_path, pos_cpu)
    write_gsd_frame(0, pos_cpu, vel_cpu)

    idx_cpu = write_piece_map(0, pos_cpu)
    # counts
    with open(counts_path,"a",newline="") as f:
        w=csv.writer(f)
        counts = [(idx_cpu==k).sum() for k in range(len(pieces))]
        w.writerow([0]+counts)
    wall_histograms(0, pos_cpu, idx_cpu)

    # --------------- integrare ---------------
    def reflect_cpu(p_old, p_new, v, prev_k):
        """Reflectă pe piesa precedentă dacă iese din union și nu intră într-o piesă vecină."""
        # dacă e încă în union -> acceptă
        if point_in_union(p_new[None,:], pieces)[0]:
            return p_new, v, prev_k, False
        # dacă iese din prev_k:
        S = pieces[prev_k] if prev_k>=0 else None
        if S is not None:
            in_prev_old = point_in_union(p_old[None,:], [S])[0]
            in_prev_new = point_in_union(p_new[None,:], [S])[0]
            if in_prev_old and not in_prev_new:
                # intră în altă piesă? atunci nu reflectăm, lăsăm să treacă
                in_other = point_in_union(p_new[None,:], [T for i,T in enumerate(pieces) if i!=prev_k])[0]
                if in_other:
                    new_k = locate_points_in_pieces(p_new[None,:], pieces)[0]
                    return p_new, v, new_k, True if new_k!=prev_k else False
                # altfel: coliziune cu prev_k
                p_corr, v_corr, hit = collide_one_piece(p_old, p_new, v, S)
                if hit:
                    return p_corr, v_corr, prev_k, False
        # caz limită: era OUT sau nu detectăm prev_k: reflectare „elastică” inversând viteza
        return p_old, -0.5*v, prev_k, False

    steps = args.steps
    write_every = max(1, args.write_every)

    start = time.time()
    for step in range(1, steps+1):
        # Langevin (Euler–Maruyama)
        if cp is not None and xp is cp:
            xi = cp.asarray(rng.standard_normal(size=pos.shape, dtype=np.float32))
        else:
            xi = np.asarray(rng.standard_normal(size=pos.shape), dtype=np.float32)

        vel = vel + (-gamma/CFG.MASS)*vel*dt + (noise_sigma/CFG.MASS)*xi

        # sub-stepping adaptiv (CPU reflect, sigur)
        # facem 1..MAX_SUBSTEPS sub-pași astfel încât deplasarea să nu depășească
        # COLLISION_CFL * distanța la perete (aproximăm distanța cu piesa curentă)
        nsub = 1
        if CFG.MAX_SUBSTEPS>1:
            pos_cpu = to_cpu(pos)
            max_speed = max_speed_cpu(vel)
            step_len  = max_speed * dt + 1e-12
            # estimăm un scale global (mai simplu): max speed * dt vs mediana distanței la perete pe track subset
            sample_ids = track_ids if track_ids.size>0 else np.arange(min(500, pos_cpu.shape[0]))
            dist_samp = []
            for pid in sample_ids:
                k = piece_idx[pid]
                d = 1.0
                if k>=0:
                    d = wall_distance_one(pos_cpu[pid], pieces[k])
                dist_samp.append(d)
            dmed = np.median(dist_samp) if len(dist_samp)>0 else 1.0
            if dmed>1e-6:
                est = step_len / (CFG.COLLISION_CFL * dmed)
                nsub = int(min(CFG.MAX_SUBSTEPS, max(1, math.ceil(est))))
        sub_dt = dt/float(nsub)

        for _ in range(nsub):
            p_old = pos
            pos = pos + vel*sub_dt
            # reflect (pe CPU)
            p_old_cpu = to_cpu(p_old)
            p_new_cpu = to_cpu(pos)
            v_cpu     = to_cpu(vel)

            # detectăm tranziții & coliziuni
            # previous indices per particle (CPU)
            prev_idx = piece_idx.copy()
            # actualizează temporar new_idx
            new_idx = locate_points_in_pieces(p_new_cpu, pieces)

            for i in range(p_new_cpu.shape[0]):
                p_new_cpu[i], v_cpu[i], piece_idx[i], crossed = reflect_cpu(
                    p_old_cpu[i], p_new_cpu[i], v_cpu[i], prev_idx[i]
                )
                # dacă a traversat din prev_idx în piece_idx[i] -> înregistrăm tranziție
                if CFG.LOG_TRANS_EVERY_STEP and crossed and (prev_idx[i] != piece_idx[i]):
                    with open(trans_path,"a",newline="") as f:
                        w=csv.writer(f)
                        fr = names[prev_idx[i]] if prev_idx[i]>=0 else "OUT"
                        to = names[piece_idx[i]] if piece_idx[i]>=0 else "OUT"
                        w.writerow([step, i, fr, to])

            pos = xp.asarray(p_new_cpu) if xp is cp else p_new_cpu
            vel = xp.asarray(v_cpu)     if xp is cp else v_cpu

        # OUTPUT
        if (step % write_every)==0:
            pos_cpu = to_cpu(pos); vel_cpu = to_cpu(vel)
            append_xyz(xyz_path, pos_cpu)
            write_gsd_frame(step, pos_cpu, vel_cpu)

            idx_cpu = write_piece_map(step, pos_cpu)
            with open(counts_path,"a",newline="") as f:
                w=csv.writer(f)
                counts = [(idx_cpu==k).sum() for k in range(len(pieces))]
                w.writerow([step]+counts)
            wall_histograms(step, pos_cpu, idx_cpu)
        if (step % args.log_every) == 0 or step == args.steps:
            log_progress(step, args.steps, extra="write")
    if gsd_file is not None:
        gsd_file.close()
    print(f"[OK] done in {time.time()-start:.1f}s.")

if __name__ == "__main__":
    main()
