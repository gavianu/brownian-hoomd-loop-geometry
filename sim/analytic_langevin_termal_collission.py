# sim/analytic_langevin_stabil_gpu_cpu.py
# Versiune stabilă cu opțiune de coliziuni vectorizate pe GPU (CuPy)
# - Fix coeficient zgomot Langevin
# - Tranziții „în volum” între piese (fără reflectare falsă)
# - Mod hibrid (CPU reflect) sau complet GPU (--gpu-collide)
# - Opțiuni de perf: --no-io, --adapt-every

import os, math, argparse, csv, time, sys
import numpy as np

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
    # Box vizual pentru sampling inițial
    Lx, Ly, Lz = 520.0, 320.0, 260.0

    # Geometrie ansamblu
    CUBE = (80.0, 80.0, 80.0)
    CUB1 = (-120.0,  70.0, 0.0)
    CUB2 = ( 120.0,  70.0, 0.0)

    FUNNEL_Y =  90.0
    FUNNEL_Z =   0.0
    FUNNEL_SEG = 6
    FUNNEL_RAD = [8.0, 17.0, 25.0, 15.0, 12.0, 10.0]
    FUNNEL_PAD = 2.0
    SEAL_OVERLAP = 6.0

    LOOP_Y   = 45.0
    VERT_R   = 12.0
    RET_R    = 12.0
    RET_EXTRA= 10.0

    # Materiale pe piesă
    MAT_CUBE_L = dict(e_n=0.9, beta_t=0.7)
    MAT_CUBE_R = dict(e_n=0.3, beta_t=0.55)
    MAT_FUN  = [dict(e_n=0.85, beta_t=0.85),dict(e_n=0.98, beta_t=0.3),dict(e_n=0.98, beta_t=0.3),dict(e_n=0.98, beta_t=0.3),dict(e_n=0.98, beta_t=0.3),dict(e_n=0.98, beta_t=0.3)]
    MAT_VERT = dict(e_n=0.98, beta_t=0.02)
    MAT_RET  = dict(e_n=0.98, beta_t=0.02)

    # Simulare (default)
    N          = 30000
    DT         = 0.001
    STEPS      = 20000
    WRITE_EVERY= 1000
    MASS       = 1.0
    GAMMA      = 1.0
    KT         = 1.0
    SEED       = 42

    # I/O
    OUT_DIR  = "sim/out_langevin_stab_gpu_fast_15"
    GSD_NAME = "run.gsd"
    XYZ_NAME = "run.xyz"

    # Vizual în GSD: scriem subset
    GSD_SUBSET = 15000

    # Tracking și loguri
    TRACK_K       = 200
    LOG_TRANS_EVERY_STEP = True
    WALL_HIST_BINS = 40
    SAVE_WALL_DIST_FOR_TRACK = True

    # Reflexii: sub-stepping adaptiv
    COLLISION_CFL = 0.5
    MAX_SUBSTEPS  = 8

# map tipuri piese
PIECE_TYPES = {"box":0, "cylx":1, "cyly":2}


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
            if S["type"]=="box":
                base=add_box(f, base, S["center"], S["size"], S["e_n"], S["name"])
            elif S["type"]=="cylx":
                base=add_cylx(f, base, S["cx"],S["cy"],S["cz"],S["R"],S["L"], S["e_n"], S["name"])
            else:
                base=add_cyly(f, base, S["cx"],S["cy"],S["cz"],S["R"],S["L"], S["e_n"], S["name"])

# ------------- Geometrie & materiale -------------

def make_geometry():
    W = CFG
    pieces = []
    pieces.append(dict(type='box', center=W.CUB1, size=W.CUBE, **W.MAT_CUBE_L, name='CUBE_L'))
    pieces.append(dict(type='box', center=W.CUB2, size=W.CUBE, **W.MAT_CUBE_R, name='CUBE_R'))

    x1 = W.CUB1[0] + W.CUBE[0]/2
    x2 = W.CUB2[0] - W.CUBE[0]/2
    dist = x2 - x1
    seg  = dist / W.FUNNEL_SEG
    segi  = [seg+20, seg-10, seg-10, seg-10, seg-10, seg+20]

    for i, R in enumerate(W.FUNNEL_RAD):
        cx = x1 + (i+0.5)*seg
        L  = segi[i] + 2*W.FUNNEL_PAD + 2*W.SEAL_OVERLAP
        pieces.append(dict(type='cylx', cx=cx, cy=W.FUNNEL_Y, cz=W.FUNNEL_Z, R=R, L=L,
                           **W.MAT_FUN[i], name=f'FUN_{i+1}'))

    yb1 = W.CUB1[1] - W.CUBE[1]/2
    yb2 = W.CUB2[1] - W.CUBE[1]/2
    y_top1 = yb1 + W.SEAL_OVERLAP
    y_bot1 = W.LOOP_Y - W.SEAL_OVERLAP
    L1 = abs(y_top1 - y_bot1); cy1 = 0.5*(y_top1+y_bot1)

    # y_top2 = yb2 + W.SEAL_OVERLAP
    # y_bot2 = W.LOOP_Y - W.SEAL_OVERLAP
    # L2 = abs(y_top2 - y_bot2); cy2 = 0.5*(y_top2+y_bot2)

    # pieces.append(dict(type='cyly', cx=W.CUB1[0], cy=cy1, cz=0.0, R=W.VERT_R, L=L1,
    #                    **W.MAT_VERT, name='VERT_L'))
    # pieces.append(dict(type='cyly', cx=W.CUB2[0], cy=cy2, cz=0.0, R=W.VERT_R, L=L2,
    #                    **W.MAT_VERT, name='VERT_R'))

    xR0 = x1 - W.RET_EXTRA
    xR1 = x2 + W.RET_EXTRA
    Lret= xR1 - xR0
    cxr = 0.5*(xR0+xR1)
    pieces.append(dict(type='cylx', cx=cxr, cy=W.LOOP_Y, cz=0.0, R=W.RET_R, L=Lret,
                       **W.MAT_RET, name='RET'))
    return pieces

def piece_names(pieces):
    return [p['name'] for p in pieces]

# ---------- inside tests (CPU+GPU friendly) ----------

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

# ---------- GPU vectorizat: pack geometrie + inside/locate ----------

def pack_geometry_xp(pieces, xp):
    K = len(pieces)
    types = xp.asarray([PIECE_TYPES[p['type']] for p in pieces], dtype=xp.int32)
    e_n   = xp.asarray([p['e_n']   for p in pieces], dtype=xp.float32)
    bt    = xp.asarray([p['beta_t']for p in pieces], dtype=xp.float32)
    # box
    box_c = xp.zeros((K,3), dtype=xp.float32)
    box_s = xp.zeros((K,3), dtype=xp.float32)
    # cylx & cyly
    cyl_c = xp.zeros((K,3), dtype=xp.float32)
    cyl_R = xp.zeros((K,),  dtype=xp.float32)
    cyl_L = xp.zeros((K,),  dtype=xp.float32)
    for i,S in enumerate(pieces):
        if S['type']=='box':
            box_c[i] = xp.asarray(S['center'], dtype=xp.float32)
            box_s[i] = xp.asarray(S['size'],   dtype=xp.float32)
        else:
            cyl_c[i] = xp.asarray([S['cx'],S['cy'],S['cz']], dtype=xp.float32)
            cyl_R[i] = float(S['R']); cyl_L[i] = float(S['L'])
    return dict(types=types, e_n=e_n, bt=bt, box_c=box_c, box_s=box_s, cyl_c=cyl_c, cyl_R=cyl_R, cyl_L=cyl_L)


def inside_masks_xp(P, G, xp):
    K = G['types'].shape[0]
    N = P.shape[0]
    masks = xp.zeros((K,N), dtype=xp.bool_)
    # box
    kb = xp.where(G['types']==0)[0]
    if kb.size>0:
        C = G['box_c'][kb][:,None,:]  # (Kb,1,3)
        S = G['box_s'][kb][:,None,:]
        Pexp = P[None,:,:]
        m = (xp.abs(Pexp - C) <= (S*0.5)).all(axis=2)
        masks[kb] = m
    # cylx
    kx = xp.where(G['types']==1)[0]
    if kx.size>0:
        C = G['cyl_c'][kx][:,None,:]
        R = G['cyl_R'][kx][:,None]
        L = G['cyl_L'][kx][:,None]
        Pexp = P[None,:,:]
        cond_x = xp.abs(Pexp[...,0]-C[...,0]) <= (L*0.5)
        dr2 = (Pexp[...,1]-C[...,1])**2 + (Pexp[...,2]-C[...,2])**2
        cond_r = dr2 <= (R**2)
        masks[kx] = cond_x & cond_r
    # cyly
    ky = xp.where(G['types']==2)[0]
    if ky.size>0:
        C = G['cyl_c'][ky][:,None,:]
        R = G['cyl_R'][ky][:,None]
        L = G['cyl_L'][ky][:,None]
        Pexp = P[None,:,:]
        cond_y = xp.abs(Pexp[...,1]-C[...,1]) <= (L*0.5)
        dr2 = (Pexp[...,0]-C[...,0])**2 + (Pexp[...,2]-C[...,2])**2
        cond_r = dr2 <= (R**2)
        masks[ky] = cond_y & cond_r
    return masks


def locate_points_in_pieces_xp(P, G, xp):
    m = inside_masks_xp(P, G, xp)  # (K,N)
    K,N = m.shape
    idxs = xp.arange(K, dtype=xp.int32)[:,None]
    scores = xp.where(m, (K - idxs), 0)
    arg = scores.argmax(axis=0)
    any_ = m.any(axis=0)
    out = xp.where(any_, arg, -1)
    return out.astype(xp.int32)

# ---------- distanța la perete (CPU, pt. histograme) ----------

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

# ---------- coliziune robustă pe CPU (fallback) ----------

def collide_one_piece(p_old, p_new, v, S):
    e_n, beta_t = S['e_n'], S['beta_t']
    if S['type']=='box':
        cx,cy,cz = S['center']; sx,sy,sz = S['size']
        x,y,z = p_new
        px = (x - cx) - np.clip(x - cx, -sx*0.5, sx*0.5)
        py = (y - cy) - np.clip(y - cy, -sy*0.5, sy*0.5)
        pz = (z - cz) - np.clip(z - cz, -sz*0.5, sz*0.5)
        pen = np.array([abs(px), abs(py), abs(pz)], dtype=np.float32)
        if (pen<=1e-12).all():
            return p_new, v, False
        axis = int(np.argmax(pen))
        n = np.zeros(3, dtype=np.float32)
        if axis==0:
            x = cx + math.copysign(sx*0.5, x-cx); n[0] = math.copysign(1.0, x-cx)
        elif axis==1:
            y = cy + math.copysign(sy*0.5, y-cy); n[1] = math.copysign(1.0, y-cy)
        else:
            z = cz + math.copysign(sz*0.5, z-cz); n[2] = math.copysign(1.0, z-cz)
        vn = np.dot(v,n)*n
        vt = v - vn
        v_ref = -e_n*vn + beta_t*vt
        return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True

    if S['type']=='cylx':
        cx,cy,cz,R,L = S['cx'],S['cy'],S['cz'],S['R'],S['L']
        x,y,z = p_new
        if abs(x-cx) > L*0.5:
            nx = +1.0 if x>cx else -1.0
            n = np.array([nx,0.0,0.0],dtype=np.float32)
            x = cx + math.copysign(L*0.5, x-cx)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
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
        if abs(y-cy) > L*0.5:
            ny = +1.0 if y>cy else -1.0
            n = np.array([0.0,ny,0.0],dtype=np.float32)
            y = cy + math.copysign(L*0.5, y-cy)
            vn = np.dot(v,n)*n; vt = v-vn
            v_ref = -e_n*vn + beta_t*vt
            return np.array([x,y,z],dtype=np.float32), v_ref.astype(np.float32), True
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

# ---------- GPU: coliziuni vectorizate ----------

def collide_one_piece_gpu(p_old, p_new, v, idx, G, xp):
    N = p_new.shape[0]
    if N==0:
        return p_new, v, xp.zeros((0,), dtype=xp.bool_)
    p_corr = p_new.copy(); v_corr = v.copy(); hit = xp.zeros((N,), dtype=xp.bool_)
    t = G['types'][idx]
    e = G['e_n'][idx]; bt = G['bt'][idx]

    # BOX
    mb = (t==0)
    if mb.any():
        I = xp.where(mb)[0]
        C = G['box_c'][idx[I]]; S=G['box_s'][idx[I]]
        X = p_new[I]
        d = X - C
        px = xp.abs(d[:,0]) - S[:,0]*0.5
        py = xp.abs(d[:,1]) - S[:,1]*0.5
        pz = xp.abs(d[:,2]) - S[:,2]*0.5
        pen = xp.stack([xp.maximum(px,0), xp.maximum(py,0), xp.maximum(pz,0)],axis=1)
        hitI = (pen.max(axis=1)>0)
        if hitI.any():
            J = I[hitI]
            dJ = p_new[J]-G['box_c'][idx[J]]; SJ=G['box_s'][idx[J]]
            pxJ = xp.abs(dJ[:,0]) - SJ[:,0]*0.5
            pyJ = xp.abs(dJ[:,1]) - SJ[:,1]*0.5
            pzJ = xp.abs(dJ[:,2]) - SJ[:,2]*0.5
            penJ = xp.stack([xp.maximum(pxJ,0), xp.maximum(pyJ,0), xp.maximum(pzJ,0)],axis=1)
            ax = penJ.argmax(axis=1)
            n = xp.zeros((J.size,3), dtype=xp.float32)
            XJ = p_corr[J]
            jx = xp.where(ax==0)[0]
            if jx.size>0:
                jj = J[jx]
                sign = xp.sign(dJ[jx,0])
                XJ[jx,0] = G['box_c'][idx[jj],0] + sign*SJ[jx,0]*0.5
                n[jx,0] = sign
            jy = xp.where(ax==1)[0]
            if jy.size>0:
                jj = J[jy]
                sign = xp.sign(dJ[jy,1])
                XJ[jy,1] = G['box_c'][idx[jj],1] + sign*SJ[jy,1]*0.5
                n[jy,1] = sign
            jz = xp.where(ax==2)[0]
            if jz.size>0:
                jj = J[jz]
                sign = xp.sign(dJ[jz,2])
                XJ[jz,2] = G['box_c'][idx[jz],2] + sign*SJ[jz,2]*0.5
                n[jz,2] = sign
            p_corr[J] = XJ
            vn = (v[J]*n).sum(axis=1, keepdims=True)*n
            vt = v[J]-vn
            en = e[J][:,None]; btJ = bt[J][:,None]
            v_corr[J] = -en*vn + btJ*vt
            hit[J] = True

    # CYLX
    mx = (t==1)
    if mx.any():
        I = xp.where(mx)[0]
        C = G['cyl_c'][idx[I]]; R = G['cyl_R'][idx[I]]; L = G['cyl_L'][idx[I]]
        X = p_new[I]
        over = xp.abs(X[:,0]-C[:,0]) > (L*0.5)
        if over.any():
            J = I[over]
            XJ = p_corr[J]
            sign = xp.sign(XJ[:,0]-G['cyl_c'][idx[J],0])
            XJ[:,0] = G['cyl_c'][idx[J],0] + sign*(G['cyl_L'][idx[J]]*0.5)
            n = xp.zeros_like(XJ); n[:,0] = sign
            vn = (v[J]*n).sum(axis=1, keepdims=True)*n
            vt = v[J]-vn
            en = e[J][:,None]; btJ = bt[J][:,None]
            v_corr[J] = -en*vn + btJ*vt
            p_corr[J] = XJ
            hit[J] = True
        I2 = xp.where(mx & (~hit))[0]
        if I2.size>0:
            C2 = G['cyl_c'][idx[I2]]; R2 = G['cyl_R'][idx[I2]]
            X2 = p_new[I2]
            ry = X2[:,1]-C2[:,1]; rz = X2[:,2]-C2[:,2]
            r = xp.sqrt(ry*ry+rz*rz)
            out = r>R2
            if out.any():
                J = I2[out]
                C3 = G['cyl_c'][idx[J]]; R3=G['cyl_R'][idx[J]]
                XJ = p_corr[J]
                ry = XJ[:,1]-C3[:,1]; rz = XJ[:,2]-C3[:,2]
                r = xp.sqrt(ry*ry+rz*rz)
                n = xp.stack([xp.zeros(J.size,dtype=xp.float32), ry/r, rz/r], axis=1)
                XJ[:,1] = C3[:,1] + R3*(ry/r)
                XJ[:,2] = C3[:,2] + R3*(rz/r)
                vn = (v[J]*n).sum(axis=1, keepdims=True)*n
                vt = v[J]-vn
                en = e[J][:,None]; btJ = bt[J][:,None]
                v_corr[J] = -en*vn + btJ*vt
                p_corr[J] = XJ
                hit[J] = True

    # CYLY
    my = (t==2)
    if my.any():
        I = xp.where(my)[0]
        C = G['cyl_c'][idx[I]]; R = G['cyl_R'][idx[I]]; L = G['cyl_L'][idx[I]]
        X = p_new[I]
        over = xp.abs(X[:,1]-C[:,1]) > (L*0.5)
        if over.any():
            J = I[over]
            XJ = p_corr[J]
            sign = xp.sign(XJ[:,1]-G['cyl_c'][idx[J],1])
            XJ[:,1] = G['cyl_c'][idx[J],1] + sign*(G['cyl_L'][idx[J]]*0.5)
            n = xp.zeros_like(XJ); n[:,1] = sign
            vn = (v[J]*n).sum(axis=1, keepdims=True)*n
            vt = v[J]-vn
            en = e[J][:,None]; btJ = bt[J][:,None]
            v_corr[J] = -en*vn + btJ*vt
            p_corr[J] = XJ
            hit[J] = True
        I2 = xp.where(my & (~hit))[0]
        if I2.size>0:
            C2 = G['cyl_c'][idx[I2]]; R2=G['cyl_R'][idx[I2]]
            X2 = p_new[I2]
            rx = X2[:,0]-C2[:,0]; rz = X2[:,2]-C2[:,2]
            r = xp.sqrt(rx*rx+rz*rz)
            out = r>R2
            if out.any():
                J = I2[out]
                C3 = G['cyl_c'][idx[J]]; R3=G['cyl_R'][idx[J]]
                XJ = p_corr[J]
                rx = XJ[:,0]-C3[:,0]; rz = XJ[:,2]-C3[:,2]
                r = xp.sqrt(rx*rx+rz*rz)
                n = xp.stack([rx/r, xp.zeros(J.size,dtype=xp.float32), rz/r], axis=1)
                XJ[:,0] = C3[:,0] + R3*(rx/r)
                XJ[:,2] = C3[:,2] + R3*(rz/r)
                vn = (v[J]*n).sum(axis=1, keepdims=True)*n
                vt = v[J]-vn
                en = e[J][:,None]; btJ = bt[J][:,None]
                v_corr[J] = -en*vn + btJ*vt
                p_corr[J] = XJ
                hit[J] = True
    return p_corr, v_corr, hit


def reflect_gpu(p_old, p_new, v, prev_idx, G, xp):
    in_union = inside_masks_xp(p_new, G, xp).any(axis=0)
    new_idx = locate_points_in_pieces_xp(p_new, G, xp)
    crossed = (new_idx != prev_idx)
    need = (~in_union)
    if need.any():
        I = xp.where(need)[0]
        p_corr, v_corr, hit = collide_one_piece_gpu(p_old[I], p_new[I], v[I], prev_idx[I], G, xp)
        sel = I[hit]
        if sel.size>0:
            p_new[sel] = p_corr[hit]
            v[sel]     = v_corr[hit]
            new_idx[sel] = prev_idx[sel]
            crossed[sel] = False
        sel2 = I[~hit]
        if sel2.size>0:
            p_new[sel2] = p_old[sel2]
            v[sel2] *= -0.5
            new_idx[sel2] = prev_idx[sel2]
            crossed[sel2] = False
    return p_new, v, new_idx, crossed

# ---------- sampling inițial ----------

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
        f.write(f"{N}frame")
        for i in range(N):
            x,y,z = pos[i]
            f.write(f"He {x:.6f} {y:.6f} {z:.6f}")

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
    ap.add_argument("--gpu-collide", action="store_true", help="rulează și coliziunile vectorizat pe GPU")
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--gsd-subset", type=int, default=CFG.GSD_SUBSET)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--no-io", action="store_true")
    ap.add_argument("--adapt-every", type=int, default=1, help="recalculează sub-stepping la fiecare k pași")
    ap.add_argument("--quiet", action="store_true")

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

    t0 = time.time()
    last_print_len = 0

    def log_progress(step, total, extra=""):
        nonlocal last_print_len
        if args.quiet: return
        elapsed = time.time() - t0
        rate = step / elapsed if elapsed > 0 else 0.0
        eta  = (total - step) / rate if (rate > 0 and step > 0) else float("inf")
        msg = f"[{step}/{total}] {rate:7.1f} steps/s  elapsed {human_time(elapsed)}  ETA {human_time(eta)}"
        if extra: msg += f"  | {extra}"
        pad = " " * max(0, last_print_len - len(msg))
        sys.stdout.write("" + msg + pad)
        sys.stdout.flush()
        last_print_len = len(msg)
        if step >= total:
            sys.stdout.write(""); 
            sys.stdout.flush()

    # backend numeric
    xp = np
    if args.gpu >= 0 and cp is not None:
        try:
            cp.cuda.Device(args.gpu).use(); xp = cp
            print(f"[GPU] CuPy on device {args.gpu}")
        except Exception as e:
            print(f"[WARN] CuPy device {args.gpu} unavailable ({e}); falling back to CPU.")
            xp = np
    elif args.gpu >= 0 and cp is None:
        print("[WARN] CuPy not installed; running on CPU.")

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    xyz_path = os.path.join(CFG.OUT_DIR, CFG.XYZ_NAME)
    if os.path.exists(xyz_path): os.remove(xyz_path)

    pieces = make_geometry(); names = piece_names(pieces)

    # OBJ preview
    write_obj(os.path.join(CFG.OUT_DIR,"geom.obj"),
              os.path.join(CFG.OUT_DIR,"geom.mtl"),
              pieces, alpha=0.45)
    
    # init
    pos0 = sample_uniform_in_union(args.n, pieces, seed=args.seed)
    vel0 = np.zeros_like(pos0, dtype=np.float32)

    # RNG pentru Langevin
    rng = np.random.default_rng(args.seed)
    gpu_rng = None
    if xp is cp:
        try:
            cp.random.seed(args.seed)
            gpu_rng = cp.random
        except Exception:
            gpu_rng = None

    # piece index (CPU/GPU)
    Gx = None
    if xp is cp:
        Gx = pack_geometry_xp(pieces, cp)
        piece_idx = locate_points_in_pieces_xp(cp.asarray(pos0), Gx, cp).get()
    else:
        piece_idx = locate_points_in_pieces(pos0, pieces)

    # tracking
    track_k = min(CFG.TRACK_K, args.n)
    track_ids = np.arange(track_k, dtype=np.int32)

    # mută pe GPU dacă e cazul
    pos = xp.asarray(pos0)
    vel = xp.asarray(vel0)

    m  = float(CFG.MASS)
    dt = float(args.dt)
    gamma = float(args.gamma)
    kt = float(args.kt)

    # Corecție zgomot: dv = -(γ/m)v dt + [sqrt(2γkT)/m] dW
    noise_sigma = math.sqrt(2.0*gamma*kt) * math.sqrt(dt) / m

    # GSD
    gsd_file = None
    if HAVE_GSD and not args.no_io:
        gsd_path = os.path.join(CFG.OUT_DIR, CFG.GSD_NAME)
        gsd_file = gsd.hoomd.open(name=gsd_path, mode='w')
        print(f"[GSD] writing to {gsd_path}")
    elif not HAVE_GSD and not args.no_io:
        print("[WARN] GSD not available; only XYZ/CSV will be written.")

    # CSV loguri
    trans_path  = os.path.join(CFG.OUT_DIR, "transitions.csv")
    counts_path = os.path.join(CFG.OUT_DIR, "piece_counts.csv")
    if not args.no_io:
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
        v = to_cpu(vel_xp)
        if v.size == 0: return 0.0
        v = np.where(np.isfinite(v), v, 0.0)
        speeds = np.linalg.norm(v, axis=1)
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
        if not args.no_io:
            with open(os.path.join(CFG.OUT_DIR, f"wall_hist_step{step:06d}.csv"),"w",newline="") as f:
                w=csv.writer(f); w.writerows(rows)

        if CFG.SAVE_WALL_DIST_FOR_TRACK and track_ids.size>0 and not args.no_io:
            for pid in track_ids:
                kk = idx_cpu[pid]
                nm = names[kk] if kk>=0 else "OUT"
                dist = 0.0 if kk<0 else wall_distance_one(pos_cpu[pid], pieces[kk])
                track_rows.append([int(pid), nm, f"{dist:.6f}"])
            with open(os.path.join(CFG.OUT_DIR, f"wall_dist_track_step{step:06d}.csv"),"w",newline="") as f:
                w=csv.writer(f); w.writerows(track_rows)

    # scrie frame inițial
    pos_cpu = to_cpu(pos); vel_cpu = to_cpu(vel)
    if not args.no_io:
        append_xyz(xyz_path, pos_cpu)
        write_gsd_frame(0, pos_cpu, vel_cpu)
        idx_cpu = write_piece_map(0, pos_cpu)
        with open(counts_path,"a",newline="") as f:
            w=csv.writer(f)
            counts = [(idx_cpu==k).sum() for k in range(len(pieces))]
            w.writerow([0]+counts)
        wall_histograms(0, pos_cpu, idx_cpu)

    # --------------- integrare ---------------
    steps = args.steps
    write_every = max(1, args.write_every)

    def compute_nsub(step, pos, vel):
        if CFG.MAX_SUBSTEPS<=1: return 1
        if args.adapt_every>1 and (step % args.adapt_every)!=0:
            return compute_nsub._last if hasattr(compute_nsub, '_last') else 1
        pos_cpu = to_cpu(pos)
        max_speed = max_speed_cpu(vel)
        step_len  = max_speed * dt + 1e-12
        sample_ids = track_ids if track_ids.size>0 else np.arange(min(500, pos_cpu.shape[0]))
        dist_samp = []
        for pid in sample_ids:
            k = piece_idx[pid] if isinstance(piece_idx, np.ndarray) else int(piece_idx[pid])
            d = 1.0
            if k>=0:
                d = wall_distance_one(pos_cpu[pid], pieces[k])
            dist_samp.append(d)
        dmed = np.median(dist_samp) if len(dist_samp)>0 else 1.0
        nsub = 1
        if dmed>1e-6:
            est = step_len / (CFG.COLLISION_CFL * dmed)
            nsub = int(min(CFG.MAX_SUBSTEPS, max(1, math.ceil(est))))
        compute_nsub._last = nsub
        return nsub

    start = time.time()
    for step in range(1, steps+1):
        # Langevin (Euler–Maruyama) pe v
        if gpu_rng is not None:
            xi = gpu_rng.standard_normal(size=pos.shape, dtype=cp.float32)
        else:
            xi = np.asarray(rng.standard_normal(size=pos.shape), dtype=np.float32)
            if xp is cp:
                xi = cp.asarray(xi)
        vel = vel + (-(gamma/m))*vel*dt + noise_sigma*xi

        nsub = compute_nsub(step, pos, vel)
        sub_dt = dt/float(nsub)
        for _ in range(nsub):
            p_old = pos
            pos = pos + vel*sub_dt
            prev_idx = piece_idx.copy()
            if args.gpu_collide and xp is cp:
                pos, vel, new_idx_xp, crossed_xp = reflect_gpu(p_old, pos, vel, cp.asarray(prev_idx), Gx, cp)
                piece_idx = new_idx_xp.get()
                if CFG.LOG_TRANS_EVERY_STEP and not args.no_io:
                    crossed = crossed_xp.get()
                    ids = np.where(crossed & (piece_idx!=prev_idx))[0]
                    if ids.size>0:
                        with open(trans_path,"a",newline="") as f:
                            w=csv.writer(f)
                            for i in ids:
                                fr = names[prev_idx[i]] if prev_idx[i]>=0 else "OUT"
                                to = names[piece_idx[i]] if piece_idx[i]>=0 else "OUT"
                                w.writerow([step, int(i), fr, to])
            else:
                # CPU robust path
                p_old_cpu = to_cpu(p_old); p_new_cpu = to_cpu(pos); v_cpu = to_cpu(vel)
                prev_idx = piece_idx.copy()
                for i in range(p_new_cpu.shape[0]):
                    p_new_cpu[i], v_cpu[i], piece_idx[i], crossed = reflect_cpu(
                        p_old_cpu[i], p_new_cpu[i], v_cpu[i], prev_idx[i]
                    )
                    if CFG.LOG_TRANS_EVERY_STEP and crossed and (prev_idx[i] != piece_idx[i]) and (not args.no_io):
                        with open(trans_path,"a",newline="") as f:
                            w=csv.writer(f)
                            fr = names[prev_idx[i]] if prev_idx[i]>=0 else "OUT"
                            to = names[piece_idx[i]] if piece_idx[i]>=0 else "OUT"
                            w.writerow([step, i, fr, to])
                pos = xp.asarray(p_new_cpu) if xp is cp else p_new_cpu
                vel = xp.asarray(v_cpu)     if xp is cp else v_cpu

        # OUTPUT
        if (step % write_every)==0 and not args.no_io:
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
            log_progress(step, args.steps, extra=("write" if (not args.no_io and (step % write_every)==0) else ""))

    if gsd_file is not None:
        gsd_file.close()
    print(f"[OK] done in {time.time()-start:.1f}s.")

if __name__ == "__main__":
    main()
