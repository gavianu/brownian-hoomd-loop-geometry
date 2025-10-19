# sim/analysis_post.py
# Analiză post-procesare pentru simularea Langevin cu geometrii analitice.
# Necesită: numpy, pandas, matplotlib, (opțional) gsd
# Citeste dintr-un director de output (ex: sim/out_langevin_2) si scrie rezultate in subfolder `analysis`.

import os, re, glob, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- optional GSD ----------
try:
    import gsd, gsd.hoomd
    HAVE_GSD = True
except Exception:
    HAVE_GSD = False

# ---------- mic utilitar ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def human_int(n): return f"{n:,}".replace(",","_")

# ---------- încărcare traiectorii ----------
def load_gsd_positions(gsd_path, step_stride=1, max_frames=None):
    pos_frames, steps = [], []
    with gsd.hoomd.open(gsd_path, mode='r') as f:
        n = len(f)
        idxs = list(range(0, n, step_stride))
        if max_frames is not None:
            idxs = idxs[:max_frames]
        for i in idxs:
            fr = f[i]
            pos_frames.append(fr.particles.position.astype(np.float32))
            steps.append(int(fr.configuration.step))
    return steps, pos_frames

def load_xyz_positions(xyz_path, frame_stride=1, max_frames=None):
    # Format:
    # N
    # frame
    # He x y z
    # ...
    steps, pos_frames = [], []
    with open(xyz_path, "r") as f:
        data = f.read().strip().splitlines()
    i=0; frame_id=0
    while i < len(data):
        N = int(data[i].strip()); i+=1
        if i>=len(data): break
        _hdr = data[i]; i+=1
        rows = data[i:i+N]; i+=N
        if frame_id % frame_stride == 0:
            P = np.zeros((N,3), dtype=np.float32)
            for j, line in enumerate(rows):
                # He x y z
                _, x, y, z = line.split()
                P[j,0]=float(x); P[j,1]=float(y); P[j,2]=float(z)
            pos_frames.append(P)
            steps.append(frame_id)   # dacă vrei echivalență cu GSD (pasul de integrare), suprascrie la rulare
        frame_id += 1
        if max_frames is not None and len(pos_frames)>=max_frames:
            break
    return steps, pos_frames

# ---------- MSD ----------
def msd_time_averaged(pos_frames, dt):
    """TAMSD (averaged over time origins) – robust la ‘drift’ global."""
    T = len(pos_frames)
    N = pos_frames[0].shape[0]
    max_tau = T//2
    taus = np.arange(1, max_tau+1, dtype=int)
    tamsd = np.zeros_like(taus, dtype=np.float64)
    for k, tau in enumerate(taus):
        acc = 0.0; cnt = 0
        for t0 in range(0, T - tau):
            d = pos_frames[t0+tau] - pos_frames[t0]
            acc += (d*d).sum(axis=1).mean()
            cnt += 1
        if cnt>0:
            tamsd[k] = acc / cnt
        else:
            tamsd[k] = np.nan
    return taus*dt, tamsd

def msd_from_origin(pos_frames, dt):
    """MSD față de frame 0 – sensibil la drift global."""
    ref = pos_frames[0]
    msd = []
    for P in pos_frames:
        d = P - ref
        msd.append( (d*d).sum(axis=1).mean() )
    t = np.arange(len(pos_frames))*dt
    return t, np.asarray(msd)

# ---------- geometrii pentru volum (pt. densități normalizate) ----------
def pieces_geometry_from_counts_header(header_names):
    """Recuperează numele pieselor din header-ul lui piece_counts.csv (după 'step,').
       Volumul îl obținem din fișier config_volumes.csv dacă există; altfel lăsăm 1.0."""
    names = [h for h in header_names if h!="step"]
    vols = {n:1.0 for n in names}
    return names, vols

def try_load_piece_volumes(base_dir):
    p = os.path.join(base_dir, "config_volumes.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return dict(zip(df["name"], df["volume"]))
    return None

# ---------- tranziții ----------
def transitions_matrix(transitions_csv, piece_names, dt_write):
    """Construiește matricea de tranziție pe unitate de timp din log-ul transitions.csv."""
    if not os.path.exists(transitions_csv):
        return None, None, None
    df = pd.read_csv(transitions_csv)
    # filtrăm OUT -> ignorăm ieșiri
    df = df[(df["from"]!="OUT") & (df["to"]!="OUT")]
    idx = {n:i for i,n in enumerate(piece_names)}
    M = np.zeros((len(piece_names), len(piece_names)), dtype=np.float64)
    for _,row in df.iterrows():
        i = idx.get(row["from"], None)
        j = idx.get(row["to"], None)
        if i is not None and j is not None:
            M[i,j] += 1
    # normalizare în rate/s: aproximăm pasul temporal = dt_write (pasul dintre frames scrise)
    rates = M / (dt_write * max(1, df["particle_id"].nunique()))
    return M, rates, df

def stationary_from_rates(R):
    """Distribuție staționară din matrice de rate (aprox. discret): left eigenvector al lui R normat."""
    if R is None:
        return None
    # row-stochastic approx – normalizăm pe linii
    A = R.copy()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums==0]=1.0
    P = A / row_sums
    # vector propriu pentru autovalorea 1 (stânga)
    w, v = np.linalg.eig(P.T)
    k = np.argmin(np.abs(w-1.0))
    pi = np.real(v[:,k])
    pi = np.maximum(pi, 0.0)
    if pi.sum()>0: pi /= pi.sum()
    return pi

# ---------- prim-trecere ----------
def first_passage_times(transitions_df, start_piece, target_piece, names, dt_write):
    if transitions_df is None:
        return None
    df = transitions_df.copy()
    df = df[df["from"].isin([start_piece])]  # intrări "pleacă din start_piece"
    # pentru fiecare particulă urmărită: când atinge target_piece prima oară?
    fpt = []
    for pid, g in df.groupby("particle_id"):
        g = g.sort_values("step")
        # căutăm primul row unde 'to'==target
        hit = g[g["to"]==target_piece]
        if len(hit)>0:
            t0 = g["step"].min()
            t1 = hit["step"].iloc[0]
            fpt.append( (t1 - t0)*dt_write )
    if len(fpt)==0:
        return None
    return np.asarray(fpt, dtype=np.float64)

# ---------- wall hist ----------
def load_wall_hist_all(base_dir):
    """Citește toate wall_hist_step*.csv și uniformizează schema.
       Acceptă:
       - format A: coloana 'dist' și 'count'
       - format B: 'bin_left','bin_right','count' (=> dist = (left+right)/2)
    """
    files = sorted(glob.glob(os.path.join(base_dir, "wall_hist_step*.csv")))
    if not files:
        return None
    recs = []
    for fp in files:
        m = re.search(r"step(\d+)", os.path.basename(fp))
        step = int(m.group(1)) if m else None
        df = pd.read_csv(fp)

        # detectăm schema și o aducem la ['step','dist','count',...]
        cols = {c.lower(): c for c in df.columns}  # map case-insensitive -> real name
        if "dist" in cols and "count" in cols:
            dist_col  = cols["dist"]
            count_col = cols["count"]
            out = pd.DataFrame({
                "step": step,
                "dist": df[dist_col].values.astype(float),
                "count": df[count_col].values.astype(float)
            })
            # păstrează și metadate utile dacă există (ex. piece/axis)
            for key in ["piece","axis","name","part"]:
                if key in cols:
                    out[key] = df[cols[key]]
            recs.append(out)

        elif "bin_left" in cols and "bin_right" in cols and "count" in cols:
            left  = df[cols["bin_left"]].values.astype(float)
            right = df[cols["bin_right"]].values.astype(float)
            cnt   = df[cols["count"]].values.astype(float)
            out = pd.DataFrame({
                "step": step,
                "dist": 0.5*(left+right),
                "count": cnt
            })
            for key in ["piece","axis","name","part"]:
                if key in cols:
                    out[key] = df[cols[key]]
            recs.append(out)

        else:
            # format necunoscut: log și sari
            print(f"[WARN] wall_hist schema necunoscută în {os.path.basename(fp)}: {df.columns.tolist()}")
            continue

    if not recs:
        return None
    W = pd.concat(recs, ignore_index=True)
    return W


# ---------- plot helpers ----------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="sim/out_langevin_1", help="directorul cu rezultatele unei rulari")
    ap.add_argument("--dt", type=float, default=0.001, help="pasul de integrare folosit la simulare")
    ap.add_argument("--write-every", type=int, default=1000, help="pasul de scriere cadre")
    ap.add_argument("--use-xyz", action="store_true", help="obliga folosirea run.xyz in loc de run.gsd")
    ap.add_argument("--frame-stride", type=int, default=1, help="ia din 1 in 1 frame (sub-esantionare la citire)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    base = args.outdir
    ana_dir = os.path.join(base, "analysis")
    ensure_dir(ana_dir)

    # ---- încărcare poziții
    gsd_path = os.path.join(base, "run.gsd")
    xyz_path = os.path.join(base, "run.xyz")
    steps, frames = None, None

    if (not args.use_xyz) and HAVE_GSD and os.path.exists(gsd_path):
        print(f"[READ] GSD: {gsd_path}")
        steps, frames = load_gsd_positions(gsd_path, step_stride=args.frame_stride, max_frames=args.max_frames)
        dt_frame = args.dt * args.write_every * args.frame_stride
    elif os.path.exists(xyz_path):
        print(f"[READ] XYZ: {xyz_path}")
        steps, frames = load_xyz_positions(xyz_path, frame_stride=args.frame_stride, max_frames=args.max_frames)
        # la XYZ nu știm dt real; dacă vrei corect, dă --dt și --write-every ca la simulare:
        dt_frame = args.dt * args.write_every * args.frame_stride
    else:
        print("[ERR] Nu am găsit nici run.gsd, nici run.xyz.")
        return

    N = frames[0].shape[0]
    print(f"[INFO] frames: {len(frames)} | particles: {human_int(N)} | dt_frame={dt_frame:g}")

    # ---- MSD
    t_tamsd, tamsd = msd_time_averaged(frames, dt_frame)
    t_msd0, msd0 = msd_from_origin(frames, dt_frame)
    pd.DataFrame({"t":t_tamsd, "TAMSD":tamsd}).to_csv(os.path.join(ana_dir,"tamsd.csv"), index=False)
    pd.DataFrame({"t":t_msd0, "MSD0":msd0}).to_csv(os.path.join(ana_dir,"msd0.csv"), index=False)

    plt.figure()
    plt.loglog(t_tamsd, tamsd+1e-16, label="TAMSD")
    plt.loglog(t_msd0,  msd0+1e-16,  label="MSD (origin)")
    plt.xlabel("t [s] (echiv.)"); plt.ylabel("<Δr^2>")
    plt.legend(); savefig(os.path.join(ana_dir,"msd.png"))

    # ---- densități pe piese (normalizate cu volum)
    counts_csv = os.path.join(base, "piece_counts.csv")
    if os.path.exists(counts_csv):
        C = pd.read_csv(counts_csv)
        piece_names = [c for c in C.columns if c!="step"]
        vols = try_load_piece_volumes(base)
        if vols is None:
            # volum necunoscut => 1
            piece_names, vols_map = pieces_geometry_from_counts_header(C.columns.tolist())
            vols = vols or vols_map
        # densitate = count / volum; și fracție = count / Ntotal
        C["Ntot"] = C[piece_names].sum(axis=1)
        for nm in piece_names:
            C[nm+"_rho"] = C[nm] / float(vols.get(nm, 1.0))
            C[nm+"_frac"] = C[nm] / C["Ntot"].replace(0, np.nan)
        C.to_csv(os.path.join(ana_dir,"piece_counts_aug.csv"), index=False)

        # plot fracții
        plt.figure()
        for nm in piece_names:
            plt.plot(C["step"], C[nm+"_frac"], label=nm)
        plt.xlabel("step (frame)"); plt.ylabel("fracție din total")
        plt.legend(ncol=2, fontsize=8)
        savefig(os.path.join(ana_dir,"pieces_fraction.png"))

    # ---- tranziții
    trans_csv = os.path.join(base, "transitions.csv")
    if os.path.exists(trans_csv) and os.path.exists(counts_csv):
        piece_names = [c for c in pd.read_csv(counts_csv, nrows=1).columns if c!="step"]
        M, R, Tdf = transitions_matrix(trans_csv, piece_names, dt_write=args.dt*args.write_every)
        if M is not None:
            pd.DataFrame(M, index=piece_names, columns=piece_names).to_csv(os.path.join(ana_dir,"transition_counts.csv"))
            pd.DataFrame(R, index=piece_names, columns=piece_names).to_csv(os.path.join(ana_dir,"transition_rates.csv"))
            pi = stationary_from_rates(R)
            if pi is not None:
                pd.DataFrame({"piece":piece_names, "pi_stationary":pi}).to_csv(os.path.join(ana_dir,"stationary_from_rates.csv"), index=False)
            # heatmap simplu
            plt.figure()
            plt.imshow(R, origin='lower', interpolation='nearest')
            plt.xticks(range(len(piece_names)), piece_names, rotation=45, ha='right')
            plt.yticks(range(len(piece_names)), piece_names)
            plt.colorbar(label="rate [1/s] (aprox)")
            savefig(os.path.join(ana_dir,"transition_rates_heatmap.png"))

            # FPT un exemplu: CUBE_L -> CUBE_R (modifică după nevoie)
            if "CUBE_L" in piece_names and "CUBE_R" in piece_names:
                fpt = first_passage_times(Tdf, "CUBE_L", "CUBE_R", piece_names, dt_write=args.dt*args.write_every)
                if fpt is not None and len(fpt)>0:
                    pd.Series(fpt).to_csv(os.path.join(ana_dir,"fpt_CUBE_L_to_CUBE_R.csv"), index=False, header=["time"])
                    plt.figure(); plt.hist(fpt, bins=30, density=True)
                    plt.xlabel("first passage time [s]"); plt.ylabel("PDF")
                    savefig(os.path.join(ana_dir,"fpt_CUBE_L_to_CUBE_R.png"))

    # ---- drift / acumulare la pereți
    W = load_wall_hist_all(base)
    if W is not None and len(W):
        # medie ponderată cu count, pe fiecare step (și opțional pe piesă dacă există)
        group_keys = ["step"] + ([c for c in ["piece","axis","name","part"] if c in W.columns][:1])
        # sum(count), sum(dist*count)
        # ---- drift / acumulare la pereți (media ponderată cu 'count')
        cols_keep = ["step","dist","count","piece","axis","name","part"]
        cols_keep = [c for c in cols_keep if c in W.columns]
        W2 = W[cols_keep].copy()

        # cheile de grupare: întotdeauna "step", opțional prima dintre meta-coloane dacă există
        extra_key = next((c for c in ["piece","axis","name","part"] if c in W2.columns), None)
        group_keys = ["step"] + ([extra_key] if extra_key else [])

        def _weighted(g: pd.DataFrame) -> pd.Series:
            w = g["count"].to_numpy(dtype=float)
            d = g["dist"].to_numpy(dtype=float)
            wsum = w.sum()
            return pd.Series({
                "count_sum": wsum,
                "dist_mean": (d*w).sum()/wsum if wsum > 0 else np.nan
            })

        agg = W2.groupby(group_keys, dropna=False).apply(_weighted).reset_index()
        agg.to_csv(os.path.join(ana_dir, "wall_mean_dist.csv"), index=False)

        # curbă globală: dacă avem piesă, mediem peste piese pentru o singură curbă
        if extra_key:
            agg_global = agg.groupby("step", dropna=False)["dist_mean"].mean().reset_index()
        else:
            agg_global = agg[["step","dist_mean"]].copy()

        plt.figure()
        plt.plot(agg_global["step"], agg_global["dist_mean"])
        plt.xlabel("step (frame)"); plt.ylabel("<dist to wall> (weighted)")
        savefig(os.path.join(ana_dir,"wall_mean_dist.png"))

        # histogramă început vs final (comparație)
        s0 = int(W2["step"].min()); s1 = int(W2["step"].max())
        cnt0 = W2.loc[W2["step"]==s0, "count"].astype(int).clip(lower=1).to_numpy()
        cnt1 = W2.loc[W2["step"]==s1, "count"].astype(int).clip(lower=1).to_numpy()
        d0   = np.repeat(W2.loc[W2["step"]==s0, "dist"].to_numpy(), cnt0)
        d1   = np.repeat(W2.loc[W2["step"]==s1, "dist"].to_numpy(), cnt1)

        if d0.size and d1.size:
            plt.figure()
            plt.hist(d0, bins=40, histtype='step', density=True, label=f"step {s0}")
            plt.hist(d1, bins=40, histtype='step', density=True, label=f"step {s1}")
            plt.xlabel("dist to wall"); plt.ylabel("PDF"); plt.legend()
            savefig(os.path.join(ana_dir,"wall_dist_compare_begin_end.png"))


        agg.to_csv(os.path.join(ana_dir, "wall_mean_dist.csv"), index=False)

        # plot „global” (agregăm încă o dată peste piesă dacă există)
        if "piece" in agg.columns:
            agg_global = agg.groupby("step")["dist_mean"].mean().reset_index()
        else:
            agg_global = agg.rename(columns={"dist_mean":"dist_mean"})[["step","dist_mean"]]

        plt.figure()
        plt.plot(agg_global["step"], agg_global["dist_mean"])
        plt.xlabel("step (frame)")
        plt.ylabel("<dist to wall> (weighted)")
        savefig(os.path.join(ana_dir,"wall_mean_dist.png"))

        # comparație distribuții început vs final (hist ne-ponderat la nivel de bin)
        s0 = int(W["step"].min()); s1 = int(W["step"].max())

        cnt0 = W.loc[W["step"]==s0, "count"].astype(int).clip(lower=1).values
        cnt1 = W.loc[W["step"]==s1, "count"].astype(int).clip(lower=1).values

        d0 = np.repeat(W.loc[W["step"]==s0, "dist"].values, cnt0)
        d1 = np.repeat(W.loc[W["step"]==s1, "dist"].values, cnt1)

        if len(d0) and len(d1):
            plt.figure()
            plt.hist(d0, bins=40, histtype='step', density=True, label=f"step {s0}")
            plt.hist(d1, bins=40, histtype='step', density=True, label=f"step {s1}")
            plt.xlabel("dist to wall"); plt.ylabel("PDF"); plt.legend()
            savefig(os.path.join(ana_dir,"wall_dist_compare_begin_end.png"))
    else:
        print("[INFO] Nu am găsit wall_hist_* cu schema recunoscută; sar analiza de pereți.")


    print(f"[OK] scris in: {ana_dir}")

if __name__ == "__main__":
    main()
