#!/usr/bin/env python3
# ness_check.py
# Verdict: EQUILIBRIUM vs NESS from transition data (counts or edge list), optional rates.
import argparse, sys, math
import numpy as np, pandas as pd

def load_counts(path_counts: str|None, path_edges: str|None):
    """
    Returns (states, C) where C is counts matrix (n x n), states is list of labels.
    Priority: matrix CSV (transition_counts.csv) with header row+col.
              else edge list CSV (transitions.csv) with columns from,to.
    """
    if path_counts:
        Cdf = pd.read_csv(path_counts)
        states = list(Cdf.columns[1:])
        C = Cdf.iloc[:,1:].to_numpy(float)
        return states, C

    if not path_edges:
        raise ValueError("Provide either --counts or --edges")

    # build from (from,to) edges
    E = pd.read_csv(path_edges)
    # normalize column names
    E.columns = [c.strip().lower() for c in E.columns]
    if not {"from","to"} <= set(E.columns):
        raise ValueError("edges CSV must have columns: from,to")
    states = sorted(list(set(E["from"].astype(str)).union(set(E["to"].astype(str)))))
    idx = {s:i for i,s in enumerate(states)}
    n = len(states)
    C = np.zeros((n,n), dtype=float)
    for f,t in zip(E["from"].astype(str), E["to"].astype(str)):
        C[idx[f], idx[t]] += 1.0
    return states, C

def stationary_from_P(P, tol=1e-15, itmax=20000):
    n = P.shape[0]
    pi = np.ones(n)/n
    for _ in range(itmax):
        new = pi @ P
        if np.linalg.norm(new - pi, 1) < tol:
            break
        pi = new
    s = pi.sum()
    return pi/s if s>0 else pi

def group_zone(name: str) -> str:
    s = str(name)
    if s in ["CUBE_L","CUBE_R","RET","VERT_L","VERT_R"]:
        return s
    if s.startswith("FUN_") or s.startswith("FUN"):
        return "FUNNELS"
    return "OTHER"

def loop_current(R, states, loop=("CUBE_L","FUNNELS","CUBE_R","RET")):
    zones = list(dict.fromkeys(loop))  # unique in order
    zidx = {z:i for i,z in enumerate(zones)}
    G = np.zeros((len(zones), len(zones)))
    for i,si in enumerate(states):
        zi = group_zone(si)
        if zi not in zidx: continue
        for j,sj in enumerate(states):
            if i==j: continue
            zj = group_zone(sj)
            if zj not in zidx: continue
            G[zidx[zi], zidx[zj]] += R[i,j]
    def along(path):
        s = 0.0
        for a,b in zip(path, path[1:]+path[:1]):
            s += G[zidx[a], zidx[b]]
        return s
    return along(list(loop)) - along(list(reversed(loop))), G, zones

def detailed_balance_metrics_from_counts(states, C):
    # filter states with zero outgoing to avoid NaNs
    row_sum = C.sum(axis=1, keepdims=True)
    keep = (row_sum[:,0] > 0)
    if keep.sum() < C.shape[0]:
        states = [s for s,k in zip(states, keep) if k]
        C = C[keep][:,keep]
        row_sum = C.sum(axis=1, keepdims=True)

    P = np.divide(C, row_sum, out=np.zeros_like(C), where=row_sum>0)
    pi = stationary_from_P(P)

    eps = 1e-15
    n = P.shape[0]
    R = np.zeros_like(P)
    sigma = 0.0
    for i in range(n):
        for j in range(n):
            if i==j: continue
            R[i,j] = pi[i]*P[i,j] - pi[j]*P[j,i]
            if P[i,j] > 0 and P[j,i] > 0:
                sigma += 0.5*R[i,j]*math.log((pi[i]*P[i,j]+eps)/(pi[j]*P[j,i]+eps))
    Rmax = float(np.max(np.abs(R))) if R.size else 0.0
    Rrms = float(np.sqrt((R**2).mean())) if R.size else 0.0
    Ntr = int(C.sum())
    return dict(states=states, P=P, pi=pi, R=R, Rmax=Rmax, Rrms=Rrms, sigma=sigma, transitions=Ntr)

def detailed_balance_metrics_from_rates(path_rates):
    Kdf = pd.read_csv(path_rates)
    states = list(Kdf.columns[1:])
    K = Kdf.iloc[:,1:].to_numpy(float)
    # enforce generator rowsum=0
    for i in range(K.shape[0]):
        K[i,i] = -np.sum(K[i,:]) + K[i,i]
    n = K.shape[0]
    A = K.T.copy(); A[-1,:] = 1.0
    b = np.zeros(n); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi,0); pi = pi/pi.sum()

    eps=1e-15
    RK = np.zeros_like(K)
    sigmaK=0.0
    for i in range(n):
        for j in range(n):
            if i==j: continue
            RK[i,j] = pi[i]*K[i,j] - pi[j]*K[j,i]
            if K[i,j] > 0 and K[j,i] > 0:
                sigmaK += 0.5*RK[i,j]*math.log((pi[i]*K[i,j]+eps)/(pi[j]*K[j,i]+eps))
    RmaxK = float(np.max(np.abs(RK)))
    RrmsK = float(np.sqrt((RK**2).mean()))
    return dict(states=states, K=K, pi=pi, RK=RK, RmaxK=RmaxK, RrmsK=RrmsK, sigmaK=sigmaK)

def main():
    ap = argparse.ArgumentParser(description="Equilibrium vs NESS checker from transitions.")
    ap.add_argument("--counts", default="transition_counts.csv", help="CSV matrix with counts (rows=from, cols=to)")
    ap.add_argument("--edges",  default=None, help="CSV edge list with columns from,to (transitions.csv)")
    ap.add_argument("--rates",  default="transition_rates.csv", help="CSV matrix with rates (optional)")
    ap.add_argument("--loop",   default="CUBE_L,FUNNELS,CUBE_R,RET", help="Loop sequence for J_loop")
    ap.add_argument("--thr_Rmax", type=float, default=1e-4, help="Threshold for max|πP-πP^T|")
    ap.add_argument("--thr_sigma", type=float, default=1e-5, help="Threshold for sigma_proxy")
    args = ap.parse_args()

    # Load counts/edges
    try:
        states, C = load_counts(args.counts, args.edges)
    except Exception as e:
        print(f"[ERROR] Loading counts/edges: {e}", file=sys.stderr)
        sys.exit(2)

    # Metrics from counts
    mc = detailed_balance_metrics_from_counts(states, C)
    loop = tuple([s.strip() for s in args.loop.split(",") if s.strip()])
    J_loop, GJ, zones = loop_current(mc["R"], mc["states"], loop=loop)

    print("=== From counts ===")
    print(f"states={len(mc['states'])}  transitions={mc['transitions']}")
    print(f"max|πP-πP^T| = {mc['Rmax']:.3e}")
    print(f"RMS|πP-πP^T| = {mc['Rrms']:.3e}")
    print(f"sigma_proxy  = {mc['sigma']:.3e}")
    print(f"J_loop[{','.join(loop)}] = {J_loop:.3e}")

    # Optional: from rates
    try:
        mr = detailed_balance_metrics_from_rates(args.rates)
        print("\n=== From rates (if provided) ===")
        print(f"max|πK-πK^T| = {mr['RmaxK']:.3e}")
        print(f"RMS|πK-πK^T| = {mr['RrmsK']:.3e}")
        print(f"sigma_K      = {mr['sigmaK']:.3e}")
    except Exception:
        pass

    # Verdict (from counts)
    ok = (mc["Rmax"] <= args.thr_Rmax) and (abs(mc["sigma"]) <= args.thr_sigma) and (abs(J_loop) <= args.thr_Rmax)
    print("\n=== VERDICT ===")
    print("EQUILIBRIUM" if ok else "NESS (non-equilibrium steady state)")

    # Show top offending pairs (helpful debug)
    edges = []
    st = mc["states"]; R = mc["R"]
    for i in range(len(st)):
        for j in range(len(st)):
            if i==j: continue
            edges.append((st[i], st[j], R[i,j], abs(R[i,j])))
    top = sorted(edges, key=lambda x: x[3], reverse=True)[:12]
    print("\nTop 12 |πP-πP^T| pairs:")
    for a,b,rij,mag in top:
        print(f"{a:>12} -> {b:<12}  R={rij:+.3e}")

if __name__ == "__main__":
    main()
