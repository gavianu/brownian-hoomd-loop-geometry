"""Detectie echilibru vs NESS din date de tranzitii.

Expune: load_counts, stationary_from_P, detailed_balance_metrics, loop_current, ness_verdict.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def load_counts(
    path_counts: Optional[str],
    path_edges: Optional[str],
):
    """Incarca (states, C) din CSV matrice de conturi sau CSV lista de muchii.

    path_counts: CSV cu header row+col (transition_counts.csv)
    path_edges:  CSV cu coloane from,to (transitions.csv)
    """
    if path_counts:
        Cdf = pd.read_csv(path_counts)
        states = list(Cdf.columns[1:])
        C = Cdf.iloc[:, 1:].to_numpy(float)
        return states, C

    if not path_edges:
        raise ValueError("Furnizeaza path_counts sau path_edges")

    E = pd.read_csv(path_edges)
    E.columns = [c.strip().lower() for c in E.columns]
    if not {"from", "to"} <= set(E.columns):
        raise ValueError("CSV edges trebuie sa aiba coloanele: from,to")
    states = sorted(list(set(E["from"].astype(str)).union(set(E["to"].astype(str)))))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    C = np.zeros((n, n), dtype=float)
    for f, t in zip(E["from"].astype(str), E["to"].astype(str)):
        C[idx[f], idx[t]] += 1.0
    return states, C


def stationary_from_P(P: np.ndarray, tol: float = 1e-15, itmax: int = 20000) -> np.ndarray:
    """Distributia stationara a lantului Markov P prin iteratie de putere."""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(itmax):
        new = pi @ P
        if np.linalg.norm(new - pi, 1) < tol:
            break
        pi = new
    s = pi.sum()
    return pi / s if s > 0 else pi


def _group_zone(name: str) -> str:
    s = str(name)
    if s in ("CUBE_L", "CUBE_R", "RET", "VERT_L", "VERT_R"):
        return s
    if s.startswith("FUN_") or s.startswith("FUN"):
        return "FUNNELS"
    return "OTHER"


def loop_current(R: np.ndarray, states: list, loop=("CUBE_L", "FUNNELS", "CUBE_R", "RET")):
    """Curentul net de-a lungul ciclului loop in matricea de asimetrie R.

    Returneaza (J_loop, G_aggregated, zone_labels).
    """
    zones = list(dict.fromkeys(loop))
    zidx = {z: i for i, z in enumerate(zones)}
    G = np.zeros((len(zones), len(zones)))
    for i, si in enumerate(states):
        zi = _group_zone(si)
        if zi not in zidx:
            continue
        for j, sj in enumerate(states):
            if i == j:
                continue
            zj = _group_zone(sj)
            if zj not in zidx:
                continue
            G[zidx[zi], zidx[zj]] += R[i, j]

    def along(path):
        s = 0.0
        for a, b in zip(path, path[1:] + path[:1]):
            s += G[zidx[a], zidx[b]]
        return s

    J = along(list(loop)) - along(list(reversed(loop)))
    return J, G, zones


def detailed_balance_metrics(states: list, C: np.ndarray) -> dict:
    """Calculeaza metrici de bilantat detaliat din matricea de conturi C.

    Returneaza dict cu: states, P, pi, R, Rmax, Rrms, sigma, transitions.
    R[i,j] = pi[i]*P[i,j] - pi[j]*P[j,i]  (asimetria fluxului de probabilitate).
    sigma = entropia produsa (proxy pentru rata de productie de entropie).
    """
    row_sum = C.sum(axis=1, keepdims=True)
    keep = row_sum[:, 0] > 0
    if keep.sum() < C.shape[0]:
        states = [s for s, k in zip(states, keep) if k]
        C = C[keep][:, keep]
        row_sum = C.sum(axis=1, keepdims=True)

    P = np.divide(C, row_sum, out=np.zeros_like(C), where=row_sum > 0)
    pi = stationary_from_P(P)

    eps = 1e-15
    n = P.shape[0]
    R = np.zeros_like(P)
    sigma = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            R[i, j] = pi[i] * P[i, j] - pi[j] * P[j, i]
            if P[i, j] > 0 and P[j, i] > 0:
                sigma += 0.5 * R[i, j] * math.log(
                    (pi[i] * P[i, j] + eps) / (pi[j] * P[j, i] + eps)
                )

    Rmax = float(np.max(np.abs(R))) if R.size else 0.0
    Rrms = float(np.sqrt((R ** 2).mean())) if R.size else 0.0
    return dict(
        states=states, P=P, pi=pi, R=R,
        Rmax=Rmax, Rrms=Rrms, sigma=sigma,
        transitions=int(C.sum()),
    )


def ness_verdict(
    Rmax: float,
    sigma: float,
    J_loop: float,
    n_transitions: int = 0,
    thr_sigma: float = 1e-4,
    k_noise: float = 5.0,
) -> str:
    """Returneaza 'echilibru' sau 'NESS'.

    Pragul pentru Rmax si J_loop este adaptat la zgomotul statistic:
    thr = k_noise / sqrt(n_transitions), cu k_noise=5 (5 deviatii standard).
    La n_transitions=0 se foloseste pragul fix 1e-3.
    """
    if n_transitions > 0:
        thr = k_noise / math.sqrt(n_transitions)
    else:
        thr = 1e-3
    if Rmax <= thr and abs(J_loop) <= thr and abs(sigma) <= thr_sigma:
        return "echilibru"
    return "NESS"
