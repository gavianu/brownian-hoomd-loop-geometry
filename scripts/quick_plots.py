"""Plot-uri rapide dintr-un director sim_out: counts vs step, distribuție viteze final.

Usage:
    python scripts/quick_plots.py sim_out/loop_chambers_ou
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    d = args.run_dir
    if not (d / "piece_counts.csv").exists():
        print(f"[err] lipsește {d}/piece_counts.csv", file=sys.stderr)
        return 1

    # --- counts per step ---
    counts = pd.read_csv(d / "piece_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in counts.columns:
        if col == "step":
            continue
        ax.plot(counts["step"], counts[col], label=col)
    ax.set_xlabel("step")
    ax.set_ylabel("# particule în piesă")
    ax.set_title("Ocupare piese vs timp")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "plot_counts_vs_step.png", dpi=130)
    plt.close(fig)
    print(f"[ok] {d}/plot_counts_vs_step.png")

    # --- last-frame velocity distribution (din GSD) ---
    try:
        import gsd.hoomd
        with gsd.hoomd.open(name=str(d / "run.gsd"), mode="r") as f:
            frames = list(f)
        v = np.asarray(frames[-1].particles.velocity, dtype=np.float64)
        p = np.asarray(frames[-1].particles.position, dtype=np.float64)
        speed = np.linalg.norm(v, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(speed, bins=60, density=True, alpha=0.8)
        # teoretic: Maxwell speed f(v) = sqrt(2/pi) (m/kT)^{3/2} v^2 exp(-m v^2/2kT)
        # aici kT/m = 1  => f(v) = sqrt(2/pi) v^2 exp(-v^2/2)
        vs = np.linspace(0, speed.max(), 300)
        f_mb = np.sqrt(2 / np.pi) * vs**2 * np.exp(-vs**2 / 2)
        axes[0].plot(vs, f_mb, "r-", label="Maxwell-Boltzmann kT=m=1")
        axes[0].set_xlabel("|v|")
        axes[0].set_ylabel("density")
        axes[0].set_title(f"Distribuție viteze (frame final)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # proiecție XY
        axes[1].hexbin(p[:, 0], p[:, 1], gridsize=80, cmap="viridis")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Densitate particule XY (frame final)")
        axes[1].set_aspect("equal")
        fig.tight_layout()
        fig.savefig(d / "plot_final_state.png", dpi=130)
        plt.close(fig)
        print(f"[ok] {d}/plot_final_state.png")

        v2 = float(np.mean(np.sum(v * v, axis=1)))
        print(f"<v²> = {v2:.3f}  (target 3*kT/m)")
    except Exception as e:
        print(f"[warn] GSD plot skipped: {e}")

    # --- transitions matrix (simplu) ---
    tr = pd.read_csv(d / "transitions.csv")
    if len(tr) > 0:
        M = pd.crosstab(tr["from"], tr["to"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(M.columns)))
        ax.set_xticklabels(M.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(M.index)))
        ax.set_yticklabels(M.index)
        ax.set_xlabel("to")
        ax.set_ylabel("from")
        ax.set_title(f"Matrice tranziții ({len(tr)} total)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(d / "plot_transitions.png", dpi=130)
        plt.close(fig)
        print(f"[ok] {d}/plot_transitions.png  ({len(tr)} tranziții)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
