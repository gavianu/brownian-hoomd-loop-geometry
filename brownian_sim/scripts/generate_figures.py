"""Generează figuri noi pentru lucrarea de licență.

Figuri produse:
  fig7_density_ratio.png   — raport CUBE_L/CUBE_R în timp, toate 3 simulări lungi
  fig8_jloop_bar.png       — J_loop cu bare de eroare pentru toate 6 simulări
  fig9_transitions_heatmap.png — heatmap matrice tranziții OU vs Maxwell

Salvate în sim_out/plots/ la 150 DPI, figsize mare.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from brownian_sim.analysis.ness import load_counts, detailed_balance_metrics, loop_current

BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'sim_out')
OUT  = os.path.join(BASE, 'plots')
os.makedirs(OUT, exist_ok=True)

STYLE = dict(figsize=(10, 5.5), dpi=150)

# ── fig7: raport densitate CUBE_L / CUBE_R în timp ─────────────────────────

def fig7_density_ratio():
    sims = [
        ('loop_ou_hetero_long',   'OUBounce eterogen',   'tab:blue'),
        ('loop_ou_uniform_long',  'OUBounce uniform',    'tab:orange'),
        ('loop_maxwell_hetero_long', 'MaxwellDiffuse eterogen', 'tab:green'),
    ]

    fig, ax = plt.subplots(**STYLE)
    for dirname, label, color in sims:
        df = pd.read_csv(os.path.join(BASE, dirname, 'piece_counts.csv'))
        ratio = df['CUBE_L'] / df['CUBE_R']
        ax.plot(df['step'] / 1000, ratio, label=label, color=color, lw=1.8)

    ax.axhline(1.0, color='black', lw=1.0, ls='--', label='echilibru (raport = 1)')
    ax.set_xlabel('Pas de simulare ($\\times 10^3$)', fontsize=13)
    ax.set_ylabel('$N_{\\mathrm{CUBE\\_L}} / N_{\\mathrm{CUBE\\_R}}$', fontsize=13)
    ax.set_title('Raportul de ocupare CUBE\\_L / CUBE\\_R în timp', fontsize=14)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig7_density_ratio.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Salvat: {path}')


# ── fig8: J_loop bar chart cu bare de eroare statistică ────────────────────

def fig8_jloop_bar():
    sims = [
        ('loop_maxwell_hetero_long',   'Maxwell\neterogen\nγ=1',   'tab:green'),
        ('loop_ou_hetero_long',        'OU\neterogen\nγ=1',        'tab:blue'),
        ('loop_ou_uniform_long',       'OU\nuniform\nγ=1',         'tab:orange'),
        ('loop_maxwell_ballistic',     'Maxwell\neterogen\nγ=0.01','tab:olive'),
        ('loop_ou_hetero_ballistic',   'OU\neterogen\nγ=0.01',     'tab:cyan'),
        ('loop_elastic_ballistic',     'Elastic\nγ=0.01',          'tab:red'),
    ]

    labels, jvals, jerrs = [], [], []
    for dirname, label, _ in sims:
        path = os.path.join(BASE, dirname, 'transitions.csv')
        if not os.path.exists(path):
            print(f'  Lipsa: {path}')
            continue
        states, C = load_counts(None, path)
        m = detailed_balance_metrics(states, C)
        from brownian_sim.analysis.ness import loop_current
        J, _, _ = loop_current(m['R'], m['states'])
        n = m['transitions']
        err = 5.0 / np.sqrt(n) if n > 0 else 0.0
        labels.append(label)
        jvals.append(J)
        jerrs.append(err)

    colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:olive', 'tab:cyan', 'tab:red']
    colors = colors[:len(labels)]

    fig, ax = plt.subplots(**STYLE)
    x = np.arange(len(labels))
    bars = ax.bar(x, jvals, yerr=jerrs, capsize=6, color=colors, alpha=0.85,
                  error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.axhline(0, color='black', lw=1.0, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('$J_{\\mathrm{loop}}$', fontsize=13)
    ax.set_title('Curentul net $J_{\\mathrm{loop}}$ per simulare (bare de eroare $5/\\sqrt{N_{\\mathrm{tr}}}$)',
                 fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig8_jloop_bar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Salvat: {path}')


# ── fig9: heatmap matrice tranziții normalizată — OU hetero vs Maxwell ──────

def fig9_heatmap():
    pairs = [
        ('loop_ou_hetero_long',      'OUBounce eterogen'),
        ('loop_maxwell_hetero_long', 'MaxwellDiffuse eterogen'),
    ]

    ZONE_ORDER = ['CUBE_L', 'FUN_1', 'FUN_2', 'FUN_3', 'FUN_4', 'FUN_5', 'FUN_6', 'CUBE_R', 'RET']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    for ax, (dirname, title) in zip(axes, pairs):
        path = os.path.join(BASE, dirname, 'transitions.csv')
        states, C = load_counts(None, path)

        # reordonează după ZONE_ORDER
        order = [s for s in ZONE_ORDER if s in states]
        idx = [states.index(s) for s in order]
        C_ord = C[np.ix_(idx, idx)]

        row_sum = C_ord.sum(axis=1, keepdims=True)
        P = np.divide(C_ord, row_sum, out=np.zeros_like(C_ord), where=row_sum > 0)

        im = ax.imshow(P, cmap='YlOrRd', vmin=0, vmax=P.max())
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(order, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Stare destinație', fontsize=10)
        ax.set_ylabel('Stare sursă', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Matricea de tranziție $P_{ij}$ normalizată per rând', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig9_transitions_heatmap.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Salvat: {path}')


if __name__ == '__main__':
    print('Generez fig7...')
    fig7_density_ratio()
    print('Generez fig8...')
    fig8_jloop_bar()
    print('Generez fig9...')
    fig9_heatmap()
    print('Gata.')
