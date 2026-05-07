"""Separă figurile combinate în PNG-uri individuale.

Produce:
  fig2a_mb_scalar.png       — hist |v| (box configs)
  fig2b_mb_component.png    — hist v_x (box configs)
  fig4a_counts_reference.png — counts CUBE_L/CUBE_R pt OU hetero (referință)
  fig4b_counts_others.png   — counts pentru celelalte 4 configs (2x2)
  fig6a_mb_loop_scalar.png  — hist |v| loop_chambers_ou
  fig6b_mb_loop_components.png — hist v_x, v_y loop_chambers_ou
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gsd.hoomd

OUT = 'sim_out/plots'
os.makedirs(OUT, exist_ok=True)

COLORS = {
    'baseline_simple_box':         '#2196F3',
    'box_elastic':                 '#4CAF50',
    'box_damped':                  '#FF9800',
    'box_ou_absorbent':            '#9C27B0',
    'box_damped_absorbent':        '#F44336',
    'loop_chambers_ou':            '#2196F3',
    'loop_chambers_elastic':       '#4CAF50',
    'loop_chambers_ou_uniform':    '#9C27B0',
    'loop_chambers_damped_hetero': '#F44336',
    'loop_chambers_damped_uniform':'#FF9800',
}
LABELS = {
    'baseline_simple_box':         'Box OU (e_n=0.95) — referință',
    'box_elastic':                 'Box Elastic',
    'box_damped':                  'Box Damped (e_n=0.95)',
    'box_ou_absorbent':            'Box OU absorbant (e_n=0.05)',
    'box_damped_absorbent':        'Box Damped absorbant (e_n=0.05)',
    'loop_chambers_ou':            'Loop OU hetero — referință',
    'loop_chambers_elastic':       'Loop Elastic',
    'loop_chambers_ou_uniform':    'Loop OU uniform',
    'loop_chambers_damped_hetero': 'Loop Damped hetero',
    'loop_chambers_damped_uniform':'Loop Damped uniform',
}

kT_over_m = 1.0
v_th = np.sqrt(kT_over_m)
v_range = np.linspace(0, 5 * v_th, 300)
f_mb = 4 * np.pi * (1 / (2 * np.pi * kT_over_m))**1.5 * v_range**2 * np.exp(-v_range**2 / (2 * kT_over_m))
vx_range = np.linspace(-4 * v_th, 4 * v_th, 300)
f_gauss = np.exp(-vx_range**2 / (2 * kT_over_m)) / np.sqrt(2 * np.pi * kT_over_m)


def read_last_velocities(cfg):
    path = f'sim_out/{cfg}/run.gsd'
    if not os.path.exists(path):
        print(f'  Lipsă GSD: {path}')
        return None
    t = gsd.hoomd.open(path, 'r')
    v = t[-1].particles.velocity.copy()
    t.close()
    return v


def read_counts(cfg):
    path = f'sim_out/{cfg}/piece_counts.csv'
    if not os.path.exists(path):
        print(f'  Lipsă CSV: {path}')
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    steps = [int(r['step']) for r in rows]
    pieces = [k for k in rows[0] if k != 'step']
    counts = {p: [int(r[p]) for r in rows] for p in pieces}
    return steps, counts


# ── fig2a: distribuție viteze scalare |v| — box configs ─────────────────────
print('Generez fig2a...')
ou_configs   = ['baseline_simple_box', 'box_ou_absorbent']
damp_configs = ['box_elastic', 'box_damped', 'box_damped_absorbent']

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for cfg in ou_configs + damp_configs:
    v = read_last_velocities(cfg)
    if v is None:
        continue
    speeds = np.linalg.norm(v, axis=1)
    ax.hist(speeds, bins=60, density=True, alpha=0.4, color=COLORS[cfg], label=LABELS[cfg])

ax.plot(v_range, f_mb, 'k-', linewidth=2, label='MB teoretic')
ax.set_xlabel('$|v|$', fontsize=13)
ax.set_ylabel('Densitate de probabilitate', fontsize=12)
ax.set_title('Distribuția vitezelor scalare — cutie simplă', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT, 'fig2a_mb_scalar.png')
fig.savefig(path)
plt.close(fig)
print(f'  Salvat: {path}')


# ── fig2b: distribuție componentă v_x — box configs ─────────────────────────
print('Generez fig2b...')
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for cfg in ou_configs + damp_configs:
    v = read_last_velocities(cfg)
    if v is None:
        continue
    ax.hist(v[:, 0], bins=60, density=True, alpha=0.4, color=COLORS[cfg], label=LABELS[cfg])

ax.plot(vx_range, f_gauss, 'k-', linewidth=2, label='Gaussian teoretic')
ax.set_xlabel('$v_x$', fontsize=13)
ax.set_ylabel('Densitate de probabilitate', fontsize=12)
ax.set_title('Distribuția componentei $v_x$ — cutie simplă', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT, 'fig2b_mb_component.png')
fig.savefig(path)
plt.close(fig)
print(f'  Salvat: {path}')


# ── fig4a: counts CUBE_L vs CUBE_R — OU hetero (referință) ──────────────────
print('Generez fig4a...')
cfg_ref = 'loop_chambers_ou'
result = read_counts(cfg_ref)
if result is not None:
    steps, counts = result
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    if 'CUBE_L' in counts:
        ax.plot(steps, counts['CUBE_L'], color='#2196F3', label='CUBE\_L', linewidth=2)
    if 'CUBE_R' in counts:
        ax.plot(steps, counts['CUBE_R'], color='#F44336', label='CUBE\_R', linewidth=2)
    ax.set_title('Ocupanța camerelor — Loop OU eterogen (referință)', fontsize=13)
    ax.set_xlabel('Pas de simulare', fontsize=12)
    ax.set_ylabel('Număr particule', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig4a_counts_reference.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Salvat: {path}')

# ── fig4b: counts CUBE_L vs CUBE_R — toate 5 configs (2×3, ultimul gol) ──────
print('Generez fig4b...')
other_configs = [
    'loop_chambers_ou',
    'loop_chambers_elastic',
    'loop_chambers_ou_uniform',
    'loop_chambers_damped_hetero',
    'loop_chambers_damped_uniform',
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)

for i, cfg in enumerate(other_configs):
    ax = axes_flat[i]
    result = read_counts(cfg)
    if result is None:
        ax.set_title(LABELS.get(cfg, cfg), fontsize=9)
        ax.axis('off')
        continue
    steps, counts = result
    if 'CUBE_L' in counts:
        ax.plot(steps, counts['CUBE_L'], color='#2196F3', label='CUBE\_L', linewidth=2)
    if 'CUBE_R' in counts:
        ax.plot(steps, counts['CUBE_R'], color='#F44336', label='CUBE\_R', linewidth=2)
    ax.set_title(LABELS.get(cfg, cfg).replace(' — ', '\n'), fontsize=9)
    ax.set_xlabel('Pas de simulare', fontsize=9)
    ax.set_ylabel('Număr particule', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

fig.suptitle('Ocupanța camerelor CUBE\_L vs CUBE\_R — configurații alternative', fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, 'fig4b_counts_others.png')
fig.savefig(path)
plt.close(fig)
print(f'  Salvat: {path}')


# ── fig6a: distribuție |v| — loop_chambers_ou ───────────────────────────────
print('Generez fig6a...')
v = read_last_velocities('loop_chambers_ou')
if v is not None:
    speeds = np.linalg.norm(v, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.hist(speeds, bins=80, density=True, alpha=0.7, color='#2196F3', label='Simulare')
    ax.plot(v_range, f_mb, 'r-', linewidth=2, label='MB teoretic')
    ax.set_xlabel('$|v|$', fontsize=13)
    ax.set_ylabel('Densitate de probabilitate', fontsize=12)
    ax.set_title('Viteze scalare — Loop Chambers OUBounce eterogen', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig6a_mb_loop_scalar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Salvat: {path}')

# ── fig6b: distribuție componente v_x, v_y — loop_chambers_ou ───────────────
print('Generez fig6b...')
if v is not None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.hist(v[:, 0], bins=80, density=True, alpha=0.6, color='#2196F3', label='$v_x$ simulare')
    ax.hist(v[:, 1], bins=80, density=True, alpha=0.6, color='#F44336', label='$v_y$ simulare')
    ax.plot(vx_range, f_gauss, 'k-', linewidth=2, label='Gaussian teoretic')
    ax.set_xlabel('Componentă viteză', fontsize=13)
    ax.set_ylabel('Densitate de probabilitate', fontsize=12)
    ax.set_title('Componentele $v_x$, $v_y$ — Loop Chambers OUBounce eterogen', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig6b_mb_loop_components.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Salvat: {path}')

print('\nGata.')
