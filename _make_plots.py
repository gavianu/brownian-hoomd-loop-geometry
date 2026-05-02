"""Generare grafice pentru lucrarea de licenta."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gsd.hoomd

os.makedirs("sim_out/plots", exist_ok=True)

# ── paleta consistenta ──────────────────────────────────────────────────────
COLORS = {
    'baseline_simple_box':        '#2196F3',
    'box_elastic':                '#4CAF50',
    'box_damped':                 '#FF9800',
    'box_ou_absorbent':           '#9C27B0',
    'box_damped_absorbent':       '#F44336',
    'loop_chambers_ou':           '#2196F3',
    'loop_chambers_elastic':      '#4CAF50',
    'loop_chambers_ou_uniform':   '#9C27B0',
    'loop_chambers_damped_hetero':'#F44336',
    'loop_chambers_damped_uniform':'#FF9800',
}
LABELS = {
    'baseline_simple_box':        'Box OU (e_n=0.95) — referinta',
    'box_elastic':                'Box Elastic',
    'box_damped':                 'Box Damped (e_n=0.95)',
    'box_ou_absorbent':           'Box OU absorbant (e_n=0.05)',
    'box_damped_absorbent':       'Box Damped absorbant (e_n=0.05)',
    'loop_chambers_ou':           'Loop OU hetero — referinta',
    'loop_chambers_elastic':      'Loop Elastic',
    'loop_chambers_ou_uniform':   'Loop OU uniform',
    'loop_chambers_damped_hetero':'Loop Damped hetero (NESS fals)',
    'loop_chambers_damped_uniform':'Loop Damped uniform',
}

# ── helper: citeste v2 din toate framele GSD ────────────────────────────────
def read_v2_series(cfg):
    path = f'sim_out/{cfg}/run.gsd'
    if not os.path.exists(path):
        return None, None
    t = gsd.hoomd.open(path, 'r')
    steps, v2s = [], []
    for frame in t:
        v = frame.particles.velocity
        steps.append(frame.configuration.step)
        v2s.append(float(np.mean(np.sum(v**2, axis=1))))
    t.close()
    return np.array(steps), np.array(v2s)

# ── helper: citeste viteze din ultima frame ─────────────────────────────────
def read_last_velocities(cfg):
    path = f'sim_out/{cfg}/run.gsd'
    if not os.path.exists(path):
        return None
    t = gsd.hoomd.open(path, 'r')
    v = t[-1].particles.velocity.copy()
    t.close()
    return v

# ── helper: citeste counts CSV ──────────────────────────────────────────────
def read_counts(cfg):
    path = f'sim_out/{cfg}/piece_counts.csv'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    steps = [int(r['step']) for r in rows]
    pieces = [k for k in rows[0] if k != 'step']
    counts = {p: [int(r[p]) for r in rows] for p in pieces}
    return steps, counts


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 1: <v²> vs timp — configs box
# ════════════════════════════════════════════════════════════════════════════
box_configs = ['baseline_simple_box','box_elastic','box_damped','box_ou_absorbent','box_damped_absorbent']

fig, ax = plt.subplots(figsize=(8, 5))
for cfg in box_configs:
    steps, v2 = read_v2_series(cfg)
    if steps is None: continue
    ax.plot(steps, v2, color=COLORS[cfg], label=LABELS[cfg], linewidth=2)

ax.axhline(3.0, color='k', linestyle='--', linewidth=1, label='Tinta: $\\langle v^2 \\rangle = 3k_BT/m$')
ax.set_xlabel('Pas de timp', fontsize=12)
ax.set_ylabel(r'$\langle v^2 \rangle$', fontsize=13)
ax.set_title('Evolutia energiei cinetice medii — geometrie simpla (box)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sim_out/plots/fig1_v2_box.png', dpi=150)
plt.close()
print("Salvat: fig1_v2_box.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 2: distributie Maxwell-Boltzmann — box configs
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

kT_over_m = 1.0
v_th = np.sqrt(kT_over_m)
v_range = np.linspace(0, 5*v_th, 300)
# MB viteze scalare: f(v) = 4pi*(m/2pi kT)^3/2 * v^2 * exp(-mv^2/2kT)
f_mb = 4*np.pi * (1/(2*np.pi*kT_over_m))**1.5 * v_range**2 * np.exp(-v_range**2/(2*kT_over_m))
# distributie pe o componenta: gaussian N(0, kT/m)
vx_range = np.linspace(-4*v_th, 4*v_th, 300)
f_gauss = np.exp(-vx_range**2/(2*kT_over_m)) / np.sqrt(2*np.pi*kT_over_m)

ou_configs   = ['baseline_simple_box', 'box_ou_absorbent']
damp_configs = ['box_elastic', 'box_damped', 'box_damped_absorbent']

for cfg in ou_configs + damp_configs:
    v = read_last_velocities(cfg)
    if v is None: continue
    speeds = np.linalg.norm(v, axis=1)
    ls = '-' if cfg in ou_configs else '--'
    axes[0].hist(speeds, bins=60, density=True, alpha=0.4,
                 color=COLORS[cfg], label=LABELS[cfg])
    axes[1].hist(v[:,0], bins=60, density=True, alpha=0.4,
                 color=COLORS[cfg], label=LABELS[cfg])

axes[0].plot(v_range, f_mb, 'k-', linewidth=2, label='MB teoretic')
axes[0].set_xlabel('$|v|$', fontsize=12)
axes[0].set_ylabel('Densitate probabilitate', fontsize=11)
axes[0].set_title('Distributie viteze scalare', fontsize=11)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(vx_range, f_gauss, 'k-', linewidth=2, label='Gaussian teoretic')
axes[1].set_xlabel('$v_x$', fontsize=12)
axes[1].set_title('Distributie componenta $v_x$', fontsize=11)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Distributia Maxwell-Boltzmann — verificare echilibru termic', fontsize=12)
plt.tight_layout()
plt.savefig('sim_out/plots/fig2_mb_distribution.png', dpi=150)
plt.close()
print("Salvat: fig2_mb_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 3: <v²> vs timp — configs loop
# ════════════════════════════════════════════════════════════════════════════
loop_configs = ['loop_chambers_ou','loop_chambers_elastic','loop_chambers_ou_uniform',
                'loop_chambers_damped_hetero','loop_chambers_damped_uniform']

fig, ax = plt.subplots(figsize=(8, 5))
for cfg in loop_configs:
    steps, v2 = read_v2_series(cfg)
    if steps is None: continue
    ax.plot(steps, v2, color=COLORS[cfg], label=LABELS[cfg], linewidth=2)

ax.axhline(3.0, color='k', linestyle='--', linewidth=1, label='Tinta: 3')
ax.set_xlabel('Pas de timp', fontsize=12)
ax.set_ylabel(r'$\langle v^2 \rangle$', fontsize=13)
ax.set_title('Evolutia energiei cinetice medii — geometrie loop chambers', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sim_out/plots/fig3_v2_loop.png', dpi=150)
plt.close()
print("Salvat: fig3_v2_loop.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 4: counts CUBE_L vs CUBE_R — loop configs
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(loop_configs), figsize=(16, 4), sharey=False)

for i, cfg in enumerate(loop_configs):
    result = read_counts(cfg)
    if result is None:
        axes[i].set_title(cfg, fontsize=8)
        continue
    steps, counts = result
    ax = axes[i]
    if 'CUBE_L' in counts:
        ax.plot(steps, counts['CUBE_L'], color='#2196F3', label='CUBE_L', linewidth=2)
    if 'CUBE_R' in counts:
        ax.plot(steps, counts['CUBE_R'], color='#F44336', label='CUBE_R', linewidth=2)
    ax.set_title(LABELS[cfg].replace(' — ', '\n'), fontsize=7)
    ax.set_xlabel('Pas', fontsize=8)
    ax.set_ylabel('N particule', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

fig.suptitle('Ocupanta camerelor CUBE_L vs CUBE_R in timp', fontsize=12)
plt.tight_layout()
plt.savefig('sim_out/plots/fig4_counts_loop.png', dpi=150)
plt.close()
print("Salvat: fig4_counts_loop.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 5: raport L/R final — comparatie toate loop configs (bar chart)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))

ratios = []
labels_bar = []
colors_bar = []
for cfg in loop_configs:
    result = read_counts(cfg)
    if result is None: continue
    steps, counts = result
    cl = counts.get('CUBE_L', [0])[-1]
    cr = counts.get('CUBE_R', [1])[-1]
    ratio = cl / cr if cr > 0 else 0
    ratios.append(ratio)
    labels_bar.append(LABELS[cfg].replace(' — ', '\n').replace('Loop ', ''))
    colors_bar.append(COLORS[cfg])

x = np.arange(len(ratios))
bars = ax.bar(x, ratios, color=colors_bar, alpha=0.8, edgecolor='k', linewidth=0.5)
ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Echilibru perfect (L/R=1)')
ax.set_xticks(x)
ax.set_xticklabels(labels_bar, fontsize=8)
ax.set_ylabel('Raport N(CUBE_L) / N(CUBE_R)', fontsize=11)
ax.set_title('Asimetria de ocupanta la starea finala — loop chambers', fontsize=12)
ax.set_ylim(0.9, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('sim_out/plots/fig5_asymmetry.png', dpi=150)
plt.close()
print("Salvat: fig5_asymmetry.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURA 6: distributie MB loop OU hetero (referinta principala)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cfg = 'loop_chambers_ou'
v = read_last_velocities(cfg)
if v is not None:
    speeds = np.linalg.norm(v, axis=1)
    axes[0].hist(speeds, bins=80, density=True, alpha=0.7, color='#2196F3', label='Simulare')
    axes[0].plot(v_range, f_mb, 'r-', linewidth=2, label='MB teoretic')
    axes[0].set_xlabel('$|v|$', fontsize=12)
    axes[0].set_ylabel('Densitate probabilitate', fontsize=11)
    axes[0].set_title('Viteze scalare — Loop OU hetero', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(v[:,0], bins=80, density=True, alpha=0.7, color='#2196F3', label='$v_x$ simulare')
    axes[1].hist(v[:,1], bins=80, density=True, alpha=0.7, color='#F44336', label='$v_y$ simulare')
    axes[1].plot(vx_range, f_gauss, 'k-', linewidth=2, label='Gaussian teoretic')
    axes[1].set_xlabel('Componenta viteza', fontsize=12)
    axes[1].set_title('Componente $v_x$, $v_y$ — Loop OU hetero', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

fig.suptitle('Verificare echilibru Maxwell-Boltzmann — Loop Chambers OU', fontsize=12)
plt.tight_layout()
plt.savefig('sim_out/plots/fig6_mb_loop_ou.png', dpi=150)
plt.close()
print("Salvat: fig6_mb_loop_ou.png")

print("\nToate graficele au fost generate in sim_out/plots/")
