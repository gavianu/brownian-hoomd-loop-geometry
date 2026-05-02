"""Grafice comparative pentru cele 4 simulări balistice (gamma=0.01)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import gsd.hoomd
from pathlib import Path

OUT = Path('sim_out/plots_ballistic')
OUT.mkdir(exist_ok=True)

CONFIGS = {
    'elastic':       ('loop_elastic_ballistic',        'ElasticBounce\n(referință)'),
    'damped_uni':    ('loop_damped_uniform_ballistic',  'DampedBounce\nuniform'),
    'damped_hetero': ('loop_damped_hetero_ballistic',   'DampedBounce\nhetero'),
    'ou_hetero':     ('loop_ou_hetero_ballistic',       'OUBounce\nhetero'),
}
COLORS = {'elastic': 'steelblue', 'damped_uni': 'tomato',
          'damped_hetero': 'darkorange', 'ou_hetero': 'seagreen'}

kT, m = 1.0, 1.0
s = np.sqrt(kT / m)
v_theory = np.linspace(0, 5, 400)
# Maxwell-Boltzmann 3D
mb_theory = 4*np.pi * (m/(2*np.pi*kT))**1.5 * v_theory**2 * np.exp(-m*v_theory**2/(2*kT))

# ── 1. Distribuția |v| la ultimul frame ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
v2_last = {}

for ax, (key, (dname, label)) in zip(axes, CONFIGS.items()):
    f = gsd.hoomd.open(f'sim_out/{dname}/run.gsd')
    v = f[-1].particles.velocity  # (N,3)
    f.close()
    vmag = np.linalg.norm(v, axis=1)
    v2_last[key] = float(np.mean(vmag**2))

    ax.hist(vmag, bins=80, density=True, alpha=0.7, color=COLORS[key], label='simulare')
    ax.plot(v_theory, mb_theory, 'k--', lw=1.5, label='MB teoretic')
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('|v|')
    ax.set_ylabel('densitate prob.')
    ax.set_xlim(0, 4.5)
    vrms = np.sqrt(np.mean(vmag**2))
    ax.axvline(vrms, color=COLORS[key], lw=1.2, ls=':',
               label=f'$v_{{rms}}$={vrms:.3f}')
    ax.legend(fontsize=8)

fig.suptitle('Distribuția |v| la ultimul frame — γ=0.01 (balistic)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT / 'fig_mb_ballistic.png', dpi=150)
plt.close()
print("fig_mb_ballistic.png saved")

# ── 2. <v²>(t) — evoluție temporală ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for key, (dname, label) in CONFIGS.items():
    f = gsd.hoomd.open(f'sim_out/{dname}/run.gsd')
    v2_series = []
    steps = []
    for frame in f:
        v = frame.particles.velocity
        v2_series.append(float(np.mean(v**2)))
        steps.append(frame.configuration.step)
    f.close()
    ax.plot(steps, v2_series, color=COLORS[key], label=label.replace('\n', ' '), lw=2)

ax.axhline(3*kT/m, color='k', ls='--', lw=1.2, label='$3kT/m$ teoretic')
ax.set_xlabel('step')
ax.set_ylabel(r'$\langle v^2 \rangle$')
ax.set_title(r'Evoluția $\langle v^2 \rangle$ în timp — γ=0.01', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / 'fig_v2_time.png', dpi=150)
plt.close()
print("fig_v2_time.png saved")

# ── 3. Counts CUBE_L vs CUBE_R — asimetrie ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (key, (dname, label)) in zip(axes, CONFIGS.items()):
    df = pd.read_csv(f'sim_out/{dname}/piece_counts.csv')
    steps = df['step']
    total = df['CUBE_L'] + df['CUBE_R']
    ratio_L = df['CUBE_L'] / total
    ratio_R = df['CUBE_R'] / total

    ax.plot(steps, ratio_L, color='royalblue', lw=2, label='CUBE_L / (L+R)')
    ax.plot(steps, ratio_R, color='crimson', lw=2, label='CUBE_R / (L+R)')
    ax.axhline(0.5, color='k', ls='--', lw=1, label='simetrie 50%')
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('step')
    ax.set_ylabel('fracție particule')
    ax.set_ylim(0.44, 0.56)
    ax.legend(fontsize=8)

fig.suptitle('Asimetria CUBE_L vs. CUBE_R — γ=0.01', fontsize=13)
plt.tight_layout()
plt.savefig(OUT / 'fig_asymmetry_ballistic.png', dpi=150)
plt.close()
print("fig_asymmetry_ballistic.png saved")

# ── 4. Counts toate regiunile — stacked bar la ultimul step ──────────────────
fig, ax = plt.subplots(figsize=(11, 5))
regions = ['CUBE_L', 'CUBE_R', 'FUN_1', 'FUN_2', 'FUN_3', 'FUN_4', 'FUN_5', 'FUN_6', 'RET']
x = np.arange(len(regions))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (key, (dname, label)) in enumerate(CONFIGS.items()):
    df = pd.read_csv(f'sim_out/{dname}/piece_counts.csv')
    last = df.iloc[-1]
    counts = [last[r] for r in regions]
    ax.bar(x + offsets[i]*width, counts, width, color=COLORS[key],
           label=label.replace('\n', ' '), alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(regions, rotation=20)
ax.set_ylabel('nr. particule')
ax.set_title('Distribuția particulelor pe regiuni — ultimul frame, γ=0.01', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / 'fig_counts_final.png', dpi=150)
plt.close()
print("fig_counts_final.png saved")

# ── 5. Temperatura locală per cameră: <v²>/3 per regiune ─────────────────────
# din GSD: nu avem tag per regiune — folosim piece_counts ca proxy pentru ocupare
# Facem doar un summary text cu <v²> global per config

print("\n=== SUMMARY v² la ultimul frame ===")
print(f"{'Config':<30} {'<v²>':>8} {'T_eff':>8} {'ratio T/T0':>10}")
for key, (dname, label) in CONFIGS.items():
    v2 = v2_last[key]
    T_eff = v2 / 3.0
    print(f"{dname:<30} {v2:>8.4f} {T_eff:>8.4f} {T_eff/kT:>10.4f}")

# ── 6. Asimetrie L/R la stare staționară — bar chart comparativ ──────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels_short = ['Elastic', 'Damped\nuniform', 'Damped\nhetero', 'OU\nhetero']
asymmetry = []

for key, (dname, label) in CONFIGS.items():
    df = pd.read_csv(f'sim_out/{dname}/piece_counts.csv')
    # media ultimelor 10 frame-uri
    last10 = df.tail(10)
    total = last10['CUBE_L'] + last10['CUBE_R']
    asym = float((last10['CUBE_L'] / total - 0.5).mean())
    asymmetry.append(asym)

bars = ax.bar(labels_short, asymmetry,
              color=[COLORS[k] for k in CONFIGS.keys()], alpha=0.85, edgecolor='k')
ax.axhline(0, color='k', lw=1.2, ls='--')
ax.set_ylabel('(CUBE_L / (L+R)) − 0.5\n(pozitiv = mai multe în L)')
ax.set_title('Asimetria L−R medie (ultimele 10 frame-uri) — γ=0.01', fontsize=12)
for bar, val in zip(bars, asymmetry):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:+.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(OUT / 'fig_asym_bar.png', dpi=150)
plt.close()
print("fig_asym_bar.png saved")

print(f"\nToate graficele salvate în {OUT}/")
