"""Verifică conservarea numărului de particule în toate sim_out/loop_*."""
import pandas as pd
import numpy as np
from pathlib import Path

LOOP_DIRS = sorted(Path('sim_out').glob('loop_*/'))

print(f"{'sim':<40} {'N_init':>7} {'N_final':>7} {'N_min':>7} {'N_max':>7} {'drift':>8} {'ok?'}")
print('-' * 85)

for d in LOOP_DIRS:
    pc = d / 'piece_counts.csv'
    if not pc.exists():
        continue
    df = pd.read_csv(pc)
    # suma tuturor regiunilor = total particule la fiecare step
    region_cols = [c for c in df.columns if c != 'step']
    totals = df[region_cols].sum(axis=1)

    N_init  = int(totals.iloc[0])
    N_final = int(totals.iloc[-1])
    N_min   = int(totals.min())
    N_max   = int(totals.max())
    drift   = N_final - N_init
    ok = 'OK' if (N_min == N_max == N_init) else f'DRIFT {drift:+d} (var={N_max-N_min})'

    print(f"{d.name:<40} {N_init:>7} {N_final:>7} {N_min:>7} {N_max:>7} {drift:>+8}  {ok}")

print()
# detaliu pe cea cu cele mai multe tranziții
print("=== Detaliu per regiune — loop_elastic_ballistic (ultimele 5 steps) ===")
df = pd.read_csv('sim_out/loop_elastic_ballistic/piece_counts.csv')
print(df.tail(5).to_string(index=False))

print()
print("=== Detaliu per regiune — loop_ou_hetero_ballistic (ultimele 5 steps) ===")
df2 = pd.read_csv('sim_out/loop_ou_hetero_ballistic/piece_counts.csv')
print(df2.tail(5).to_string(index=False))

print()
print("=== Detaliu per regiune — loop_chambers_ou (ultimele 5 steps) ===")
df3 = pd.read_csv('sim_out/loop_chambers_ou/piece_counts.csv')
print(df3.tail(5).to_string(index=False))
