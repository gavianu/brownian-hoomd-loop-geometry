"""Monitor progres si J_loop in timp real pentru rulari lungi.
Rulare: python _monitor_long.py
Nu opreste simularea, citeste doar CSV-urile deja scrise.
"""
import sys, math, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ness_check import load_counts, detailed_balance_metrics_from_counts, loop_current

SIMS = {
    'ou_hetero_long':    Path('sim_out/loop_ou_hetero_long'),
    'ou_uniform_long':   Path('sim_out/loop_ou_uniform_long'),
    'maxwell_hetero_long': Path('sim_out/loop_maxwell_hetero_long'),
}
TOTAL_STEPS = 300000

def check_sim(name, d):
    tr = d / 'transitions.csv'
    pc = d / 'piece_counts.csv'

    if not pc.exists():
        print(f"  {name}: nu a inceput inca (no piece_counts.csv)")
        return

    df = pd.read_csv(pc)
    step_done = int(df['step'].max())
    pct = step_done / TOTAL_STEPS * 100
    region_cols = [c for c in df.columns if c != 'step']
    N = int(df[region_cols].sum(axis=1).iloc[-1])

    # asimetrie L/R
    last = df.iloc[-1]
    LR_total = last['CUBE_L'] + last['CUBE_R']
    asym = float(last['CUBE_L'] / LR_total - 0.5) if LR_total > 0 else 0.0

    print(f"  {name}: step {step_done}/{TOTAL_STEPS} ({pct:.1f}%)  N={N}  asym_LR={asym:+.4f}")

    if not tr.exists():
        print(f"    transitions.csv: nu exista inca")
        return

    try:
        states, C = load_counts(None, str(tr))
        mc = detailed_balance_metrics_from_counts(states, C)
        J_loop, _, _ = loop_current(mc['R'], mc['states'])
        verdict = 'NESS' if (mc['Rmax'] > 1e-4 or abs(J_loop) > 1e-4) else 'ECHILIBRU'
        print(f"    transitions={mc['transitions']}  Rmax={mc['Rmax']:.3e}  J_loop={J_loop:+.4e}  -> {verdict}")
    except Exception as e:
        print(f"    ness_check eroare: {e}")

    # J_loop evolutie in timp daca avem suficiente date
    # imparte tranzitiile in 3 ferestre temporale
    try:
        E = pd.read_csv(tr)
        E.columns = [c.strip().lower() for c in E.columns]
        if 'step' in E.columns and len(E) > 100:
            n3 = len(E) // 3
            for i, label in enumerate(['inceput', 'mijloc', 'final']):
                chunk = E.iloc[i*n3:(i+1)*n3]
                states_c = sorted(list(set(chunk['from'].astype(str)).union(set(chunk['to'].astype(str)))))
                idx = {s:j for j,s in enumerate(states_c)}
                n = len(states_c)
                C_chunk = np.zeros((n,n))
                for f,t in zip(chunk['from'].astype(str), chunk['to'].astype(str)):
                    if f in idx and t in idx:
                        C_chunk[idx[f], idx[t]] += 1
                mc_c = detailed_balance_metrics_from_counts(states_c, C_chunk)
                J_c, _, _ = loop_current(mc_c['R'], mc_c['states'])
                print(f"    J_loop [{label:7s}] = {J_c:+.4e}")
    except Exception:
        pass

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status simulari lungi\n")
for name, d in SIMS.items():
    check_sim(name, d)
    print()
