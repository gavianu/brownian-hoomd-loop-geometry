"""NESS check pe toate sim_out/loop_*."""
import sys, math
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from brownian_sim.analysis.ness import (load_counts, detailed_balance_metrics,
                                         loop_current)

LOOP_DIRS = sorted(Path('sim_out').glob('loop_*/'))

rows = []
for d in LOOP_DIRS:
    tr = d / 'transitions.csv'
    if not tr.exists():
        continue
    states, C = load_counts(None, str(tr))
    mc = detailed_balance_metrics(states, C)
    J_loop, _, _ = loop_current(mc['R'], mc['states'])
    verdict = 'EQUILIBRIUM' if (mc['Rmax'] <= 1e-4 and abs(mc['sigma']) <= 1e-5 and abs(J_loop) <= 1e-4) else 'NESS'
    rows.append({
        'sim': d.name,
        'transitions': mc['transitions'],
        'Rmax': mc['Rmax'],
        'sigma': mc['sigma'],
        'J_loop': J_loop,
        'verdict': verdict,
    })
    print(f"\n{'='*62}")
    print(f"  {d.name}")
    print(f"  transitions={mc['transitions']}")
    print(f"  Rmax       = {mc['Rmax']:.3e}")
    print(f"  sigma      = {mc['sigma']:.3e}")
    print(f"  J_loop     = {J_loop:.3e}")
    print(f"  VERDICT    : {verdict}")
    # top offending pairs
    st = mc['states']; R = mc['R']
    pairs = [(st[i], st[j], R[i,j]) for i in range(len(st)) for j in range(len(st)) if i!=j]
    top = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:6]
    for a, b, r in top:
        print(f"    {a:>12} -> {b:<12}  R={r:+.3e}")

print("\n\n=== SUMMARY ===")
print(f"{'sim':<40} {'N_tr':>6} {'Rmax':>10} {'J_loop':>10} {'verdict'}")
for r in rows:
    print(f"{r['sim']:<40} {r['transitions']:>6} {r['Rmax']:>10.3e} {r['J_loop']:>10.3e}  {r['verdict']}")
