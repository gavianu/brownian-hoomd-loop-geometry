import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from brownian_sim.analysis.ness import load_counts, detailed_balance_metrics, loop_current

dirs = [
    # === gamma=1, conditii naturale ===
    ('loop_maxwell_hetero_long',   'Maxwell hetero  g=1  LONG  (fizic corect)'),
    ('loop_maxwell_hetero_short',  'Maxwell hetero  g=1  short'),
    ('loop_ou_hetero_long',        'OUBounce hetero g=1  LONG'),
    ('loop_ou_uniform_long',       'OUBounce uniform g=1 LONG'),
    ('loop_chambers_ou',           'OUBounce hetero g=1  short'),
    # === gamma=0.01, balistic ===
    ('loop_maxwell_ballistic',     'Maxwell hetero  g=0.01 balistic'),
    ('loop_ou_hetero_ballistic',   'OUBounce hetero g=0.01 balistic'),
    ('loop_elastic_ballistic',     'Elastic         g=0.01 balistic'),
]

print(f"{'Model':<42} {'N_tr':>6}  {'J_loop':>12}  {'Rmax':>10}  verdict")
print('-' * 85)
for dname, label in dirs:
    tr = Path(f'sim_out/{dname}/transitions.csv')
    if not tr.exists():
        print(f"{label:<42}  [lipseste]")
        continue
    states, C = load_counts(None, str(tr))
    mc = detailed_balance_metrics(states, C)
    J, _, _ = loop_current(mc['R'], mc['states'])
    noise = abs(J) < 5e-3 and mc['Rmax'] < 5e-4
    verdict = 'echilibru' if noise else 'NESS/tranzitoriu'
    print(f"{label:<42} {mc['transitions']:>6}  {J:>+12.4e}  {mc['Rmax']:>10.3e}  {verdict}")
