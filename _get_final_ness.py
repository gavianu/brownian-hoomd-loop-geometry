from brownian_sim.analysis.ness import load_counts, detailed_balance_metrics, loop_current

configs = [
    ('sim_out/loop_maxwell_hetero_long/transitions.csv', 'Maxwell hetero LONG'),
    ('sim_out/loop_ou_hetero_long/transitions.csv',      'OU hetero LONG     '),
    ('sim_out/loop_ou_uniform_long/transitions.csv',     'OU uniform LONG    '),
    ('sim_out/loop_maxwell_hetero_short/transitions.csv','Maxwell hetero short'),
    ('sim_out/loop_maxwell_ballistic/transitions.csv',   'Maxwell g=0.01     '),
    ('sim_out/loop_ou_hetero_ballistic/transitions.csv', 'OU hetero g=0.01   '),
    ('sim_out/loop_elastic_ballistic/transitions.csv',   'Elastic g=0.01     '),
]

for path, label in configs:
    try:
        states, C = load_counts(None, path)
        mc = detailed_balance_metrics(states, C)
        J, _, _ = loop_current(mc['R'], mc['states'])
        print(f'{label}: N_tr={mc["transitions"]:>6}, J={J:+.4e}, Rmax={mc["Rmax"]:.3e}')
    except Exception as e:
        print(f'{label}: EROARE - {e}')
