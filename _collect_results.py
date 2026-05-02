"""Copiaza CSV-urile din sim_out/ in results/ pentru versionare.

Ruleaza dupa ce simularea s-a terminat:
    python _collect_results.py

Copiaza doar piece_counts.csv si transitions.csv (nu run.gsd care e mare).
Adauga si un summary NESS calculat din transitions.csv.
"""
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from brownian_sim.analysis.ness import load_counts, detailed_balance_metrics, loop_current, ness_verdict

SIM_OUT = Path("sim_out")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

dirs = sorted(SIM_OUT.glob("loop_*/"))
if not dirs:
    print("Niciun director loop_* gasit in sim_out/")
    sys.exit(0)

summary_lines = ["sim,steps,transitions,J_loop,Rmax,sigma,verdict"]

for d in dirs:
    dest = RESULTS / d.name
    dest.mkdir(exist_ok=True)

    copied = []
    for fname in ("piece_counts.csv", "transitions.csv"):
        src = d / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
            copied.append(fname)

    # calcul NESS daca avem transitions.csv
    tr = d / "transitions.csv"
    if tr.exists():
        try:
            states, C = load_counts(None, str(tr))
            mc = detailed_balance_metrics(states, C)
            J, _, _ = loop_current(mc["R"], mc["states"])
            verdict = ness_verdict(mc["Rmax"], mc["sigma"], J)
            # citim numarul de pasi din piece_counts
            pc = d / "piece_counts.csv"
            steps = 0
            if pc.exists():
                import csv
                with open(pc) as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    steps = int(rows[-1][0])
            summary_lines.append(
                f"{d.name},{steps},{mc['transitions']},"
                f"{J:+.4e},{mc['Rmax']:.3e},{mc['sigma']:.3e},{verdict}"
            )
            print(f"  {d.name}: {mc['transitions']} tr, J={J:+.4e}, {verdict}")
        except Exception as e:
            print(f"  {d.name}: EROARE NESS - {e}")
    else:
        print(f"  {d.name}: copiat {copied}, fara transitions.csv")

# scrie summary
summary_path = RESULTS / "ness_summary.csv"
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"\nSummary scris in {summary_path}")
print(f"Rezultate in {RESULTS}/ — gata pentru git add + commit")
