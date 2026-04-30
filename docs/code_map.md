# Code map (hartă curentă)

## Script principal probabil
- `sim/analytic_langevin.py` — candidat principal pentru rulări complete (geometrie + materiale + dinamică Langevin + coliziuni + output extins).

## Fișiere auxiliare importante
- `sim/analytic_langevin_termal_collission.py` — variantă axată pe coliziuni/termalizare la perete (de confirmat ca variantă canonică).
- `sim/analytic_langevin_stabil_gpu_cpu.py` — variantă stabilizată CPU/GPU, include mențiuni explicite pentru model de perete de tip OU.
- `sim/analytic_langevin_stabil_gpu_cpu_simple_geom.py` — variantă similară pentru geometrie simplificată.
- `sim/loop_geometry.py`, `sim/proper_geometry.py`, `sim/geometry_export.py` — suport geometrie/export.

## Analiză și post-procesare
- `sim/analysis_post.py` — post-procesare principală (MSD/tranziții/statistici).
- `sim/analyze.py` — analiză locală (de verificat robustitatea/finisarea).
- `ness_check.py`, `sim/ness_check.py` — verificări de echilibru/NESS.
- `ovito/ovito_pipeline.py` — pipeline de vizualizare.

## Candidate legacy / experimental
- `sim/gpt_shit.py`, `sim/grok_shit1.py` — scripturi ad-hoc / experimentale.
- `sim/test_gpu.py`, `sim/sanity_gpu.py` — teste de infrastructură.
- `sim/run_mpcd.py`, `sim/run_mpcd_light.py`, `sim/run_geometry_brownian.py` — linii alternative (MPCD / brownian cu wall-beads).
- `sim/analytic_brownian.py`, `sim/analytic_brownian_gpu.py` — familie brownian alternativă.
- `sim/analytic_langevin_equil_gpu` — fișier fără extensie (de confirmat rolul exact).

## Output-uri / directoare generate (din cod)
- `sim/out_langevin_8/`, `sim/out_37/`, `sim/out1/`, `sim/out/`, `out/` (în funcție de script).
- Fișiere tipice: `run.gsd`, `run.xyz`, `piece_counts.csv`, `transitions.csv`, `wall_hist_step*.csv`, `msd.csv`.

## Relații scurte între fișiere
1. Scripturile `analytic_langevin*` generează traiectorii/statistici brute.
2. `analysis_post.py` și `analyze.py` consumă output-urile pentru metrici.
3. `ovito_pipeline.py` oferă cale de vizualizare.
4. `repo_inventory.md` + documentele noi din `docs/` fixează contextul baseline/checkpoint.


## Navigare knowledge (suport documentar)
- Cadru fizic reutilizabil: `docs/domain_knowledge.md`.
- Ipoteze și nivel de certitudine: `docs/model_assumptions.md`.
- Capcane de interpretare: `docs/common_pitfalls.md`.
- Plan de validare: `docs/experiments.md` + `docs/roadmap.md`.
