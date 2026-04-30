# CLAUDE.md

## Scopul proiectului
Proiectul modelează mișcarea Browniană/Langevin a tracerilor într-o geometrie internă complexă (2 camere + 6 funnel + canal retur), pentru a înțelege statisticile de transport și rezidență când frontierele au materiale diferite. Cod pentru lucrarea de licență (UB Fizică, iunie 2026).

## Arhitectură
- Pachet Python `brownian_sim/` cu design SOLID (vezi README).
- CLI prin YAML (`configs/*.yaml`) → `python -m brownian_sim.scripts.run_sim --config ...`.
- Testele pytest (25, toate passing) acoperă primitivele geometrice, modelele de perete și echilibrul Langevin+OU.
- Scripturile vechi sunt în `legacy/sim_scripts/` ca referință istorică — nu se mai modifică.

## Modelul fizic
- Gaz ideal diluat (fără interacțiuni tracer–tracer).
- Evoluție Langevin în volum (fricțiune + zgomot termic calibrat FDT).
- Frontiere descrise geometric (Box, CylX, CylY) + material (`e_n`, `beta_t`).
- Modele de perete implementate și testate: `ElasticBounce`, `DampedBounce`, `OUBounce`.
- Modelul OU este cel "termalizant corect" — menține distribuția MB la temperatura țintă.

## Ce NU trebuie schimbat fără confirmare
1. Semantica parametrilor materiali (`e_n ∈ [0,1]`, `beta_t ∈ [0,1]`).
2. Formulele de bounce din `brownian_sim/physics/wall_models.py` — sunt testate vs teorie FDT.
3. Conservarea numelor `CUBE_L`, `CUBE_R`, `FUN_*`, `RET` în preset `loop_chambers` — sunt în CSV-uri și în lucrare.

## Separare obligatorie
- **Teorie**: `latex/main.tex` secțiunile 2.x; cod: `brownian_sim/physics/`.
- **Geometrie**: `brownian_sim/geometry/`. O geometrie nouă e un preset nou.
- **I/O**: `brownian_sim/io/`. Writer-ii sunt injectați în `SimulationConfig.writers`.
- **Analiză**: `brownian_sim/analysis/`. Rulează pe orice director `sim_out/`.

## Interpretări corecte (de păstrat)
- Geometria singură nu explică NESS în gaz ideal cu pereți termalizanți.
- Principiul 2 este respectat dacă pereții sunt OU (FDT local).
- NESS real = dezechilibru extern (ex. 2 termostate la temperaturi diferite).

## Interpretări greșite (de evitat)
- „Doar geometria produce NESS.” (fals cu OU)
- „OU la perete e zgomot arbitrar.” (fals — calibrare FDT)
- „Coeficienții `e_n`/`beta_t` sunt detalii numerice.” (fals — determină regimul fizic)

## Status lucrare
- `latex/main.tex` are Introducere + Fundamente teoretice complete (secțiunile 2.1–2.8).
- Rămân de completat: secțiunea Model numeric detaliată, tabel parametri, figuri, Rezultate cu rulări reale, Concluzii cu narațiunea corectă.
- Deadline: ~2026-05-08 (finalizare), susținere iunie 2026.

## Validare refactor vs legacy
- `<v²>` refactor = 3.016 la echilibru (țintă 3.000) ✓
- Niciun OUT în loop_chambers ✓
- Speedup ~62× vs legacy CPU
- 25/25 teste trec
