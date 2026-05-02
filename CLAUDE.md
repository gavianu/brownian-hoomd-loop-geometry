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
- Modele de perete implementate și testate: `ElasticBounce`, `DampedBounce`, `OUBounce`, `MaxwellDiffuse`.
- `MaxwellDiffuse` este modelul fizic corect: v_out complet independent de v_in, satisface bilanțul detaliat microscopic, componenta normală Rayleigh, tangențială gaussiană izotropă.
- `OUBounce` produce distribuția marginală MB corectă (FDT local), dar rupe bilanțul detaliat prin dependența de `|v_n_in|`. La γ=1 diferența e mascată de Langevin în volum.
- NESS real = dezechilibru extern (ex. 2 termostate). Geometria singură nu produce NESS cu pereți termalizanți.

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

## Status lucrare (Mai 2026)
`latex/main.tex` — 27 pagini, toate secțiunile prezente:
- §1 Introducere: completă
- §2.1–2.5 Fundamente teoretice: complete cu ecuații
- §3.1–3.4 Model numeric: complet (arhitectură, geometrie, algoritm + pseudocod, teste)
- §4.1–4.6 Rezultate: complete cu figuri reale și tabel J_loop cu date din simulări
- §5 Concluzii: complete, 5 concluzii numerotate

Simulări disponibile în `sim_out/`:
- `loop_maxwell_hetero_long`: 300k pași, γ=1, MaxwellDiffuse, în curs (~220k/300k pași)
- `loop_ou_hetero_long`: 300k pași, γ=1, OUBounce eterogen, FINALIZAT (32k tranziții)
- `loop_ou_uniform_long`: 300k pași, γ=1, OUBounce uniform, FINALIZAT (32k tranziții)
- `loop_maxwell_ballistic`, `loop_ou_hetero_ballistic`, `loop_elastic_ballistic`: γ=0.01, regim tranzitoriu

## RESTRICȚIE LaTeX — IMPORTANT
Schimbările de fraze întregi sau paragrafe în `latex/main.tex` necesită confirmare explicită din partea utilizatorului înainte de execuție. Modificările de date numerice (tabele, valori), referințe și structuri tehnice minore sunt permise fără confirmare.

## Validare cod
- `<v²>` = 3.016 la echilibru (țintă 3.000) ✓
- 25/25 teste pytest trec ✓
- Niciun OUT în loop_chambers ✓
- `brownian_sim/analysis/ness.py` integrat în pachet (funcții: load_counts, detailed_balance_metrics, loop_current, ness_verdict)
