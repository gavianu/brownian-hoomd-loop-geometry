# CLAUDE.md

## Scopul proiectului
Acest proiect modelează mișcarea Browniană/Langevin a tracerilor într-o geometrie internă complexă, pentru a înțelege statisticile de transport și rezidență când frontierele au materiale diferite.

## Modelul fizic
- Gaz ideal diluat (fără interacțiuni tracer–tracer explicite în scripturile principale).
- Evoluție Langevin în volum (fricțiune + zgomot termic).
- Frontiere descrise simultan geometric și material (`e_n`, `beta_t`).
- Ricoșeu cu descompunere normal/tangențial.
- Model OU la perete: folosit pentru compatibilitate termică la impact (de validat pe varianta canonică).

## Script principal probabil
- `sim/analytic_langevin.py` (susținut de cod).
- Variantele `sim/analytic_langevin_termal_collission.py` și `sim/analytic_langevin_stabil_gpu_cpu.py` trebuie comparate înainte de orice concluzie finală.

## Ce NU trebuie schimbat fără confirmare
1. Semantica parametrilor materiali de frontieră (`e_n`, `beta_t`).
2. Topologia geometriei de bază folosită în baseline.
3. Interpretarea termenilor stocastici ca elemente fizice (nu zgomot arbitrar).
4. Delimitarea baseline/checkpoint din documentație.

## Separare obligatorie în lucru/documentație
- **Teorie fizică**: ipoteze, ecuații, mecanisme de transfer impuls/energie.
- **Implementare numerică**: discretizare, integrator, coliziuni, I/O.
- **Analiză rezultate**: metrici, histograme, tranziții, validări statistice.
- **Output-uri**: fișiere brute generate, artefacte vizuale, rapoarte.

## Interpretări greșite de evitat
- „Doar geometria explică fenomenul.” (incorect/incomplet)
- „OU la perete este doar zgomot arbitrar.” (incorect)
- „Coeficienții de frontieră sunt detalii numerice fără semnificație fizică.” (incorect)

## Constrângeri de baseline
- Baseline logic de referință: `052a205`.
- Checkpoint curent documentație (cu inventar): `d7f4ce5`.
- Orice deviere de la aceste repere trebuie menționată explicit.

## Principii de interpretare
- Geometria nu este singura explicație a efectelor observate.
- Materialele frontierei sunt esențiale pentru transferul de impuls.
- Termenii stocastici au interpretare fizică (termostatare/cuplaj termic).
- Modelul OU la perete susține compatibilitatea termică locală.
