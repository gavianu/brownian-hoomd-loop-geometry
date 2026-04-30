# Current status

## Solid (susținut de cod)
- Repo git valid și structură de simulare activă.
- Existența unei familii de scripturi `analytic_langevin*` cu:
  - geometrie compusă;
  - parametri materiali pe frontiere;
  - termeni Langevin în volum;
  - output pentru analiză.
- Inventarul conservator din `docs/repo_inventory.md` este deja prezent.

## Probabil (dedus plauzibil)
- `sim/analytic_langevin.py` este scriptul principal probabil pentru baseline operațional.
- Rezultatele fizice depind de combinația geometrie + materiale + stocasticitate, nu doar de formă geometrică.

## Experimental
- Scripturi cu nume/adnotări de test (ex.: `gpt_shit.py`, `grok_shit1.py`, `test_gpu.py`, `sanity_gpu.py`).
- Variante multiple `analytic_langevin*` cu parametri diferiți, posibil pentru experimente de stabilitate/perf.

## De validat (de confirmat manual)
1. Scriptul canonic unic pentru raportare finală.
2. Definiția exactă a termalizării OU la perete în varianta declarată „finală”.
3. Convenția oficială de observabile și pipeline complet de reproducere.

## Ce lipsește pentru evaluare completă
- Audit comparativ controlat între variantele `analytic_langevin*`.
- Un protocol unic de rulare + set minim de teste de regresie fizică.
- Confirmare documentație externă (dacă există) pentru ancorare teoretică finală.

## Status fișiere LaTeX în workspace-ul curent
- Căutarea `.tex` în snapshot-ul actual nu a returnat fișiere.
- Prin urmare, orice referință la `main.tex`/lucrare este **context extern** și rămâne **de confirmat manual**.


## Knowledge & prompts (nou)
- Knowledge pack: `docs/domain_knowledge.md`, `docs/model_assumptions.md`, `docs/common_pitfalls.md`, `docs/experiments.md`, `docs/roadmap.md`.
- Prompturi reutilizabile: `prompts/continue_from_baseline.md`, `prompts/review_physics.md`, `prompts/review_simulation.md`.
