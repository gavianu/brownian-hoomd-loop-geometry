# brownian-hoomd-loop-geometry

## Scopul proiectului
Acest repository investighează mișcarea Browniană/Langevin a unor traceri într-o geometrie internă compusă (camere + segmente tip funnel + canale de retur), cu accent pe efectele combinate dintre geometrie, materialul frontierei și termalizarea stochastică.

## Rezumat fizic scurt
Modelul implementat tratează traceri ca gaz ideal diluat (fără interacțiuni tracer–tracer explicite în scripturile principale), cu dinamică Langevin în volum și coliziuni la perete controlate de parametri materiali (coeficient normal `e_n`, factor tangențial `beta_t`).

Ideea centrală este **gaz ideal în geometrie complexă cu frontiere materiale diferite**: comportamentul emergent nu este atribuit exclusiv formei geometrice, ci și legii de interacție cu peretele.

## Punct cheie de interpretare
- Geometria **nu** este singura sursă a fenomenelor observate.
- Materialele de frontieră (prin `e_n`, `beta_t`) sunt esențiale.
- Termenii stocastici Langevin au interpretare fizică de cuplaj termic în volum.
- Modelul de tip OU la perete este relevant pentru compatibilitate termică, dar nivelul exact al implementării trebuie verificat pe scripturile dedicate (vezi documentația din `docs/`).

## Structura repo (pe scurt)
- `sim/` — scripturi de simulare (Langevin/Brownian/MPCD), inclusiv variante CPU/GPU și versiuni experimentale.
- `docs/` — inventar și documentație conceptuală/tehnică.
- `ovito/` — pipeline de vizualizare/post-procesare.
- `ness_check.py`, `sim/ness_check.py` — verificări NESS/echilibru (de confirmat diferențele exacte de rol).

## Script principal probabil
Pentru snapshot-ul curent, candidatul principal rămâne `sim/analytic_langevin.py` (susținut de cod prin structură completă: geometrie, materiale pe piese, evoluție stochastică, coliziuni și output-uri de analiză).

## Rulare orientativă
Exemplu existent (din familia principală de scripturi):

```bash
python sim/analytic_langevin_termal_collission.py --gpu 0 --gpu-collide --n 30000 --steps 20000 --write-every 2000 --log-every 200
```

Notă: alegerea „scriptului canonic de producție” între variantele `analytic_langevin*` este încă de confirmat manual.

## Stare actuală
- Cartografierea conservatoare a fost deja făcută în `docs/repo_inventory.md`.
- Documentația de bază este extinsă în fișierele din `docs/`.
- Interpretările nesusținute explicit de cod sunt marcate `de confirmat manual`.

## Repere de baseline (obligatoriu)
- **Baseline logic: `052a205`**
- **Current documentation checkpoint: `d7f4ce5`**


## Knowledge pack pentru continuitate (nou)
- `docs/domain_knowledge.md` — context fizic reutilizabil pentru agenți noi.
- `docs/model_assumptions.md` — ipoteze explicite, delimitate pe nivel de certitudine.
- `docs/common_pitfalls.md` — capcane de interpretare de evitat.
- `docs/experiments.md` — experimente numerice recomandate pentru validare.
- `docs/roadmap.md` — pași următori ordonați, conservatori.
- `prompts/continue_from_baseline.md`, `prompts/review_physics.md`, `prompts/review_simulation.md` — prompturi reutilizabile pentru agenți.
