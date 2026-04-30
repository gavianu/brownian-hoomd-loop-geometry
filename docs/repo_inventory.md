# Repo inventory & baseline (etapa 1: cartografiere conservatoare)

## 1) Starea repository-ului
- Repository-ul este git valid (`git rev-parse --is-inside-work-tree` => `true`).
- Ramura curentă: `work`.
- Working tree: curat la momentul inspecției inițiale (`git status --short --branch` fără fișiere modificate).
- Există istoric de commit-uri; ultimele commit-uri indică focus pe geometrii și simulări Langevin/MPCD.
- **Baseline actual (probabil):** commitul HEAD `052a205` („Add geometry generation and simulation scripts for Brownian tracers and MPCD”).

## 2) Structura repo-ului (grupare funcțională)

### A. Simulare principală (candidate)
- `sim/analytic_langevin.py` — simulare analitică Langevin cu geometrii compuse (box/cylx/cyly), coliziuni la perete, coeficienți normali/tangențiali pe „materiale” de perete, output extins.
- `sim/analytic_langevin_termal_collission.py` — candidat avansat pentru coliziune/termalizare la perete (de confirmat prin rulare + audit complet).
- `sim/analytic_langevin_stabil_gpu_cpu.py` și `sim/analytic_langevin_stabil_gpu_cpu_simple_geom.py` — variante stabilizate CPU/GPU.

### B. Auxiliare geometrie / infrastructură simulare
- `sim/loop_geometry.py` — helper pentru geometrie MPCD/loop.
- `sim/proper_geometry.py` — alternativă/helper geometrie.
- `sim/geometry_export.py` — export geometrie pentru vizualizare/analiză.
- `sim/run_geometry_brownian.py` — variantă explicită „wall beads + Brownian tracers” (fără solvent MPCD).

### C. Analiză / post-procesare
- `sim/analysis_post.py` — pipeline post-procesare (MSD, tranziții, rate, wall hist etc.).
- `sim/analyze.py` — analiză local drift/densitate/MSD (script scurt, pare nefinisat la indentare).
- `sim/ness_check.py` și `ness_check.py` — verificări NESS/equilibru (de confirmat exact diferențele).
- `ovito/ovito_pipeline.py` — pipeline OVITO.

### D. Output-uri identificate
- Din cod: directoare/fișiere de output precum:
  - `sim/out_langevin_8/` (în `analytic_langevin.py`),
  - `sim/out1/` (în `run_geometry_brownian.py`),
  - `sim/out/` și `out/` (în scripturi MPCD/analiză),
  - fișiere tip `run.gsd`, `run.xyz`, `piece_counts.csv`, `transitions.csv`, `wall_hist_step*.csv`, `msd.csv`.
- **Observație:** în inventarul actual nu apar output-uri versionate în git (doar căi referite în cod).

### E. Documentație
- `README.md` — descriere proiect + exemplu de rulare.
- `LICENSE`.

### F. Legacy / experimental (candidate)
- `sim/gpt_shit.py`, `sim/grok_shit1.py` — nume sugerează experimente/ad-hoc.
- `sim/test_gpu.py`, `sim/sanity_gpu.py` — teste/sanity checks.
- `sim/run_mpcd_light.py` — minimal demo MPCD.
- `sim/analytic_brownian.py`, `sim/analytic_brownian_gpu.py` — linii alternative mai vechi față de familia `analytic_langevin*` (de confirmat cronologic).
- `sim/analytic_langevin_equil_gpu` — fișier fără extensie, probabil variantă script (de confirmat).

## 3) Fișier principal probabil al simulării

### Propunere principală
- **`sim/analytic_langevin.py`** este candidatul principal, deoarece:
  1. are configurație completă (geometrie, materiale, integrator Langevin, coliziuni, logging);
  2. definește explicit proprietăți materiale per componentă de frontieră (`e_n`, `beta_t`);
  3. produce output-uri detaliate pentru analiză (GSD/XYZ/counts/transitions/histograme perete);
  4. în README exemplul de rulare indică aceeași familie de scripturi (`analytic_langevin_*`).

### Dependențe principale ale acestui script
- Biblioteci: `numpy`, opțional `cupy`, opțional `gsd`.
- Ecosistem intern: geometrii și coliziuni implementate direct în fișier (dependență internă minimă pe alte module locale).
- Pentru analiză ulterioară: `sim/analysis_post.py`, `ovito/ovito_pipeline.py`.

## 4) Lucrarea LaTeX (`main.tex`) și extragerea modelului fizic

- Căutarea în repo nu a găsit fișiere `.tex` și nici `main.tex`.
- Consecință: **nu se poate extrage direct** din lucrare în această etapă, din conținutul curent al repository-ului.
- Marcaj: **de confirmat manual** (posibil fișier nelivrat în repo, alt branch, submodul sau arhivă externă).

## 5) Model fizic dedus din cod (ce e sigur vs de confirmat)

### Aspecte fizice sigure (din cod)
1. **Particule tracer / gaz ideal diluat (model efectiv):** nu apare interacțiune explicită tracer-tracer în `analytic_langevin.py`; dinamica dominantă este stocastică + coliziuni cu frontiere.
2. **Geometrie complexă modulară:** uniune de volume (cutii + cilindri) pentru camere/funnel/retur.
3. **Frontiere materiale eterogene:** fiecare piesă are parametri `e_n` (normal) și `beta_t` (tangențial), deci ricoșeu generalizat anisotrop material.
4. **Mișcare liberă între interacții + zgomot termic/Langevin în volum:** integratorul include termeni tip Langevin (fricțiune + zgomot), iar coliziunile la perete sunt tratate separat.
5. **Coliziuni perete explicite:** proiecție pe frontieră + actualizare viteză prin descompunere normal/tangențial.

### Aspecte „de confirmat”
1. **Frontieră termalizantă de tip OU la perete**: contextul fizic cere acest mecanism, dar în această etapă nu pot afirma ferm din `analytic_langevin.py` că implementarea este OU completă la impact; trebuie audit țintit în `analytic_langevin_termal_collission.py` și/sau documentația lipsă.
2. **Observabilele exacte din lucrare**: fără `main.tex`, setul de observabile „oficiale” rămâne de confirmat.
3. **Scriptul canonic „de producție”**: candidatul principal e clar (`analytic_langevin.py`), dar selecția finală între variantele `analytic_langevin_*` trebuie confirmată prin istoricul de execuții/notes.

## 6) Pași recomandați pentru etapa următoare (fără cleanup agresiv)
1. Confirmare sursă `main.tex` (locație/branch/submodul).
2. Audit comparativ strict între `analytic_langevin.py` și `analytic_langevin_termal_collission.py` pentru mecanismul de termalizare la perete (OU vs reflecție inelastică simplă).
3. Stabilirea scriptului „single source of truth” pentru rulări baseline.
4. Rulare scurtă de sanity cu output minim, apoi verificare automată a observabilelor de bază.
