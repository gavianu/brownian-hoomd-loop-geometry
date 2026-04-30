# brownian-hoomd-loop-geometry

Simulare Brownian/Langevin a unui gaz ideal în geometrii de confinare complexe, cu pereți având proprietăți materiale și termice diferite. Cod de bază pentru lucrarea de licență (UB Fizică, iunie 2026).

## Structură

```
brownian_sim/                   # pachetul Python (SOLID, extensibil)
├── geometry/                   # primitive (Box, CylX, CylY) + Assembly
│   └── presets/                # geometrii predefinite (simple_box, loop_chambers)
├── materials/                  # WallMaterial (e_n, beta_t)
├── physics/                    # wall_models, Langevin, backend numpy/cupy
├── simulation/                 # engine, sampler, substepping
├── io/                         # writers GSD + CSV
├── analysis/                   # MSD, tranziții, echilibru, statistici perete
└── scripts/                    # CLI entry points

configs/                        # YAML config-uri (reproductibile)
tests/                          # pytest (25 teste)
latex/                          # lucrarea LaTeX
legacy/                         # scripturile originale, păstrate pentru referință
```

## Instalare

```bash
pip install -e .                # instalare editabilă
pip install -e .[dev]           # + pytest pentru dezvoltare
```

## Rulare simulări

Prin YAML config:

```bash
python -m brownian_sim.scripts.run_sim --config configs/baseline_simple_box.yaml
python -m brownian_sim.scripts.run_sim --config configs/loop_chambers_ou.yaml
```

Override pe parametri:

```bash
python -m brownian_sim.scripts.run_sim --config configs/loop_chambers_ou.yaml --steps 5000 --n 10000
```

## Testare

```bash
python -m pytest tests/
```

25 teste: geometrie (inside/wall_distance/snap), modele de perete (conservare energie elastic, MB la OU), Langevin + OU în echilibru.

## Modele fizice implementate

### Dinamică volum — Langevin
```
dv = -(γ/m) v dt + √(2γkT)/m · dW
```
Integrator Euler-Maruyama, parametri în YAML: `mass`, `gamma`, `kT`, `dt`.

### Modele de perete (interschimbabile prin config)

| Nume | Formulă | Folosire |
|------|---------|----------|
| `elastic` | `v' = v - 2(v·n)n` | reflexie speculară, conservă energia |
| `damped`  | `v_n' = -e_n v_n; v_t' = β_t v_t` | disipativ, fără zgomot termic |
| `ou`      | `v_n' = e_n\|v_n\| + √(1-e_n²)·s·ξ_n`, idem tangent | OU bounce, termalizant, FDT local |

Unde `s = √(kT/m)`, iar `ξ` sunt gaussieni standard.

## Extensibilitate

**Geometrie nouă:** scrie o clasă care moștenește `Primitive` în `brownian_sim/geometry/primitives.py`, sau un preset nou în `brownian_sim/geometry/presets/`.

**Wall model nou:** scrie o clasă care moștenește `WallModel` în `brownian_sim/physics/wall_models.py` și adaug-o în `make_wall_model()`.

Engine-ul și restul codului nu se schimbă.

## Context fizic

Modelul pornește de la o întrebare veche: poate geometria + materiale asimetrice să producă NESS (dezechilibru staționar) într-un gaz ideal? Răspuns final (coerent cu principiul 2): **nu**, dacă pereții sunt termalizanți corect (OU bounce) sistemul relaxează către echilibru. NESS real necesită dezechilibru extern (al 2-lea termostat sau sistem deschis).

## Rezultate validare (refactor vs legacy)

- `<v²>` la echilibru = 3.016 (țintă 3.000 pentru kT=m=1)  ✓
- Niciun drift sistematic per-axă
- Niciun OUT (particule pierdute) în loop_chambers
- 62× mai rapid decât legacy CPU la aceiași parametri

## Legacy

Scripturile originale din `sim/` sunt păstrate în `legacy/sim_scripts/` ca referință istorică. Vezi `legacy/README.md` pentru mapare pe script.
