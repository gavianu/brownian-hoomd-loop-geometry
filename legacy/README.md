# Legacy scripts

Scripturile originale din `sim/`, păstrate ca referință istorică și pentru validarea refactor-ului.

## Rol pe script

| Fișier | Rol istoric | Stare |
|--------|-------------|-------|
| `analytic_langevin.py` | **Prima implementare funcțională** — are `reflect_cpu` complet. Folosit ca baseline. | CPU OK |
| `analytic_langevin_termal_collission.py` | Varianta cu **OU bounce** (model termalizant la perete). Modelul fizic final. | GPU-only (lipsește `reflect_cpu`) |
| `analytic_langevin_stabil_gpu_cpu.py` | Versiune intermediară cu sub-stepping CFL. | GPU-only |
| `analytic_langevin_stabil_gpu_cpu_simple_geom.py` | Aceeași ca mai sus, geometrie simplificată pt. test. | GPU-only |
| `analytic_brownian.py`, `analytic_brownian_gpu.py` | Varianta Brownian (fără Langevin). | Referință istorică |
| `run_geometry_brownian.py` | Try-out cu HOOMD + wall beads. Abandonat. | Abandonat |
| `run_mpcd*.py` | Experiment MPCD, niciodată finalizat. | Abandonat |
| `gpt_shit.py`, `grok_shit1.py` | Experimente LLM ad-hoc. | Abandonat |
| `sanity_gpu.py`, `test_gpu.py` | Verificări CuPy. | Utilitar |
| `analysis_post.py`, `analyze.py`, `ness_check.py` | Post-procesare. Logica a fost portată în `brownian_sim/analysis/`. | Portat |
| `geometry_export.py`, `loop_geometry.py`, `proper_geometry.py` | Helper-e de geometrie. Logica a fost portată în `brownian_sim/geometry/`. | Portat |

## De ce au fost înlocuite

Toate au cod duplicat (geometrie + integrator + I/O amestecate în același fișier, ~1000 linii fiecare).
Refactor-ul în `brownian_sim/` separă aceste responsabilități și permite:

- rulare CPU identică cu GPU (același cod);
- adăugare geometrie nouă fără a atinge engine-ul;
- adăugare model de perete nou (alt OU, alt damped) fără a atinge restul;
- teste unitare pe părțile fizice critice.
