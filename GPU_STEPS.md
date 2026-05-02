# GPU Steps — setup, benchmark, simulari, rezultate

Instructiuni complete pentru calculatorul cu CUDA (NVIDIA 16 GB).
Durata totala estimata: ~30-60 minute (din care ~25 min simulari).

---

## 1. Setup mediu

```bash
git clone https://github.com/gavianu/brownian-hoomd-loop-geometry.git
cd brownian-hoomd-loop-geometry
pip install -r requirements.txt

# instaleaza CuPy pentru versiunea CUDA instalata
nvcc --version                  # afiseaza versiunea CUDA (ex. 12.x sau 11.x)
pip install cupy-cuda12x        # pentru CUDA 12.x
# sau
pip install cupy-cuda11x        # pentru CUDA 11.x

# verifica CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

---

## 2. Benchmark kernele GPU (~5 minute)

```bash
python _gpu_benchmark.py
```

Testeaza 3 kernele vectorizate (N=100k, 500 pasi):
- `LangevinIntegrator.step_velocity`
- `OUBounce.bounce_batch`
- `MaxwellDiffuse.bounce_batch`

Salveaza output-ul (copy-paste) — il putem pune in teza ca date reale.

---

## 3. Simulari noi (~25 minute total)

Ruleaza in doua terminale separate (sau secvential):

**Terminal 1 — Maxwell diffuse, 100k pasi:**
```bash
python -m brownian_sim.scripts.run_sim --config configs/loop_maxwell_gpu_100k.yaml
```
Estimat: ~15 minute pe CPU, ~2-3 minute pe GPU (dupa vectorizare geometrie).
Output: `sim_out/loop_maxwell_gpu_100k/`

**Terminal 2 — OUBounce, 50k pasi:**
```bash
python -m brownian_sim.scripts.run_sim --config configs/loop_ou_gpu_50k.yaml
```
Estimat: ~8 minute pe CPU, ~1-2 minute pe GPU.
Output: `sim_out/loop_ou_gpu_50k/`

---

## 4. Verifica NESS dupa simulari

```bash
python _quick_ness.py
```

Verdict asteptat:
- `loop_maxwell_gpu_100k`: J_loop ~ 10^-3 sau mai mic, `echilibru`
- `loop_ou_gpu_50k`: J_loop ~ 10^-3, `echilibru`

Daca J_loop scade fata de rularea scurta (2.5k tranzitii) => confirma convergenta spre echilibru.

---

## 5. Colecteaza rezultatele pentru commit

```bash
python _collect_results.py
```

Copiaza `piece_counts.csv`, `transitions.csv` si calculeaza `ness_summary.csv`
din toate simulările in `results/`.

Verifica ce s-a creat:
```bash
ls results/
cat results/ness_summary.csv
```

---

## 6. Commit si push rezultate

```bash
git add results/ _collect_results.py .gitignore
git commit -m "Adauga rezultate simulari GPU: maxwell 100k, ou 50k"
git push origin main
```

---

## Ce ne uitam in rezultate

| Metrica | Maxwell 100k | OU 50k | Interpretare |
|---|---|---|---|
| `N_tr` (tranzitii) | ~50k | ~16k | mai multe = statistici mai bune |
| `J_loop` | < 5e-3 | < 5e-3 | zgomot statistic, nu curent real |
| `Rmax` | < 5e-4 | < 5e-4 | bilanț detaliat respectat |
| `verdict` | echilibru | echilibru | confirma fizica corecta |

Daca `J_loop` la maxwell_100k e mai mic decat la maxwell_hetero_long (300k pasi, 17k tr):
=> mai multe tranzitii bat zgomotul mai repede pe GPU.

---

## Nota despre GPU si engine

Kernelele de fizica (Langevin + bounce_batch) sunt complet vectorizate si
ruleaza pe GPU prin CuPy. Bottleneck-ul curent este bucla de detecție
coliziuni din `engine.py` (`_step_positions` — snap+normal per-particula).
Vectorizarea acesteia este urmatorul pas si ar aduce speedup total ~50x.

Speedup real masurat pe kernele izolate va fi vizibil in `_gpu_benchmark.py`.
