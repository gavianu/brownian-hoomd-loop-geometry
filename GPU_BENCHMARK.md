# GPU Benchmark — brownian_sim

Testează accelerarea GPU pentru kernelele vectorizate din simulator.
Durată estimată: **3–8 minute** pe o placă NVIDIA cu 16 GB VRAM.

## Cerințe

```bash
pip install cupy-cuda12x   # pentru CUDA 12.x (verifică cu: nvcc --version)
# sau
pip install cupy-cuda11x   # pentru CUDA 11.x
```

Verificare instalare:
```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

## Rulare benchmark

```bash
cd brownian-hoomd-loop-geometry
python _gpu_benchmark.py
```

## Ce testează

| Kernel | Descriere | Status GPU |
|---|---|---|
| `LangevinIntegrator.step_velocity` | Pas Euler-Maruyama pe viteze, N=100k | ✅ complet vectorizat |
| `OUBounce.bounce_batch` | Ricoșeu stocastic OU pe N particule | ✅ complet vectorizat |
| `MaxwellDiffuse.bounce_batch` | Ricoșeu Maxwell diffuse pe N particule | ✅ complet vectorizat |
| `snap_and_normal` (geometrie) | Detecție coliziuni + proiecție pe frontieră | ⏳ buclă Python, urmează |

## Output așteptat (exemplu NVIDIA RTX / A-series 16 GB)

```
GPU: NVIDIA GeForce RTX ...
     VRAM libera: ~15.x GB / 16.0 GB

── 1. LangevinIntegrator.step_velocity ──
  CPU: ~8.00 ms
  GPU: ~0.15 ms  →  speedup ~50x

── 2. OUBounce.bounce_batch ──
  CPU: ~12.00 ms
  GPU: ~0.30 ms  →  speedup ~40x

── 3. MaxwellDiffuse.bounce_batch ──
  CPU: ~14.00 ms
  GPU: ~0.35 ms  →  speedup ~40x
```

## Interpretare

Kernelele de fizică (Langevin + bounce) sunt complet vectorizate și
beneficiază direct de GPU. Bottleneck-ul curent este `_step_positions`
din `engine.py`, care are o buclă Python per-particulă pentru
snap+normal la geometrie. Vectorizarea acesteia (următorul pas) ar
aduce speedup-ul total al simulării la același ordin.

Pe CPU (Intel Xeon, N=30k): **31 ms/pas**, ~2.6 ore pentru 300k pași.
Cu GPU complet (estimat): **< 1 ms/pas**, ~5 minute pentru 300k pași.

## Scalare N

Complexitatea este liniară în N pentru kernelele vectorizate:

| N | CPU estimat (300k pași) | GPU estimat (300k pași) |
|---|---|---|
| 10,000 | ~0.9 ore | ~2 min |
| 30,000 | ~2.6 ore | ~5 min |
| 100,000 | ~8.7 ore | ~15 min |
| 1,000,000 | ~87 ore | ~2.5 ore* |

*la N=1M transferul CPU↔GPU devine bottleneck dacă nu se păstrează
tot pe GPU.
