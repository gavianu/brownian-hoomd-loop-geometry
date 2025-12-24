# brownian-hoomd-loop-geometry


Simulare **MPCD (SRD) + traceri** în HOOMD-blue pentru o geometrie modulară tip: `IN ⟶ funnels ⟶ MID ⟶ funnels ⟶ OUT` cu un **tub inferior** care leagă IN↔OUT. Scopul este observarea mișcării browniene „naturale” la **echilibru** (ΔT = 0), distribuții spațiale și „micro-drifturi” locale.


## Instalare


### Varianta recomandată (conda)
```bash
conda env create -f env/environment.yml
conda activate brownian-hoomd
```

### Examples
set OUT_DIR
python sim\analytic_langevin_termal_collission.py --gpu 0 --gpu-collide --n 30000 --steps 20000 --write-every 2000 --log-every 200