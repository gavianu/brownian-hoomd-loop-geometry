# Roadmap (conservator)

## Principiu
Nicio curățare structurală agresivă înaintea validării scriptului canonic și a protocolului numeric.

## Pași ordonați
1. **Validarea scriptului principal canonic.**
   Decide explicit ce script din familia `analytic_langevin*` este „source of truth” (`de confirmat manual`).

2. **Validarea parametrilor baseline.**
   Fixează set minim: `DT`, `GAMMA`, `KT`, condiții la frontieră, seed-uri, lungime rulare.

3. **Rerularea cazului simplu de referință.**
   Verifică stabilitate numerică + consistența observabilelor de bază.

4. **Definirea experimentelor comparative.**
   Geometrie fixă, frontiere variabile; apoi geometrie variabilă, frontiere fixate.

5. **Clarificarea output-urilor importante.**
   Marchează ce fișiere sunt rezultate primare vs derivate regenerabile.

6. **Aliniere cu lucrare LaTeX externă (dacă există).**
   Mapare ecuații–parametri–observabile; tot ce lipsește rămâne `de confirmat manual`.

7. **Curățare structurală ulterioară (opțional).**
   Doar după confirmare: separare clară între baseline, experimental și legacy.
