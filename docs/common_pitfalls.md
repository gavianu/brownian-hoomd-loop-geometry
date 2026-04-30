# Common pitfalls

## Interpretări greșite de evitat

1. **Reducerea întregii fizici la geometrie.**
   Geometria este necesară, dar insuficientă; condițiile materiale la perete schimbă statisticile de transport.

2. **Tratarea coeficienților de frontieră ca artificii numerice pure.**
   `e_n` și `beta_t` au rol fizic (transfer de impuls), nu doar rol de „tuning” numeric.

3. **Interpretarea greșită a termenului OU.**
   OU/thermal collision trebuie citit ca model de termalizare/cuplaj termic; detaliile implementării rămân `de confirmat manual` în scriptul canonic.

4. **Confundarea modelului fizic cu schema numerică.**
   Faptul că un integrator produce rezultate stabile nu implică validitate fizică automată.

5. **Confundarea scriptului principal probabil cu „adevăr definitiv”.**
   `sim/analytic_langevin.py` este candidat principal, nu verdict final, până la validare comparativă.

6. **Amestecarea liniilor experimentale cu fluxul baseline.**
   Scripturile legacy/experimental trebuie tratate separat de pipeline-ul de evaluare.

7. **Ignorarea trasabilității output-urilor.**
   Fișierele generate trebuie legate de parametri, commit și protocolul de rulare.
