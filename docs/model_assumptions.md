# Model assumptions

## Definiții operaționale (în acest repo)

### Ce înseamnă aici „gaz ideal”
Set de traceri fără interacțiuni explicite tracer–tracer în scripturile principale; dinamica este guvernată de termeni Langevin în volum și de condiții la frontieră.

### Ce este „dinamică în volum”
Evoluția pozițiilor/vitezelor între coliziuni cu frontiera, incluzând fricțiune și zgomot termic conform schemei Langevin.

### Ce este „condiție la frontieră”
Regula de actualizare la impact cu pereții (decompoziție normal/tangențial + parametri materiali per piesă). În unele variante apare și termalizare de tip OU/thermal collision.

## Ipoteze explicite ale modelului
- Mediul este reprezentat efectiv prin termeni de zgomot și disipare (nu prin solvent molecular explicit în fluxul principal).
- Transferul de impuls la perete este parametrizat local pe piese geometrice.
- Observabilele principale sunt statistice (tranziții, rezidență, histograme, MSD), nu traiectorii individuale izolate.

## Susținut de cod
1. Absența interacțiunilor tracer–tracer explicite în familia principală `analytic_langevin*`.
2. Existența parametrilor de volum tip `DT`, `MASS`, `GAMMA`, `KT`.
3. Existența coeficienților de frontieră (`e_n`, `beta_t`) per piesă.
4. Existența exportului de observabile pentru analiză (CSV/GSD/XYZ, în funcție de script).

## Deduc plauzibil
1. Geometria singură nu explică complet transportul; materialele frontierei au contribuție majoră.
2. Variantele scripturilor `analytic_langevin*` pot reprezenta ipoteze fizice diferite, nu doar optimizări tehnice.
3. Scriptul `sim/analytic_langevin.py` este candidat principal de baseline operațional.

## De confirmat manual
1. Definiția exactă și validarea statistică a modelului OU la perete în varianta declarată finală.
2. Scriptul canonic unic pentru rezultate publicabile.
3. Corelarea strictă cu o referință teoretică externă (ex. lucrare LaTeX), dacă aceasta nu este versionată aici.
