# Domain knowledge (reutilizabil pentru agenți)

## Esența proiectului
Acest repo modelează traceri într-o geometrie internă compusă, cu interes pe transport statistic rezultat din interacția dintre:
1. topologia geometrică;
2. proprietățile materiale ale frontierei;
3. termalizarea stochastică în volum (și posibil la perete, în variante dedicate).

Miza este separarea contribuțiilor fizice, nu doar obținerea de traiectorii numerice.

## Model fizic pe scurt
- Traceri tratați ca gaz ideal diluat (fără forțe tracer–tracer explicite în scripturile principale).
- Evoluție Langevin în volum: fricțiune + zgomot Gaussian cu semnificație de cuplaj termic efectiv.
- Coliziuni la perete cu descompunere normal/tangențial și coeficienți materiali locali (`e_n`, `beta_t`).

## De ce geometria singură nu este suficientă
Geometria fixează conectivitatea și restricțiile cinematice, dar aceeași geometrie poate produce răspuns statistic diferit dacă legea de impact la frontieră diferă. Reducerea fenomenelor numai la formă geometrică este incompletă fizic.

## De ce materialele frontierei contează
Parametrii de frontieră (`e_n`, `beta_t`) modifică transferul local de impuls la impact, deci afectează timpii de rezidență, probabilitățile de tranziție și fluxurile nete. În consecință, „peretele” este parte din modelul fizic, nu doar din discretizare.

## De ce termenii stocastici au interpretare fizică
Termenii stochastici nu sunt introduși doar pentru stabilitate numerică; ei reprezintă cuplajul cu un rezervor termic efectiv (model Langevin/OU). Interpretarea corectă cere separarea între:
- semnificație fizică (temperatură, disipare, fluctuații);
- implementare numerică (schema de integrare, pas de timp, random sampling).

## Ce trebuie reținut de orice agent nou
1. Distinge strict: teorie fizică vs implementare concretă vs post-procesare.
2. Nu trata `analytic_langevin*` ca echivalente fără audit comparativ.
3. Marchează explicit orice deducție nesusținută direct în cod cu `de confirmat manual`.
4. Nu confunda output-uri regenerate cu artefacte științifice finale validate.
5. Menține abordarea conservatoare: fără refactor structural până la confirmarea scriptului canonic.
