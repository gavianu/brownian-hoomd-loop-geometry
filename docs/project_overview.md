# Project overview

## Descriere generală
Proiectul urmărește simularea și analiza mișcării particulelor tracer într-o topologie internă compusă, cu segmente conectate și proprietăți de frontieră neomogene. Ținta este separarea contribuțiilor din:
1. geometrie;
2. materialul pereților;
3. termalizare stochastică în volum și la interfață.

## Problema fizică
Se studiază cum apare transportul local/statistic într-un ansamblu fără interacțiuni tracer–tracer explicite, unde transferul de impuls este dominat de:
- cuplajul termic Langevin din volum;
- coliziuni cu pereți având legi materiale distincte pe segmente.

## De ce este util modelul de gaz ideal
Modelul de gaz ideal diluat reduce complexitatea și permite izolarea mecanismelor de frontieră și de geometrie, fără confuzia introdusă de interacții multe-corp între traceri.

## Rolul geometriei
Geometria definește conectivitatea domeniului, zonele de confinare și secțiunile locale de trecere (funnel/canal/cameră), deci modifică statisticile de rezidență și tranziție.

## Rolul frontierelor materiale
Frontierele au parametri locali (`e_n`, `beta_t`) pe fiecare piesă geometrică; acestea controlează redistribuția impulsului normal/tangențial la impact. Din acest motiv, două geometrii identice pot produce dinamică diferită dacă materialele peretelui diferă.

## Ce pare deja implementat (susținut de cod)
- Geometrie compusă din `box`, `cylx`, `cyly` cu mapare pe piese.
- Dinamică Langevin în volum (`GAMMA`, `KT`, zgomot Gaussian).
- Coliziuni la perete cu descompunere normal/tangențial și coeficienți materiali pe piesă.
- Output pentru analiză: traiectorii + contoare/tranziții/histograme la perete (în scripturile principale).

## Ce rămâne de confirmat
- Care variantă `analytic_langevin*` este fluxul canonic final de producție.
- În ce măsură modelul de perete implementat în varianta „termal collision” este OU complet vs aproximare echivalentă.
- Setul „oficial” de observabile din lucrare (dacă există documentație externă LaTeX neversionată aici).
