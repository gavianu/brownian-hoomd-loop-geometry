# Experiments (plan numeric recomandat)

## Obiectiv
Definirea unui set minim de experimente pentru validare fizică și consistență internă, fără schimbări structurale de cod.

## 1) Caz simplu de referință
- Geometrie simplificată (unde există variantă „simple_geom”).
- Parametri moderați (pas de timp/stocasticitate) pentru stabilitate.
- Scop: verificare de bază a pipeline-ului și a observabilelor (MSD, tranziții, distribuții la perete).

**Status repo:** parțial posibil direct (există scripturi simple). Configurația canonică exactă rămâne `de confirmat manual`.

## 2) Geometrii complexe
- Rulare pe geometria compusă (camere/funnel/canal retur).
- Comparare a distribuțiilor de rezidență pe segmente.

**Status repo:** posibil direct în familia `analytic_langevin*`.

## 3) Relaxare din stare inițială neuniformă
- Inițializare concentrată într-o sub-zonă.
- Urmărire relaxare către regim staționar/NESS cu metrici temporale.

**Status repo:** plauzibil posibil; protocolul exact de inițializare este `de confirmat manual`.

## 4) Pereți elastici vs frontiere termalizante
- Set A: coliziuni elastice/inerte (sau aproape elastice).
- Set B: coliziuni cu termalizare (varianta thermal/OU).
- Scop: izolarea efectului condiției la frontieră la geometrie fixă.

**Status repo:** posibil conceptual; maparea exactă a seturilor de parametri este `de confirmat manual`.

## Observabile de urmărit
- MSD pe ferestre temporale.
- Rate/probabilități de tranziție între piese.
- Timp de rezidență per segment geometric.
- Histograme de distanță la perete / frecvență de impact.
- (Opțional) indicatori simpli de staționaritate.

## Ce este deja posibil vs ce rămâne de confirmat

### Deja posibil în repo
- Generare traiectorii și output-uri CSV/GSD/XYZ.
- Post-procesare de bază pentru MSD/tranziții/histograme.

### De confirmat manual
- Scriptul unic folosit ca referință oficială.
- Setul oficial de parametri și seed-uri pentru reproducere.
- Criteriul final de acceptare pentru consistență fizică.
