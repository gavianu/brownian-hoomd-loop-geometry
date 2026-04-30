# Model fizic (formulare explicită)

## 1) Obiectul modelat
Se modelează traceri într-un domeniu 3D compus, tratați ca gaz ideal diluat.

- **Susținut de cod:** în scripturile principale nu apare interacțiune explicită tracer–tracer; dinamica este individuală + condiții de frontieră.
- **Deducție plauzibilă:** regimul urmărit este unul în care efectele colective între traceri sunt neglijate intenționat.

## 2) Dinamica în volum
Între interacții cu pereții, particulele evoluează cu schemă Langevin (fricțiune + zgomot termic Gaussian).

- **Susținut de cod:** parametri globali de tip `DT`, `MASS`, `GAMMA`, `KT` și termeni stochastici în scripturile `analytic_langevin*`.
- **Interpretare:** aceasta reprezintă cuplaj efectiv la un rezervor termic în volum.

## 3) Frontiera: componentă geometrică + materială + termică
Frontiera nu este doar contur geometric; fiecare piesă poartă parametri materiali pentru impact.

- **Geometric:** piese `box`, `cylx`, `cyly`, combinate într-o topologie conectată.
- **Material:** pentru fiecare piesă apar `e_n` (componenta normală la ricoșeu) și `beta_t` (componenta tangențială).
- **Termic:** există variante de script care indică explicit un model de tip „thermal collision”, inclusiv mențiuni OU.

## 4) Legea de ricoșeu la perete
La impact, viteza este descompusă în normală (`v_n`) și tangențială (`v_t`), apoi actualizată în forma:
- componenta normală scalată/reversată cu `e_n`;
- componenta tangențială scalată cu `beta_t`.

- **Susținut de cod:** actualizare explicită `v_ref = -e_n * v_n + beta_t * v_t` în logica de coliziune.
- **Consecință fizică:** transferul de impuls depinde local de materialul frontierei.

## 5) OU / termalizare la perete
Modelul OU la perete nu trebuie tratat ca zgomot arbitrar; rolul său este compatibilitatea termică a vitezelor după contact.

- **Susținut de cod:** există un script dedicat „thermal collision” și variante cu mențiuni `WALL_MODEL = "ou"`.
- **De confirmat manual:** forma exactă a operatorului OU la impact (inclusiv calibrare statistică și invarianta temperaturii țintă) necesită audit focalizat al versiunii folosite în producție.

## 6) Observabile și transfer de impuls
Observabilele de interes (contorizări pe piese, tranziții, histograme de distanță la perete, traiectorii) sunt legate de transferul local de impuls și de timpul de rezidență.

- **Susținut de cod:** scripturile principale exportă fișiere pentru tranziții și distribuții legate de perete.
- **Deducție plauzibilă:** asimetriile emergente pot proveni din combinația geometrie + materiale + termalizare, nu doar din geometrie.

## 7) Delimitare explicită

### Susținut de cod
1. Gaz ideal diluat (fără forțe tracer–tracer explicite).
2. Geometrie compusă modulară.
3. Coliziuni cu parametri materiali pe piesă (`e_n`, `beta_t`).
4. Dinamică Langevin în volum.
5. Output-uri orientate către analiză de tranziții/perete.

### Deducție plauzibilă
1. Materialele de frontieră controlează o parte majoră din transportul local.
2. Geometria singură nu explică complet rezultatele.
3. Configurațiile din variantele script pot produce regimuri calitative diferite.

### De confirmat manual
1. Scriptul unic „source of truth” pentru rezultate finale.
2. Definiția operațională exactă a modelului OU la perete în varianta validată.
3. Corelarea cu eventuale ecuații/observabile din documentație LaTeX externă.
