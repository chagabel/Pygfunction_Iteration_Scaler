
# Einführung

Dieses Projekt liefert ein **offenes Python‑Werkzeug** zur überschlägigen **Potenzialabschätzung und Dimensionierung** von Erdwärmesonden‑Feldern (BHE, Borehole Heat Exchangers). Es baut auf der Bibliothek **pygfunction** auf, ergänzt sie aber um praxisnahe Bausteine: **Stundenprofile**, **VDI‑Grenzkriterien**, **hydraulische Plausibilisierung des Massenstroms**, **Export/Validierung für GHEtool** sowie **Solar‑Rückspeisung** und **Szenario‑Automatisierung**.

> Ziel: Eine nachvollziehbare, reproduzierbare Vorstudie („Screening“) – **nicht** die Ausführungsplanung. Für die Planung gelten die einschlägigen Normen (z. B. VDI 4640) und geologische Standortgutachten.

---

## Was das Modul macht – in Kürze

1. **Geometrie & Feldaufbau**  
   Rechteck‑ oder U‑Felder aus Parametern **(H, Bx, By, Reihen/Spalten)** oder aus **GIS‑Polygonen** erzeugen (`borefields.py`, `auto_borefield.py`).

2. **Lastprofil verarbeiten (8760 h)**  
   - **Excel‑Pfad**: `Wärmelastprofil2.xlsx` → stündliche **Gebäudelast** und optionale **solare Rückspeisung** (Kapitel *Lastprofil*).  
   - **Alternativ**: synthetisches **Heiz‑Only‑Profil** ohne Rückspeisung (`synthetic_heating_profile.py`).  
   - Umrechnung nach **Linienlast**: \(q'(t)=Q(t)/(n_{bh}\cdot H)\).

3. **Thermische Antwort (pygfunction)**  
   Für das Feld werden **G‑Funktionen** bestimmt und daraus **Boden-** und **Fluidtemperaturen** über die Zeit berechnet – inkl. **Interferenz** innerhalb des Feldes.

4. **Grenzkriterien prüfen (VDI‑Logik)**  
   Standard: **Monatsmittel EWT ≥ 0 °C im Jahr 50**; optional andere Kriterien (−3/−5 °C, Bodenmittel, Mindest‑Tb).

5. **Hydraulische Plausibilität (Massenstrom)**  
   `massflow_limits_scanner.py` liefert **min/nom/max**‑Durchflussbänder aus **Re‑Zahl**, **Δp‑Budget** (Darcy–Weisbach/Haaland) und dem Effekt auf **R\_b,eff**. So bleibt der gewählte Massenstrom realistisch.

6. **Skalierung / Szenarien**  
   `longterm_profile_scaler_v13.py` skaliert die Feldlast (mit/ohne Rückspeisung), bis die **Grenze** erreicht ist; liefert **MWh/a‑Potenzial**, **q′(t)**, CSV/JSON‑Exporte. Varianten: **Tiefe fix**, **Bohrzahl variieren**, **Feldabstand ändern**, **Solar an/aus**.

7. **Validierung & Austausch**  
   - `excel_to_GHE3.py`: **hourly CSV** für **GHEtool** (Extraction/Injection getrennt).  
   - `GHEtool_Dimensionierer_v4.py`: GUI‑Brücke und Gegencheck.  
   - `hp_cop_analyzer.py`: optionale **COP/EWT‑Auswertung** über das Jahr.

---

## Wie es funktioniert – verständlich erklärt

- **Wärme entziehen & zurückspeisen:** Die Gebäudelast (W) wird stündlich auf die **Gesamtlänge** aller Sonden verteilt (\(n_{bh}\cdot H\)). So entsteht eine **Linienlast** \(q'(t)\) in **W/m**. Positive Werte bedeuten Entzug (Kälte im Boden), optionale **Solar‑Rückspeisung** wirkt als Gegenlast.  
- **Boden reagiert träge:** Mit **pygfunction** berechnen wir, wie stark der **Untergrund** und die **Bohrlochwand** auf die Last reagieren (**G‑Funktionen**, Superposition). Dichte Felder kühlen sich gegenseitig stärker ab → geringeres Potenzial je Sonde.  
- **Grenztemperaturen sichern:** Damit **Wärmepumpen** stabil und effizient arbeiten, setzen wir **Temperaturgrenzen** (z. B. EWT‑Monatsmittel ≥ 0 °C nach 50 Jahren). Der **Longterm‑Scaler** erhöht die Last so weit, bis die festgelegte Grenze **gerade noch** eingehalten ist. Das ergibt das **nutzbare Potenzial** unter diesen Randbedingungen.  
- **Hydraulik nicht vergessen:** Mehr Durchfluss verbessert den Wärmeübergang (kleineres **R\_b,eff**), kostet aber **Pumpenleistung** und erzeugt **Druckverluste**. Unser **Massenstrom‑Scanner** steckt einen **realistischen Korridor** ab; der Scaler nutzt daraus passende Werte, statt mit unrealistischen Durchflüssen zu rechnen.  
- **Validieren & austauschen:** Für einen zweiten Blick exportieren wir **hourly‑Profile** als **GHEtool‑CSV**. So lassen sich Ergebnisse **gegenseitig prüfen** und mit anderen Workflows teilen.

---

## Typische Workflows

**A) Schnelltest (ohne Projekt‑Excel)**  
1. `synthetic_heating_profile.py` ausführen → 8760‑Heizprofil erzeugen.  
2. Feld anlegen (`borefields.py`), Standard **6 m Abstand**, **100 m Tiefe**.  
3. `longterm_profile_scaler_v13.py` starten → **Potenzial** bei EWT‑Grenze ermitteln.  
4. Optional: `excel_to_GHE3.py` → CSV für GHEtool exportieren.

**B) Projektspezifisch (mit Excel & Solar)**  
1. `Wärmelastprofil2.xlsx` anpassen (Kapitel *Lastprofil*): Gebäudelast (B), Solar‑Strahlung, \(η_{th}\), \(η_{inj}\), Fläche.  
2. `excel_to_qprime.py`: q′(t) aus **Ergebnis!B** (mit) oder **C** (ohne Rückspeisung) erzeugen.  
3. `massflow_limits_scanner.py`: **min/nom/max** prüfen und wählen.  
4. `longterm_profile_scaler_v13.py`: Szenario laufen lassen (Tiefe/Abstand/Bohrzahl).  
5. Optional: `excel_to_GHE3.py` → **Extraction/Injection** an GHEtool übergeben.

---

## Stärken des Ansatzes (gegenüber „reinen“ Daumenwerten)

- **Zeitlich aufgelöst (8760 h)** statt statischer W/m‑Tabellen.  
- **Interferenz im Feld** durch G‑Funktionen statt pauschaler Abschläge.  
- **Explizite Grenzkriterien** (EWT‑Monatsmittel etc.) und **Hydraulik‑Check**.  
- **Transparenz**: Alle Schritte bleiben als Dateien/Parameter nachvollziehbar.

---

## Grenzen & Annahmen

- **Homogene Bodenschicht** (vereinfachte Mittelwerte), keine Grundwasser‑Advektion.  
- **Vertikale Sonden** (BHE); horizontale Kollektoren sind nicht Bestandteil.  
- **Hydraulik** als Plausibilisierung (Δp‑Budget, Re‑Zahl), **kein** vollständiger Netzstrang‑/Pumpenplaner.  
- **Vorstudien‑Werkzeug**: Ergebnisse dienen der Orientierung, **nicht** der Genehmigungs‑ oder Ausführungsplanung.

---

## Modul‑Überblick (Dateien)

- **Geometrie & BHE:** `borefields.py`, `borehole.py`, `auto_borefield.py`  
- **Profile & Exporte:** `Wärmelastprofil2.xlsx`, `excel_to_qprime.py`, `excel_to_GHE3.py`, `synthetic_heating_profile.py`  
- **Skalierung & Validierung:** `longterm_profile_scaler_v13.py`, `GHEtool_Dimensionierer_v4.py`, `vdi_grobcheck.py`  
- **Hydraulik & COP:** `massflow_limits_scanner.py`, `hp_cop_analyzer.py`  
- **Defaults & Projekt‑Daten:** `Data_Base.py`, `ANWENDUNG_START.py`

---

## Weiteres

- **Kapitel „Lastprofil“** beschreibt Herkunft, Aufbau und Anpassung der Excel.  
- **Kapitel „Longterm‑Scaler“** erklärt die Skalierlogik, Grenzkriterien und Ausgaben.  
- **Kapitel „Hydraulik“** fasst die Massenstrom‑Scanner‑Logik (Re, Δp, R\_b,eff) zusammen.

