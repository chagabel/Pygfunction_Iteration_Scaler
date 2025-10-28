
# Lastprofil – Herkunft, Aufbau, Anpassung

> **Kurzfassung:** Unsere Module arbeiten mit stündlichen Lastprofilen (8760 h) und optionaler solarer Rückspeisung. Standardmäßig lesen sie die Daten aus der mitgelieferten Excel-Arbeitsmappe `Wärmelastprofil2.xlsx`. Die Profile werden für **GHEtool-Exporte** und den **Longterm‑Scaler** verwendet (siehe Kapitel „Longterm‑Scaler“).

## Das Lastprofil (Überblick)

- **Gebäudelast (8760 h):** Im Beispiel aus dem Excel‑Tool „Synthese von Wärmelastprofilen nach BDEW“ (Hochschule Trier).
- **Solare Einstrahlung (W/m²):** Im Beispiel aus Renewables.ninja für **2019, Hardegsen (Niedersachsen, DE)**.
- Aus diesen Eingaben erzeugt die Arbeitsmappe im Blatt **„Ergebnis“** fertige Zeitreihen:
  - **Spalte B:** Nettolast **mit** Rückspeisung (W)
  - **Spalte C:** Bruttolast **ohne** Rückspeisung (W)
  - **Spalte D:** Rückspeisung‑ONLY (W)

> Die Hardegsen‑Daten sind projektspezifisch (Herkunft des Codes). Für andere Projekte lässt sich die Mappe schnell anpassen (siehe unten).

## Grundsätzliche Funktion & Bedienung

- **excel_to_qprime.py** → erzeugt **q′(t) [W/m]** aus Excel:
  - *Mit Rückspeisung* = liest **Ergebnis!B**
  - *Ohne Rückspeisung* = liest **Ergebnis!C**
  - rechnet intern **W → W/m** mit `L_total = n_bh · H` (Feldlänge).
- **excel_to_GHE3.py** → erzeugt **GHEtool‑CSV (hourly)**:
  - *Mit Rückspeisung*: **Extraction = C**, **Injection = D**
  - *Ohne Rückspeisung*: **Extraction = C**, **Injection = 0**
- **longterm_profile_scaler_v13.py** und **GHEtool_Dimensionierer_v4.py** nutzen die erzeugten Zeitreihen/Dateien weiter (Details in den jeweiligen Kapiteln).

> **Wichtig:** Entweder **B** (Netto) **oder** **C/D** (saubere Trennung) verwenden – nicht beides gleichzeitig, sonst Doppelzählung der Rückspeisung.

## XLS – Herkunft (Projekt Hardegsen) & Anpassung für eigene Projekte

Die mitgelieferte `Wärmelastprofil2.xlsx` ist ein **Projekt‑Beispiel** (Hardegsen). So passt du sie an:

### 1) Eigene Gebäudelast eintragen
- Blatt **„Umrechnung“ → Spalte B** (8760 Werte): stündliche Gebäudelast (kWh).
  - Falls bereits **W** oder **kW** vorliegen: direkt in der entsprechenden Umrechnungszelle anpassen. Wichtig ist, dass auf **„Ergebnis“** in **C** die stündliche **Bruttolast in W** entsteht.

### 2) Eigene Solarstrahlung & Parameter setzen *(optional)*
- **„Umrechnung“ → Spalte I**: stündliche Solarstrahlung **W/m²** (z. B. via Renewables.ninja für Ort/Jahr deines Projekts).
- **Zeile 1 (Eingaben):**
  - **Thermischer Wirkungsgrad** `η_th` (z. B. 0,30)
  - **Einspeisefaktor** `η_inj` (z. B. 0,75)
  - **Kollektorfläche** `A_coll` [m²]
- Daraus berechnet die Mappe **Rückspeisung‑ONLY** (Ergebnis **D**) und **Nettolast** (Ergebnis **B**).

### 3) Check & Export
- **8760** Zeilen prüfen; **C ≥ 0**, **D ≥ 0**, **B** kann ± sein.
- In den Tools den passenden Modus wählen (siehe oben).

## Alternative ohne Excel: synthetisches Profil
Für schnelle, generische Tests steht **`synthetic_heating_profile.py`** bereit (Heizen‑only, „nach Bernier“, **ohne** Rückspeisung). So erhältst du ein sauberes 8760‑Profil ohne Solar‑Teil – ideal für Vortests, Sensitivitäten und Debugging.

## Einheiten & Konventionen
- **Leistung:** W bzw. kW
- **Energie:** kWh, MWh (Aggregat)
- **Solarstrahlung:** W/m²
- **Linienlast:** q′(t) in **W/m** (Umrechnung W → W/m erfolgt in den Skripten)
- **Feldlänge:** `L_total = n_bh · H` (m)
- **Vorzeichen:** Entzug = **positiv** (C), Einspeisung = **positiv** (D), Netto (B) kann ± sein

## Häufige Stolpersteine
- **Doppelzählung:** B (Netto) **und** D (Injection) gleichzeitig an GHEtool übergeben → zu hohe Einspeisung.
- **8760‑Lücken:** Fehlende Stunden oder falsche Zeitzonen‑Verschiebung führen zu Inkonsistenzen.
- **Einheitenmix:** kWh ↔ W nicht sauber getrennt → falsche Skalierung.

## Weiterentwicklungsideen
- Optionales Python‑Modul zum **Direktimport** der stündlichen Solarstrahlung (API/CSV) und zur **Validierung** (Ampel‑Check: 8760, Vorzeichen, Jahressummen).
- Parametrisierte Profile für **Warmwasser** oder **Kühlbetrieb** (separate Spalten/Dateien).

---

**Verweise**
- BDEW‑basierte Profil‑Synthese (Excel) – Hochschule Trier
- Renewables.ninja – stündliche Solar‑/Winddaten (CSV‑Download)
