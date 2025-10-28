# Beispiel-Workflow (Hardegsen)

Dieses Kapitel führt exemplarisch durch einen typischen Ablauf anhand der beiliegenden Arbeitsmappe `Wärmelastprofil2.xlsx` (Projekt Hardegsen). Die Werte dienen als Beispiel; für andere Projekte Excel/Parameter entsprechend anpassen.

## 1) Feld aus Shapefile ableiten
- Shapefile(s) in `shp/` bereitstellen (z. B. `FF_Grundschule.*`).
- `python auto_borefield.py` → Modus `fitmax`, Sondenabstand S und Tiefe H wählen.
- Projektwerte patchen (rows/cols/Bx/By/H). Artefakte/Backups in `autoborehole/` prüfen.

## 2) Massenstrom plausibilisieren
- `python massflow_limits_scanner.py` oder via Launcher Schritt 2.
- „nominal“ auswählen und nach `Data_Base.py` übernehmen (wirkt auf `R_b,eff`).

## 3) Excel → Profile exportieren
- `python excel_to_qprime.py`
  - Spalte B = mit Rückspeisung (Netto); Spalte C = ohne Rückspeisung (Brutto)
  - Optional „Solar‑Only“ (Spalte D) mit exportieren
  - Ergebnis: `qprime_profile_from_excel.csv` (+ `.json`) und optional `qprime_recharge_only.csv`
- `python excel_to_GHE3.py`
  - Mit Rückspeisung: Extraction=C, Injection=D
  - Ohne Rückspeisung: Extraction=C, Injection=0

## 4) GHEtool L4 prüfen (optional)
- `python GHEtool_Dimensionierer_v4.py`
- CSV auswählen, λ/T_g/ρc/Temperaturgrenzen prüfen
- Tab „Standard“: H_req für gewähltes Raster; Tab „Tiefe fix“: Layoutsuche bei H_fix

## 5) Longterm‑Simulation (v13)
- `python longterm_profile_scaler_v13.py`
- Profilquelle CSV (Gebäude und optional Solar) setzen
- Feldwahl (rectangle/U/freeform), Zeitfaltung (CJ), Grenzwerte (z. B. Monatsmittel EWT ≥ 0 °C)
- Iterationsziel „Last“ oder „Solar“ wählen; Startenergie/Feinschritt konfigurieren
- Starten → Plots/Exporte/Ergebnisdokumentation

## 6) COP‑Einschätzung (optional)
- `python hp_cop_analyzer.py`
- q′‑Profil laden → COP_Carnot/COP_real über die Heizstunden prüfen

## Hinweise
- Doppelzählung vermeiden: Für GHEtool entweder Netto (B) ODER sauber getrennt C/D verwenden.
- JSON‑Meta an q′‑CSV sichern Energieinvarianz bei geänderter Feldlänge.
- Ergebnisse sind Orientierung für Vorplanung – Standortgutachten/VDI‑Details sind weiterhin maßgeblich.

