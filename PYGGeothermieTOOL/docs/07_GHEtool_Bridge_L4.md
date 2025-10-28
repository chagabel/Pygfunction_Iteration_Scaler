# GHEtool‑Bridge (L4 hourly)

`GHEtool_Dimensionierer_v4.py` ist eine GUI‑Brücke zu GHEtool. Sie liest stündliche Lasten, setzt Bodenparameter und sucht mit L4 (hourly) entweder die erforderliche Tiefe oder ein Layout bei fixer Tiefe.

## Eingaben
- CSV‑Format: `hour;Q_extraction_kW;Q_injection_kW` (siehe `excel_to_GHE3.py`).
- Bodendaten: `k_s` [W/mK], `T_g` [°C], optional ρc [J/m³K].
- Temperaturgrenzen: min/max für mittlere Fluidtemperatur (T_f,avg).
- SCOP/SEER & Modus: "CSV ist Gebäudelast" (dann erfolgt interne Umrechnung in Bodenlast; sonst direkt Erdlast übernehmen).
- R_b (optional): Verwendung eines konstanten R_b‑Werts.

## Modi
- Tab „Standard (H_req – L4)“: Finde erforderliche Tiefe H_req für gegebenes Raster (nx × ny, Bx, By).
- Tab „Tiefe fix (Layouts – L4)“: Prüfe Layouts (nx, ny) bis zu Obergrenzen, ob H_req ≤ H_fix.

## Exporte
- JSON mit Parametern/Ergebnissen.
- CSV (Kurzüberblick): Feasibility, bestes Layout, H_req_best, Randbedingungen.

## Best Practices
- CSV ist Gebäudelast: „CSV ist Gebäudelast“ aktivieren, SCOP/SEER setzen; Rückspeisung verwenden = true (für Kühlbetrieb/Injection).
- CSV ist Erdlast: „CSV ist Gebäudelast“ deaktivieren; SCOP/SEER = 0; Injection‑Spalte muss dann der Erdreinspeisung entsprechen.
- Für die Kopplung mit Longterm q′(t): nutze `excel_to_GHE3.py` (Korrektur Extraction/Injection) aus dem gleichen Excel.

