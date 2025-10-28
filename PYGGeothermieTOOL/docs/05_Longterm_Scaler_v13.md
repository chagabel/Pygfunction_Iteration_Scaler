# Longterm‑Scaler v13

`longterm_profile_scaler_v13.py` ist die integrierte GUI samt Rechenkern für die Langzeitbewertung. Es faltet die effektive Bodenlast mit den G‑Funktionen und iteriert die Gebäudelast oder die solare Rückspeisung so weit hoch, bis ein gewähltes Grenzkriterium gerade verletzt wird. Daraus ergibt sich ein belastbares Potenzial.

## Ziel & Grundidee
- Zeitscheiben: 8760/h über `YEARS` (Standard: 50 Jahre).
- Effektive Bodenlast: Gebäude_ground − Solar (intern mit COP, wenn aktiviert).
- Faltung: G‑Funktionen aus `pygfunction` (CJ/Liu/MLAA oder stündliche Faltung).
- Iteration: genau eine Seite wird iteriert (Last ODER Solar); die andere bleibt original/fixiert.

## Inputs (wichtig)
- Feld: `rectangle`/`ushaped`/`freeform` (via `choose_field_by_name()`; Freeform aus `autoborehole/borefield_polygon_points.csv`).
- Profile (Standard: CSV):
  - `qprime_profile_from_excel.csv` (hour;W_per_m) – Gebäudelast (B oder C)
  - optional `qprime_recharge_only.csv` (hour;W_per_m) – externe Solar‑Only
  - JSON‑Meta wird eingelesen, um die Energieinvarianz bei veränderter Feldlänge sicherzustellen (L_ref vs. L_total).
- Zeitfaltung: `CJ`, `Liu`, `MLAA` oder `hourly` (stündliche Konvolution der g‑Funktion).
- Rohrmodell: `utube` oder `coaxial` (berechnet `R_b,eff` abhängig vom Massenstrom je Sonde).
- Grenzwerte: Min/Max für `T_f,avg`, `EWT≈T_in` und `T_b` (auch Monatsmittel). Zusätzlich optional Drift‑Kriterien (max. jährliche Abkühlrate).
- COP (optional): wirkt auf die Aufteilung Gebäude/W_el/Boden.

## Iterationslogik (vereinfacht)
1) Lese Profile und skaliere Basisjahresserien (Gebäude/Solar) auf Startwerte.
2) Wiederhole bis `MAX_ITERS_*`:
   - Skaliere die Zielseite (Last oder Solar) per Jahresenergie.
   - Simuliere YEARS*8760 Schritte (Faltung + Rohrmodell → `T_b`, `T_in`, `T_out`, `T_f,avg`).
   - Prüfe Grenzwerte (inkl. Monatsmittel/Drift). Bei erster Verletzung: stoppe.
   - Erhöhe Jahresenergie um `FINE_STEP_PCT_*` und wiederhole.
3) Plotte Zeitreihen, Monatsmittel, Zielprofile (Vorjahr/Verletzungsjahr) und optional Violation‑Detail/Drift.
4) Exportiere Excel der Vorjahresprofile (Gebäude/Solar) je nach Optionen.

## Wichtige Konfigurationsfelder (Auszug)
```python
Config(
  FIELD_NAME='rectangle', YEARS=50, dt=3600,
  PROFILE_SOURCE='csv', CSV_PATH='qprime_profile_from_excel.csv', CSV_PATH_RECHARGE='qprime_recharge_only.csv',
  ITERATION_TARGET='last', LAST_SCALING_MODE='iterate', SOLAR_SCALING_MODE='original',
  GF_METHOD='equivalent', FOLDING_MODE='CJ', BOUNDARY='UBWT', PIPE_TYPE='utube',
  USE_LAST_MIN_TFAVG=True, LAST_MONTH_MEAN_EWT_C=0.0, USE_DRIFT_MIN_MONTH_MEAN_TB=False,
  EXPORT_EXCEL_LAST=True, EXCEL_FILENAME_LAST='heating_profile_W_per_m_prev_year_last.xlsx',
)
```

## Ausgabegrößen
- Temperaturen: `T_b`, `T_in` (EWT), `T_out` (LWT), `T_f_avg` je Zeitschritt.
- Effektive Profile (Bodenlast) für Vorjahr/Verletzungsjahr.
- Jahresenergiebilanz (Gebäude, W_el, Solar intern/extern, Netto Boden).
- Violation‑Paket (Art, Grenzwert, Extremwert, Zeitfenster) für Detailplots.
- Excel‑Profile (Vorjahr): Gebäude/Solar als q′(t) [W/m].

## Hinweise & Best Practices
- Für Performance zunächst `CJ` (Load Aggregation) nutzen; `hourly` nur bei Bedarf (Konvolution 1:1, aber langsamer).
- Feld/Massenstrom vorab plausibilisieren (`auto_borefield.py` + `massflow_limits_scanner.py`).
- Monatsmittel‑Grenzen (EWT/T_f,avg) sind für konservative Kriterien oft zielführend.
- JSON‑Meta an den Profilen ermöglicht energieinvariante Skalierung bei geänderter Feldlänge.

