# Projektaufbau & Architektur

Dieses Kapitel ordnet die Python‑Module ein und zeigt die Datenflüsse.

## Übersicht (Top‑Level Module)

- `ANWENDUNG_START.py`
  - Zentrale Start‑GUI („Launcher“): ruft die anderen GUIs/Tools in sinnvoller Reihenfolge auf, zeigt Projektstatus, übernimmt Massenstrom in `Data_Base.py`.

- Geometrie / Feld
  - `borefields.py`: Rechteck‑/U‑Felder aus `pygfunction` (Parameter aus `Data_Base.py`); optionales Freiform‑Feld aus `autoborehole/borefield_polygon_points.csv`.
  - `auto_borefield.py`: GUI/CLI zum Ableiten von rows×cols/Bx/By/H aus einer Polygonfläche (`shp/…`). Erzeugt CSV/Meta/Preview und kann Projektdateien patchen.
  - `borehole.py`: Diagnose/Visualisierung von Rohrmodellen (Single‑U/Coaxial), inkl. effektivem Bohrlochwiderstand `R_b,eff`.

- Profile / Exporte
  - `Wärmelastprofil2.xlsx`: Beispiel‑Arbeitsmappe (Hardegsen). Ergebnisblatt liefert Gebäudelast B/C und Solar‑Only D.
  - `excel_to_qprime.py`: Excel → q′(t) [W/m] (8760/h) + JSON‑Meta. Optional Solar‑Only.
  - `excel_to_GHE3.py`: Excel → GHEtool‑kompatibles `hour;Q_extraction_kW;Q_injection_kW`.
  - `synthetic_heating_profile.py`: Heizen‑only (keine Rückspeisung), auf Jahresenergie skaliert.
  - `build_heating_only_profile.py`: Writer für ein sauberes Heizen‑only q′(t) Excel (alternativ zum obigen Modul).

- Simulation / Bewertung
  - `longterm_profile_scaler_v13.py`: Integrierte GUI + Rechner. Faltet die Last(en) mit den G‑Funktionen (CJ/Liu/MLAA/hourly) und iteriert Last oder Solar bis Grenzverletzung. Viele Plot‑/Exportoptionen.
  - `GHEtool_Dimensionierer_v4.py`: Brücke zu GHEtool (L4 hourly): Sizing, Layout‑Suche, JSON/CSV‑Export.
  - `vdi_grobcheck.py`: VDI 4640 B2–B7 Tabellen‑Checker (konservativer Grobabgleich, 6 m, Doppel‑U 32).
  - `massflow_limits_scanner.py`: Hydraulisch/thermisch sinnvolle Massenstrom‑Bänder (min/nom/max) je Feld.
  - `hp_cop_analyzer.py`: COP‑Analyse (Heizbetrieb) aus q′(t) [W/m].

- Defaults / Datenbasis
  - `Data_Base.py`: Projektweite Default‑Parameter (Boden, Rohr, Fluid, Zeitraster, synthetisches Profil). Wird von fast allen Tools importiert.

## Datenflüsse (vereinfacht)

1) Geometrie
   - `auto_borefield.py` → erzeugt `autoborehole/borefield_*.csv/.meta.json` und schreibt `rows_sonden/Columns_sonden/Bx/By/H` in `Data_Base.py` und `borefields.py` (Backup wird angelegt).
   - `borefields.py` → stellt `rectangle_field`, `U_shaped_field`, optional `freeform_field` bereit.

2) Profil
   - `Wärmelastprofil2.xlsx` (B/C/D) → `excel_to_qprime.py` → `qprime_profile_from_excel.csv` (+ `.json` Meta) und optional `qprime_recharge_only.csv`.
   - Longterm v13 liest die CSV(s) und simuliert q′(t) (Boden) mit/ohne externe Rückspeisung.

3) Bewertung
   - Longterm v13: iteriert Gebäudelast ODER Solar gegen Temperaturgrenzen (EWT/T_favg/T_b, optional Monatsmittel/Drift) → Plots, Excel‑Exporte.
   - GHEtool v4: nutzt `hour;Q_extraction_kW;Q_injection_kW` (aus `excel_to_GHE3.py` oder anderen Quellen) → L4‑Sizing/Layouts.
   - VDI‑Grobcheck: q′ tabellarisch aus VDI‑Werten, konservativ.
   - Massenstrom‑Scanner: liefert ṁ‑Bänder für realistische Simulationen (auch als Eingabe für Longterm v13 geeignet).

## Vorzeichen & Konventionen (wichtig)
- q′(t) in W/m (Linienlast bezogen auf Feldlänge `n_bh·H`).
- Entzug (Heizen) = positiv. Einspeisung (Solar) = separat positiv; Netto kann ± sein.
- CSV für GHEtool: kW, getrennte Spalten für Extraction/Injection.

## Typische Reihenfolge in Projekten
1. Feld aus Shapefile ableiten → Projektwerte patchen.
2. Massenstrom‑Band bestimmen und „nominal“ setzen.
3. Excel‑Profil(e) generieren (q′ und GHEtool‑CSV).
4. GHEtool‑L4 prüfen (Layout/Tiefe) und Longterm v13 iterieren (Grenzen/Varianten).
5. Optional: COP‑Analyse und VDI‑Grobcheck als Gegencheck.

