# Datenformate & Schnittstellen

Dieses Kapitel beschreibt die wichtigsten Ein- und Ausgabeformate der Tools.

## q′(t) [W/m] – Longterm & COP
- CSV: `qprime_profile_from_excel.csv`
  - Spalten: `hour;W_per_m` (8760 Zeilen; 1‑h Raster)
  - Einheit: `W_per_m` in W/m (Vorzeichen erlaubt; Entzug positiv; Einspeisung positiv in separater Datei)
- JSON‑Meta: `qprime_profile_from_excel.json`
  - Beispielinhalte: `dt_s`, `unit`, `source_excel`, `field` (`type`, `n_bh`, `H_m`, `L_total_m`), Statistik
  - Zweck: Energieinvarianz – bei längeren/kürzeren Feldern wird automatisch mit der Referenzfeldlänge `L_ref` korrekt skaliert
- Optional: `qprime_recharge_only.csv/.json` – externe Solar‑Only (positiv)

## GHEtool‑CSV (hourly)
- Aus `excel_to_GHE3.py` erzeugt, robust gegenüber DE/EN‑Locale
- Format: `hour;Q_extraction_kW;Q_injection_kW`
  - Mit Rückspeisung: Extraction = Excel C (Heizlast), Injection = Excel D (Solar‑Only)
  - Ohne Rückspeisung: Extraction = Excel C, Injection = 0
- Einheit: kW, nicht W

## Synthetisches Heiz‑Only Excel
- `build_heating_only_profile.py` erzeugt `heating_profile_W_per_m.xlsx`
  - Sheet `profile`: `hour`, `W_per_m`
  - Sheet `info`: Feldlänge, n_bh, H, Jahresenergie, dt_s

## Auto‑Borefield Artefakte
- `autoborehole/borefield_polygon_points.csv`: Punkte (x_m, y_m) im Polygon
- `autoborehole/borefield_layout_points.csv`: standardisiertes Layout (Rechteck/U) mit gleicher Anzahl N
- `autoborehole/borefield_layout.meta.json`: Parameter (spacing_m, depth_m, edge_offset_m, rows, cols, N, BBox, CRS EPSG, SHP‑Datei, Notizen)
- `autoborehole/borefield_preview.png`: Overlay‑Plot (schwarz: Polygon‑Punkte, blau: Layout‑Punkte)

## Longterm‑Exporte (optional)
- `heating_profile_W_per_m_prev_year_last.xlsx`: Gebäudeprofil (Vorjahr) als W/m
- `solar_profile_W_per_m_prev_year.xlsx`: Solar‑Only (Vorjahr) als W/m
- Plots: Zeitreihen, Monatsmittel, Profil‑Vergleiche, Violation‑Detail, Drift

## Pfadkonventionen
- Profile/CSV/JSON werden im Regelfall im Ordner der jeweiligen Quelle gespeichert (z. B. neben der Excel).
- Auto‑Borefield nutzt den Unterordner `autoborehole/` für alle Artefakte und Backups.

