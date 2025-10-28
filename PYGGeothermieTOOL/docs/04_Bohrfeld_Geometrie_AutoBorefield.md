# Bohrfeld-Geometrie (Auto-Borefield)

`auto_borefield.py` erzeugt aus einer Polygonfläche (Shapefile) ein nutzbares Borefield‑Layout und übernimmt die Kernwerte ins Projekt.

## Funktionen
- Einlesen von Shapefile(s) aus `shp/` (mit `.prj/.shx/.dbf`).
- Reprojektion in metrische Koordinaten (automatisch per UTM).
- Zwei Modi:
  - `fitmax`: Anzahl Bohrlöcher N maximal in die Fläche mit gegebenem Abstand S, optionaler Scan über Rotation/Phasenlage, anschließend standardisiertes Rechteck/U‑Layout mit gleicher N.
  - `legacy`: Rechteck/U innerhalb der Fläche, wie frühere Versionen.
- Export:
  - `autoborehole/borefield_polygon_points.csv` (tatsächliche Punkte im Polygon)
  - `autoborehole/borefield_layout_points.csv` (standardisiertes Rechteck/U mit identischer N)
  - `autoborehole/borefield_layout.meta.json` (Parameter/CRS/Bounding Box)
  - `autoborehole/borefield_preview.png` (Overlay‑Plot)
- Projektdateien patchen (Backups): `Data_Base.py` und `borefields.py` (`rows_sonden`, `Columns_sonden`, `Bx`, `By`, `H`).

## GUI‑Nutzung
- Start: `python auto_borefield.py`
- Wichtige Eingaben:
  - Projektordner (Root dieses Repos)
  - Shapefile‑Ordner (`shp/`)
  - Modus: `fitmax` (empfohlen) oder `legacy`
  - Layout: `rectangle` / `ushaped` / `freeform` (Freeform wird nur von Longterm genutzt)
  - `spacing` S [m], Tiefe H [m], Randabstand zum Polygon [m], U‑Gap [m]
  - Optional: Rotations‑Scan [°] und Phasen‑Scan [Schritte]
  - „Projektwerte überschreiben“ einschalten, damit rows/cols/Bx/By/H ins Projekt geschrieben werden (Backups in `autoborehole/`).

## CLI‑Nutzung (Kurz)
- `python auto_borefield.py --help`
- Erzeugte Artefakte und Backups liegen in `autoborehole/`.

## Hinweise
- Die Felder `rectangle_field`/`U_shaped_field` in `borefields.py` werden aus `Data_Base.py` gespeist und nach einem Patch sofort von allen Tools genutzt.
- Das optionale `freeform_field` wird aus `autoborehole/borefield_polygon_points.csv` geladen (falls vorhanden) und steht dann u. a. im Longterm‑Checker zur Verfügung.

