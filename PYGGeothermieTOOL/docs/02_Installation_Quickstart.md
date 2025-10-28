# Installation & Quickstart

Dieses Kapitel hilft beim Einrichten der Umgebung und beim schnellen Einstieg per GUI und CLI.

## Voraussetzungen
- Python 3.9 oder neuer (getestet mit 3.10/3.11)
- Betriebssystem mit Tkinter-Unterstützung (für die GUIs)
- Empfohlene Pakete (PyPI):
  - `pygfunction`, `numpy`, `matplotlib`, `pandas`
  - `pyshp` (shapefile), `shapely`, `pyproj` (für Auto‑Borefield)
  - `xlsxwriter` (optionaler Excel‑Export)
  - optional: `GHEtool` (für den Dimensionierer, L4 hourly)

Beispiel‑Installation in einer venv:

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install pygfunction numpy matplotlib pandas pyshp shapely pyproj xlsxwriter
# optional für Dimensionierer:
pip install GHEtool
```

## Startübersicht

- Zentraler Launcher (GUI):
  - `python ANWENDUNG_START.py`
  - Orchestriert Feldfindung, Massenstrom, Profil‑Exporte, GHEtool, Longterm‑Checker, COP‑Analyse.

- Einzeltools (GUI):
  - Feld (Shapefile → Layout/Projektwerte): `python auto_borefield.py`
  - Heiz-/Bodenlast CSV → GHEtool: `python excel_to_GHE3.py`
  - Excel → q′(t) [W/m]: `python excel_to_qprime.py`
  - Dimensionierer (L4): `python GHEtool_Dimensionierer_v4.py`
  - Longterm‑Checker v13: `python longterm_profile_scaler_v13.py`
  - COP‑Analyse: `python hp_cop_analyzer.py`
  - VDI‑Grobcheck: `python vdi_grobcheck.py`

## Quickstart A – Komplett per Launcher
1) `python ANWENDUNG_START.py` starten.
2) Schritt 1: Auto‑Borefield öffnen → Shapefile wählen → Layout erzeugen (rows/cols, Bx/By, H werden ins Projekt übernommen, Backups in `autoborehole/`).
3) Schritt 2: Massenstrom‑Korridor berechnen → „nominal“ in `Data_Base.py` schreiben.
4) Schritt 3: Excel→CSV erzeugen:
   - GHEtool: `excel_to_GHE3.py` (Extraction=C, Injection je nach Modus, kW)
   - Longterm: `excel_to_qprime.py` (hour;W_per_m, inkl. JSON‑Meta)
5) Schritt 4: GHEtool‑GUI aufrufen → L4 Sizing/Layouts prüfen.
6) Schritt 5: Longterm v13 ausführen → Grenzwert‑konformes Potenzial + Plots/Exporte.
7) Optional: COP‑Schätzer → COP über Heizstunden visualisieren.

## Quickstart B – Minimal ohne Excel (Heiz‑Only, kein Solar)
1) Synthetisches Heizprofil nutzen: `synthetic_heating_profile.py` (wird in `Data_Base.py` als `Q` verwendet, auf `Bedarf` skaliert).
2) Feld festlegen (`Data_Base.py` Werte oder Auto‑Borefield).
3) Longterm v13 starten und „synthetic“ als Profilquelle wählen.

## Hinweise & Tipps
- Setze realistische Bodenparameter (`k_s`, `T_g`) und Feldgeometrie (Abstände/Bohrtiefe) in `Data_Base.py`.
- Für Freiform‑Felder: Shapefile(s) in `shp/` ablegen; die Auto‑Borefield‑GUI erzeugt CSV/Meta/Preview in `autoborehole/`.
- Wenn GUIs keine Fenster zeigen: `MPLBACKEND` in der Umgebung leeren (die Tools tun dies meist selbst für Subprozesse).

