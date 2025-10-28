# FAQ & Troubleshooting

## Häufige Fragen
- Was ist das Ziel des Tools?
  - Vorstudie/Screening von Erdwärmesonden‑Feldern mit stündlicher Auflösung, nicht die Ausführungsplanung.

- Warum W/m statt W pro Sonde?
  - Die Longterm‑Berechnung arbeitet feldweit mit der Linienlast q′(t) (W/m). Das ist robust gegenüber Änderungen an Bohrzahl/Tiefe (Energieinvarianz mit JSON‑Meta).

- Welche Grenzwerte sind sinnvoll?
  - Üblich: EWT‑Monatsmittel ≥ 0 °C nach 50 Jahren. Alternativ T_f,avg oder direkte T_b‑Grenzen – abhängig von Projekt/Vorgaben.

- Muss Solar‑Only getrennt exportiert werden?
  - Ja, wenn Netto/Bilanz getrennt betrachtet werden soll (GHEtool getrennte Spalten). Für die reine Netto‑Betrachtung reicht Spalte B.

## Typische Fehler & Lösungen
- GUI startet ohne Plotfenster (Spyder/Jupyter):
  - Ursache: Inline‑Backend. Lösung: Die Tools setzen `MPLBACKEND=TkAgg` in Subprozessen; ggf. manuell Umgebungsvariable leeren und neu starten.

- `excel_to_qprime.py`: „8760 Zeilen erwartet“:
  - Das Ergebnisblatt muss 8760 Stunden durchgehend enthalten (1‑h Raster). Prüfe Zeitachsen und Trim‑Hinweise im Tool.

- Longterm v13 bricht sofort ab:
  - Meist wegen dt ≠ 3600 s in Kombination mit `month_indices_for_years`. Stelle `dt=3600` ein oder wähle den `hourly`‑Modus konsistent.

- Freeform‑Feld fehlt:
  - Prüfe `autoborehole/borefield_polygon_points.csv` und Spalten `x_m,y_m`. Alternativ Freiform in `borefields.py` deaktiviert.

- GHEtool‑Importe schlagen fehl:
  - `pip install GHEtool` und sicherstellen, dass die Version Hourly‑Loads unterstützt. Das Skript unterstützt „neue“ und „alte“ API.

- Hydraulik wirkt unrealistisch:
  - Massenstrom‑Korridor (min/nom/max) berechnen und `flow` nominal nach `Data_Base.py` schreiben. Einfluss auf `R_b,eff` prüfen.

## Weiterführende Ideen
- Automatisierter Import stündlicher Solar‑Daten (API/CSV) + Validierung (8760/Gaps/Einheiten) als Python‑Modul.
- Erweiterung auf Kühlbetrieb und Mischszenarien im Longterm‑Kern.

