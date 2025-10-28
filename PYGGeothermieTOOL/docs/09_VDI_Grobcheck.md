# VDI‑Grobcheck (VDI 4640, Anhang B)

`vdi_grobcheck.py` liefert eine konservative Vorabschätzung der spezifischen Entzugsleistung q′ [W/m] sowie Spitzen‑ und Jahreswerte anhand der Tabellen B2–B7 (Heizen/Heizen+TWW) mit optionalem Laminarfaktor (Tabelle B1).

## Annahmen der Tabellen
- Linienanordnung mit 6 m Sondenabstand
- Doppel‑U 32×3,0; R_b,eff ≈ 0,12 K/(W/m)
- Ungestörte Untergrundtemperatur ≈ 11 °C
- Turbulente Strömung (Laminarfaktor optional)
- Tabellen sind für 1…5 Sonden je Linie definiert; darüber wird konservativ geclamped

## Eingaben
- Feldwahl (`rectangle_field`, `U_shaped_field`, `freeform_field`)
- λ [W/mK] (Interpolation zwischen 1…4 W/mK)
- Volllaststunden h/a (Tabellenbereiche, z. B. 1200…2400 h/a)
- Modus: nur Heizen oder Heizen + TWW
- TWP‑Austritt (−5/−3/0 °C)
- Optional: Laminar‑Korrektur und Großfeld‑Abschlag (heuristisch)

## Ausgaben
- q′ [W/m], Q_peak [kW/MW], Jahresenergie [kWh/MWh/GWh], Spezifisch [kWh/(m·a)]
- Textreport im UI (oder in der Konsole im CLI‑Fallback)

## Nutzung
- `python vdi_grobcheck.py`
- Parameter setzen → „Berechnen“. Ergebnisse dienen als Orientierung vor der detaillierten Longterm‑Simulation.

