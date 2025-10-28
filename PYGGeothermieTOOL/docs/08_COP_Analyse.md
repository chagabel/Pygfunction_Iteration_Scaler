# COP‑Analyse aus q′(t)

`hp_cop_analyzer.py` berechnet zeitaufgelöst den theoretischen (Carnot) und einen realistischen COP über die Heizstunden aus einem q′(t)‑Profil [W/m].

## Inputs
- q′‑Profil (8760/h) als CSV oder Excel:
  - CSV: Spalte `W_per_m` (Delimiter automatisch)
  - Excel: Blatt `profile_prev_year` oder erstes Blatt, Spalte `W_per_m`
- Feld: `rectangle`/`ushaped`/`freeform` (wie in Longterm v13)
- Faltung: standardmäßig `CJ`; `hourly` möglich
- Rohrmodell: `utube`/`coaxial` (berechnet `R_b,eff` aus ṁ je Sonde)
- COP‑Parameter: Vorlauf (z. B. 60 °C), ΔT an Kondensator/Verdampfer, Gütegrad η_Carnot

## Outputs
- Zeitreihen: COP_Carnot, COP_real, T_b, T_in≈EWT, T_out≈LWT, q_ground
- Kennzahlen: Median und arithmetisches Mittel über Heizstunden (q_ground > 0)
- Plots: COP_Carnot und COP_real

## Nutzung
- GUI: `python hp_cop_analyzer.py`
- Profil auswählen und Start drücken. Abbruch jederzeit möglich.

## Hinweise
- COP wird nur für Heizstunden berechnet; Einspeisestunden (q_ground ≤ 0) werden als NaN ausgeblendet.
- Die Ergebnisse sind sensitiv gegenüber `k_s`, `T_g`, Feldgeometrie und ṁ – bitte vorab plausibilisieren.

