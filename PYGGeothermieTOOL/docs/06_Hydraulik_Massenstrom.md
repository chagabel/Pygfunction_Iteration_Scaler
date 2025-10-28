# Hydraulik & Massenstrom (Scanner)

`massflow_limits_scanner.py` liefert ohne Lastannahmen drei Massenstrom‑Stufen je Feld: min / nominal / max. Diese Bänder sind als realistische Massenstrom‑Korridore gedacht und helfen, langfristige Simulationen und Dimensionierungen plausibel zu halten.

## Definitionen
- min (Baseline‑Deckel, heutiges „max“):
  - Geschwindigkeitslimit v ≤ 1.00 m/s, Δp je Bohrlochpfad ≤ 0.20 bar
  - zusätzlich Turbulenz‑Floor: Re ≥ 2500 UND v ≥ 0.40 m/s (Single‑U als Referenz)
- nominal (Empfehlung):
  - thermische Sättigung erreicht (ΔR_b,eff/R_b,eff ≤ 1 %) UND
  - moderates Hydraulik‑Limit v ≤ 1.10 m/s, Δp ≤ 0.30 bar
- max (Markt‑Deckel):
  - thermische Sättigung UND v ≤ 1.50 m/s, Δp ≤ 0.50 bar
  - berücksichtigt marktübliche Varianten (Single‑U 40, Double‑U 2×32/2×40) für die hydraulische Obergrenze.

## Rechenweg
- Hydraulik: Darcy–Weisbach (Haaland‑f), Pfadlänge ≈ `2·H·(1 + L_EQ_FACTOR)` mit `L_EQ_FACTOR = 0.15`. Pro Sonde `LEGS_PER_BH = 2` (Single‑U).
- Thermik: `R_b,eff(ṁ_bh)` aus `pygfunction.pipes`. „Sättigung“: relative Änderung ≤ 1 % bei Verdopplung des Massenstroms.
- Ausgabe jeweils pro Sonde (kg/s) und gesamt (kg/s).

## Nutzung
- Direkt: `python massflow_limits_scanner.py` → Konsolenbericht + Info‑Plot.
- Im Launcher: Schritt 2 „Korridor berechnen“ und „nominal“ direkt nach `Data_Base.py` schreiben.

## Hinweise
- Der Header‑Druckverlust (Verteiler) wird nicht berechnet; Empfehlung ≤ 0.10 bar (als Hinweis im Report aufgeführt).
- Die Scanner‑Ergebnisse sind lastunabhängig – für die Simulation im Longterm‑Checker (Einfluss auf `R_b,eff`) ist der nominale Korridor ein guter Startwert.

