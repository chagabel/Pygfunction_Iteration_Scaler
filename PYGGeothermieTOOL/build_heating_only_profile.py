# build_heating_only_profile.py
# Erzeugt aus Data_Base.py ein Heizprofil in W/m (feldweit):
#   q'(t) = Q_total(t) / (n_bh * H)
# Optionaler Excel-Export mit zwei Spalten: [hour, W_per_m]
# und Zusatzblatt "info" (L_total, n_bh, H, Feldtyp, Jahresbedarf).

# ==== EIN/AUS: Excel-Export ==========================================
EXPORT_EXCEL   = True    # Auf False setzen, wenn keine Excel-Datei geschrieben werden soll
EXCEL_FILENAME = "heating_profile_W_per_m.xlsx"  # Dateiname für den Export
# =====================================================================

import numpy as np
import Data_Base as bd
import borefields as bf

def fmt_de(n: float, decimals: int = 3, thousands: bool = True) -> str:
    """
    Formatiert n mit deutschem Zahlenformat:
    Dezimalzeichen ',' ; Tausenderpunkt '.'. Gibt String zurück.
    """
    if thousands:
        s = f"{float(n):,.{decimals}f}"   # US: 1,234.567
    else:
        s = f"{float(n):.{decimals}f}"    # US: 1234.567
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s

def choose_field():
    sel = input("Feld wählen: [1] rectangle_field | [2] U_shaped_field -> ").strip()
    if sel == "2":
        return "U_shaped_field", bf.U_shaped_field
    return "rectangle_field", bf.rectangle_field

def main():
    # Feld wählen (bestimmt n_bh und L_total)
    feld_name, feld = choose_field()
    try:
        n_bh = len(feld)
    except TypeError:
        n_bh = len(list(feld))
    H = float(bd.H)
    L_total = n_bh * H

    # Zeitachse in Stunden (0…8760) und Lasten
    hours = bd.time / 3600.0
    Q_total = np.asarray(bd.Q, dtype=float)  # Heizprofil: Sommer bereits 0, auf Bedarf skaliert

    # W/m (feldweit, pro Sonde identisch): q' = Q_total / (n_bh * H)
    denom = max(1.0, n_bh * H)
    q_per_m = Q_total / denom
    q_per_m = np.clip(q_per_m, 0.0, None)     # numerische Minusränder vermeiden

    # Kurzer Konsolenreport
    print("—" * 70)
    print(f"Feld:            {feld_name}")
    print(f"Sonden gesamt:   {n_bh}")
    print(f"Bohrlochlänge H: {H:.2f} m")
    print(f"Gesamtbohrmeter: {L_total:.0f} m")
    print(f"Jahresbedarf:    {bd.Bedarf:.1f} kWh")
    print(f"q' (W/m):        min={q_per_m.min():.2f} | max={q_per_m.max():.2f}")
    print("—" * 70)

    if EXPORT_EXCEL:
        try:
            import pandas as pd
            with pd.ExcelWriter(EXCEL_FILENAME, engine="xlsxwriter") as writer:
                # Numerisch schreiben; Excel zeigt mit Komma an (lokales Format)
                df = pd.DataFrame({"hour": hours, "W_per_m": q_per_m})
                df.to_excel(writer, index=False, sheet_name="profile")

                info = pd.DataFrame({
                    "key": ["field", "n_bh", "H_m", "L_total_m", "annual_kWh", "dt_s"],
                    "value": [feld_name, n_bh, H, L_total, bd.Bedarf, bd.dt]
                })
                info.to_excel(writer, index=False, sheet_name="info")

                # Nummernformat – Excel interpretiert lokal (DE: 1.234,000000)
                wb  = writer.book
                ws1 = writer.sheets["profile"]
                fmt_hours = wb.add_format({"num_format": "0"})
                fmt_wpm   = wb.add_format({"num_format": "#,##0.000000"})
                ws1.set_column("A:A", 10, fmt_hours)
                ws1.set_column("B:B", 18, fmt_wpm)
            print(f"Excel exportiert: {EXCEL_FILENAME}")

        except Exception as e:
            # Fallback: CSV mit deutschem Zahlenformat (Semikolon als Trenner)
            print(f"Excel-Export nicht möglich ({e}). Fallback: CSV.")
            try:
                csv_name = EXCEL_FILENAME.replace(".xlsx", ".csv")
                with open(csv_name, "w", encoding="utf-8") as f:
                    f.write("hour;W_per_m\n")
                    for h, wpm in zip(hours, q_per_m):
                        # Stunden als ganze Zahl ohne Dezimalstellen; W/m deutsch formatiert
                        f.write(f"{fmt_de(h, 0, thousands=False)};{fmt_de(wpm, 6, thousands=True)}\n")
                print(f"CSV exportiert: {csv_name}")
            except Exception as e2:
                print(f"CSV-Export ebenfalls fehlgeschlagen: {e2}")

if __name__ == "__main__":
    main()
    
