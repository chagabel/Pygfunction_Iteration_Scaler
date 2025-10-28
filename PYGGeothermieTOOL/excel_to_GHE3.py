#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_GHE3.py  – Excel → GHEtool-CSV (Version 3)

Korrektur + Härtung:
- „MIT Rückspeisung“: Extraction aus Spalte C, Injection aus Spalte D („Rücksp only“).
- „OHNE Rückspeisung“: Extraction aus Spalte C, Injection = 0.
- Locale-sicherer CSV-Export: Bei Dezimal-Komma immer ';' als Trennzeichen (UTF-8-BOM),
  bei Dezimal-Punkt ',' als Trennzeichen.

Ausgabeformat: hour;Q_extraction_kW;Q_injection_kW
Einheiten: Eingabe in W oder kW; intern wird auf kW normalisiert.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    print("Tkinter konnte nicht importiert werden.", file=sys.stderr)
    raise

DEFAULT_SHEET = "Ergebnis"
DEFAULT_DELIM = ";"
DEFAULT_FLOAT_DIGITS = 2


def _col_letter_to_index(letter: str) -> int:
    letter = letter.strip().upper()
    idx = 0
    for ch in letter:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Ungültiger Spaltenbuchstabe: {letter}")
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def detect_header_and_columns(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    possible = [c for c in df.columns]
    has_header = not all(isinstance(c, (int, float)) for c in possible)
    hour_name = None
    if has_header:
        lower = {str(c).strip().lower(): c for c in df.columns}
        for key in ["hour", "hours", "stunde", "stunden", "zeitstunde", "t"]:
            if key in lower:
                hour_name = lower[key]
                break
    return has_header, hour_name


def _read_xlsx(path: Path, sheet: str, header):
    return pd.read_excel(path, sheet_name=sheet, header=header, decimal=",", thousands=".")


def load_from_excel(
    xlsx_path: Path,
    sheet_name: str,
    hours_col_letter: str,
    extraction_col_letter: str,
    input_unit_kw: bool,
    injection_col_letter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Liest Excel und gibt DataFrame ['hour','Q_extraction_kW','Q_injection_kW'] zurück.
    - extraction_col_letter: Spalte für Heiz-/Entzugswerte (hier C)
    - injection_col_letter: falls gesetzt, wird daraus die Injection gelesen (z. B. D="Rücksp only")
    - input_unit_kw: True, wenn Eingabespalten in kW; sonst W→kW
    """
    try:
        df = _read_xlsx(xlsx_path, sheet_name, header=0)
        has_header, hour_col_name = detect_header_and_columns(df)
    except Exception:
        df = _read_xlsx(xlsx_path, sheet_name, header=None)
        has_header, hour_col_name = False, None

    if not has_header:
        df = _read_xlsx(xlsx_path, sheet_name, header=None)

    # Stunden und Extraction lesen
    if has_header:
        if hour_col_name is None:
            h_idx = _col_letter_to_index(hours_col_letter)
            hours = df.iloc[:, h_idx]
        else:
            hours = df[hour_col_name]
        ext_idx = _col_letter_to_index(extraction_col_letter)
        p_ext = df.iloc[:, ext_idx]
    else:
        h_idx = _col_letter_to_index(hours_col_letter)
        ext_idx = _col_letter_to_index(extraction_col_letter)
        hours = df.iloc[:, h_idx]
        p_ext = df.iloc[:, ext_idx]

    # Injection separat lesen (z. B. Spalte D = Rücksp only)
    p_inj = None
    if injection_col_letter:
        inj_idx = _col_letter_to_index(injection_col_letter)
        p_inj = df.iloc[:, inj_idx]

    # Numerik
    hours = pd.to_numeric(hours, errors="coerce")
    p_ext = pd.to_numeric(p_ext, errors="coerce")
    if p_inj is not None:
        p_inj = pd.to_numeric(p_inj, errors="coerce")

    mask = ~(hours.isna() | p_ext.isna())
    hours = hours[mask].astype(int)
    p_ext = p_ext[mask].astype(float)
    if p_inj is not None:
        p_inj = p_inj[mask].astype(float)

    # Einheiten zu kW normalisieren
    if not input_unit_kw:
        p_ext = p_ext / 1000.0
        if p_inj is not None:
            p_inj = p_inj / 1000.0

    # Nicht-negative Reihen erzeugen
    extraction = p_ext.clip(lower=0.0)
    if p_inj is not None:
        injection = p_inj.clip(lower=0.0)
    else:
        injection = pd.Series(0.0, index=extraction.index)

    out = pd.DataFrame({
        "hour": hours.values,
        "Q_extraction_kW": extraction.values,
        "Q_injection_kW": injection.values,
    }).sort_values("hour").reset_index(drop=True)
    return out


def save_csv(df: pd.DataFrame, out_path: Path, delimiter: str = DEFAULT_DELIM,
             decimal_comma: bool = True, float_digits: int = DEFAULT_FLOAT_DIGITS) -> None:
    """
    Locale-sicher schreiben:
    - Bei Dezimal-Komma erzwingen wir ';' als Trennzeichen und formatieren Zahlen als Strings mit Komma.
    - Bei Dezimal-Punkt verwenden wir ',' als Trennzeichen und float_format.
    - UTF-8 mit BOM (utf-8-sig) hilft Excel, die Datei korrekt zu öffnen.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if decimal_comma:
        sep = ";"
        df2 = df.copy()
        fmt = "{:." + str(float_digits) + "f}"
        for col in ["Q_extraction_kW", "Q_injection_kW"]:
            df2[col] = df2[col].map(lambda x: fmt.format(float(x)).replace(".", ","))
        df2["hour"] = df2["hour"].astype(int)
        df2.to_csv(out_path, index=False, sep=sep, encoding="utf-8-sig")
    else:
        sep = ","
        df.to_csv(out_path, index=False, sep=sep, float_format=f"%.{float_digits}f", encoding="utf-8-sig")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Excel → GHEtool Last (CSV) – Version 3")
        self.geometry("760x520")
        self.resizable(False, False)

        self.var_xlsx = tk.StringVar(value="Wärmelastprofil2.xlsx")
        self.var_sheet = tk.StringVar(value=DEFAULT_SHEET)
        self.var_hours_col = tk.StringVar(value="A")
        self.var_choice = tk.StringVar(value="B")  # B=MIT Rueckspeisung (C+D), C=OHNE (nur C)
        self.var_delim = tk.StringVar(value=DEFAULT_DELIM)
        self.var_decimal_comma = tk.BooleanVar(value=True)
        self.var_float_digits = tk.IntVar(value=DEFAULT_FLOAT_DIGITS)
        self.var_input_unit = tk.StringVar(value="W")  # "W" oder "kW"

        pad = {"padx": 10, "pady": 6}

        frm_file = ttk.LabelFrame(self, text="Excel-Eingabe")
        frm_file.place(x=10, y=10, width=740, height=120)

        ttk.Label(frm_file, text="Excel-Datei:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_file, textvariable=self.var_xlsx, width=60).grid(row=0, column=1, **pad)
        ttk.Button(frm_file, text="Durchsuchen…", command=self.browse).grid(row=0, column=2, **pad)

        ttk.Label(frm_file, text="Tabellenblatt:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_file, textvariable=self.var_sheet, width=20).grid(row=1, column=1, sticky="w", **pad)

        frm_opts = ttk.LabelFrame(self, text="Mapping & Einheiten")
        frm_opts.place(x=10, y=140, width=740, height=220)

        ttk.Label(frm_opts, text="Stunden-Spalte (A..Z):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.var_hours_col, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm_opts, text="Lastquelle:").grid(row=1, column=0, sticky="w", **pad)
        rb1 = ttk.Radiobutton(frm_opts, text="MIT Rückspeisung (Extraction=C, Injection=D)", variable=self.var_choice, value="B")
        rb2 = ttk.Radiobutton(frm_opts, text="OHNE Rückspeisung (Extraction=C, Injection=0)", variable=self.var_choice, value="C")
        rb1.grid(row=1, column=1, sticky="w", **pad)
        rb2.grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(frm_opts, text="Eingabe-Einheit der Last:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_opts, text="kW", variable=self.var_input_unit, value="kW").grid(row=2, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm_opts, text="W",  variable=self.var_input_unit, value="W").grid(row=2, column=2, sticky="w", **pad)

        ttk.Label(frm_opts, text="CSV‑Trennzeichen:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm_opts, textvariable=self.var_delim, width=6).grid(row=3, column=1, sticky="w", **pad)

        ttk.Checkbutton(frm_opts, text="Dezimal‑Komma (DE)", variable=self.var_decimal_comma).grid(row=4, column=0, sticky="w", **pad)
        ttk.Label(frm_opts, text="Nachkommastellen:").grid(row=4, column=1, sticky="w", **pad)
        ttk.Spinbox(frm_opts, from_=0, to=6, textvariable=self.var_float_digits, width=5).grid(row=4, column=2, sticky="w", **pad)

        frm_actions = ttk.Frame(self)
        frm_actions.place(x=10, y=370, width=740, height=60)

        ttk.Button(frm_actions, text="CSV exportieren", command=self.on_export).grid(row=0, column=0, **pad)
        ttk.Button(frm_actions, text="Beenden", command=self.destroy).grid(row=0, column=1, **pad)

        self.status = tk.Text(self, height=6)
        self.status.place(x=10, y=430, width=740, height=80)
        self.log("Bereit. Bitte Excel wählen und Einstellungen prüfen.")

    def log(self, msg: str):
        self.status.insert("end", msg + "\n")
        self.status.see("end")

    def browse(self):
        path = filedialog.askopenfilename(
            title="Excel-Datei wählen",
            filetypes=[("Excel Dateien", "*.xlsx *.xls"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.var_xlsx.set(path)

    def on_export(self):
        try:
            xlsx_path = Path(self.var_xlsx.get()).expanduser()
            if not xlsx_path.exists():
                messagebox.showerror("Fehler", "Excel-Datei nicht gefunden.")
                return
            sheet = self.var_sheet.get().strip() or DEFAULT_SHEET
            hours_col = self.var_hours_col.get().strip() or "A"
            choice = self.var_choice.get().strip().upper()
            # In Version 3: Extraction IMMER aus C, Injection nur bei „B“ aus D
            ext_col = "C"
            inj_col = "D" if choice == "B" else None
            delim = self.var_delim.get() or DEFAULT_DELIM
            dec_comma = bool(self.var_decimal_comma.get())
            digits = int(self.var_float_digits.get())
            input_unit_kw = (self.var_input_unit.get().strip().lower() == "kw")

            self.log(
                f"Lese: {xlsx_path.name} | Blatt: {sheet} | Stunden: {hours_col} | Extraction: {ext_col}"
                + (" | Injection: D" if inj_col else " | Injection: 0")
                + f" ({'kW' if input_unit_kw else 'W'})"
            )

            df = load_from_excel(
                xlsx_path,
                sheet,
                hours_col,
                ext_col,
                input_unit_kw=input_unit_kw,
                injection_col_letter=inj_col,
            )

            suffix = "mit_Rueckspeisung" if choice == "B" else "ohne_Rueckspeisung"
            out_name = f"{xlsx_path.stem}_GHEload_{suffix}.csv"
            out_path = xlsx_path.parent / out_name
            save_csv(df, out_path, delimiter=delim, decimal_comma=dec_comma, float_digits=digits)

            # Kurzer Qualitätscheck/Preview
            nonzero_inj = int((df["Q_injection_kW"] > 0).sum())
            first_inj_vals = df["Q_injection_kW"].head(10).to_list()
            self.log(f"Zeilen: {len(df)} | Injection>0: {nonzero_inj}")
            self.log(f"Injection (erste 10): {first_inj_vals}")

            messagebox.showinfo("Erfolg", f"Gespeichert:\n{out_path}\nZeilen: {len(df)} | Injection>0: {nonzero_inj}")
            self.log(f"Export OK → {out_path}")
        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Fehler", f"{e}\n\n{tb}")
            self.log(tb)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()

