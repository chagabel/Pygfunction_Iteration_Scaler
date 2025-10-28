#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_qprime.py
------------------
Liest ein Excel-Wärmelastprofil (Ergebnisseite) ein und erzeugt
pygfunction-kompatible Linienlast-CSV(s) q'(t) [W/m] (8760 Zeilen, 1-h Raster).

Spalten (Ergebnisseite, typisch A/B/C/D):
- Spalte A: Stunde (1..8760 oder 0..8759)
- Spalte B: Gebäudelast MIT solarer Rückspeisung [W] (Vorzeichen erlaubt)
- Spalte C: Gebäudelast OHNE solare Rückspeisung [W] (>=0)
- Spalte D: SOLAR-ONLY Rückspeisung [W] (optional – positive Werte = Einspeisung)

UI (Tkinter):
- Radio: „Mit Rückspeisung (B)“ oder „Ohne Rückspeisung (C)“
- Checkbox: „Solar-only (Spalte D) zusätzlich exportieren“
- Dateiauswahl (Default: ./Wärmelastprofil2.xlsx)
- Optional: JSON mit Metadaten mitschreiben (nur für das Hauptprofil)

Ausgabe (im Ordner der Excel-Datei):
- qprime_profile_from_excel.csv         (Spalten: hour;W_per_m)  -> Hauptprofil (B oder C)
- qprime_profile_from_excel.json        (optional, Metadaten für Hauptprofil)
- qprime_recharge_only.csv              (optional, wenn D exportiert wird; hour;W_per_m)

Wichtig:
- COP wird hier NICHT angewendet (Excel = Gebäudelast bzw. Solar-Einspeisung).
- Umrechnung W → W/m mit aktuellem Feld (n_bh, H) aus dem Projekt.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Drittmodule
try:
    import numpy as np
    import pandas as pd
except Exception:
    print("Fehlende Pakete: pandas/numpy erforderlich. Bitte installieren (z. B. conda-forge).")
    raise

# Projekt-Module (liegen im Root)
try:
    import Data_Base as bd
    import borefields as bf
except Exception as e:
    print("Konnte Projektmodule nicht importieren (Data_Base, borefields). Stelle sicher, dass das Skript im Projekt-Root läuft.")
    raise


def _detect_results_sheet(xl: pd.ExcelFile) -> str:
    """Finde eine sinnvolle Ergebnisseite. Bevorzugt Blattnamen mit 'Ergebnis'."""
    sheets = xl.sheet_names
    for name in sheets:
        low = name.lower()
        if "ergebnis" in low or "results" in low:
            return name
    return sheets[0] if sheets else None


def _read_excel_series(path: Path, use_column: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Liest Excel und gibt (hours, Q_building_W) zurück.
    use_column: 'B' -> mit Rückspeisung; 'C' -> ohne Rückspeisung.
    Nimmt die ersten Spalten als A/B/C[(/D)] des Ergebnisblatts.
    """
    if not path.exists():
        raise FileNotFoundError(f"Excel nicht gefunden: {path}")

    xl = pd.ExcelFile(path)
    sheet = _detect_results_sheet(xl)
    if sheet is None:
        raise RuntimeError("Keine Arbeitsblätter in der Excel gefunden.")

    df = xl.parse(sheet)
    if df.shape[1] < 3:
        raise RuntimeError(f"Ergebnisblatt '{sheet}' hat weniger als 3 Spalten.")

    col_A = df.iloc[:, 0]
    col_B = df.iloc[:, 1]
    col_C = df.iloc[:, 2]

    # Stunden säubern → integer 1..8760 oder 0..8759 akzeptieren
    hours = pd.to_numeric(col_A, errors='coerce').to_numpy()
    if np.any(np.isnan(hours)):
        n = len(col_A)
        hours = np.arange(1, n+1, dtype=float)

    # Gebäudelast wählen
    if use_column.upper() == 'C':
        q_building = pd.to_numeric(col_C, errors='coerce').to_numpy()
    else:
        q_building = pd.to_numeric(col_B, errors='coerce').to_numpy()

    hours = np.where(np.isnan(hours), 0, hours).astype(float)
    q_building = np.where(np.isnan(q_building), 0.0, q_building).astype(float)
    return hours, q_building


def _read_excel_solar_only(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Liest die SOLAR-ONLY Serie (Spalte D) als (hours, Q_solar_W).
    Erwartet mindestens 4 Spalten auf dem Ergebnisblatt.
    """
    if not path.exists():
        raise FileNotFoundError(f"Excel nicht gefunden: {path}")

    xl = pd.ExcelFile(path)
    sheet = _detect_results_sheet(xl)
    if sheet is None:
        raise RuntimeError("Keine Arbeitsblätter in der Excel gefunden.")

    df = xl.parse(sheet)
    if df.shape[1] < 4:
        raise RuntimeError(f"Ergebnisblatt '{sheet}' hat keine Spalte D (Solar-only).")

    col_A = df.iloc[:, 0]
    col_D = df.iloc[:, 3]

    hours = pd.to_numeric(col_A, errors='coerce').to_numpy()
    if np.any(np.isnan(hours)):
        n = len(col_A)
        hours = np.arange(1, n+1, dtype=float)

    q_solar = pd.to_numeric(col_D, errors='coerce').to_numpy()
    hours = np.where(np.isnan(hours), 0, hours).astype(float)
    q_solar = np.where(np.isnan(q_solar), 0.0, q_solar).astype(float)
    return hours, q_solar


def _field_params_from_project(prefer: str | None = None) -> tuple[int, float, str]:
    """Ermittelt (n_bh, H, field_type) aus den verfügbaren Feldern.
    prefer: 'freeform'|'freeform_field'|'rectangle'|'rectangle_field'|'ushaped'|'U_shaped_field' oder None.
    Bevorzugt Freeform, dann Rectangle, dann U‑shaped, falls 'prefer' nicht gesetzt
    oder nicht verfügbar. Konvertiert Borefield via .to_boreholes()."""
    H = float(getattr(bd, "H"))

    def _as_len(obj):
        if obj is None:
            return 0
        try:
            return len(obj)
        except TypeError:
            try:
                return len(list(obj))
            except Exception:
                return 0

    # Mapping UI → intern
    pref_map = {
        'freeform': 'freeform_field', 'freeform_field': 'freeform_field',
        'rectangle': 'rectangle_field', 'rectangle_field': 'rectangle_field',
        'ushaped': 'U_shaped_field', 'u_shaped': 'U_shaped_field', 'U_shaped_field': 'U_shaped_field'
    }
    order = []
    if prefer:
        prefer_norm = pref_map.get(prefer, prefer)
        order.append(prefer_norm)
    # Standardreihenfolge
    for name in ('freeform_field', 'rectangle_field', 'U_shaped_field'):
        if name not in order:
            order.append(name)

    for name in order:
        obj = getattr(bf, name, None)
        if obj is None:
            continue
        try:
            holes = obj.to_boreholes() if hasattr(obj, 'to_boreholes') else obj
        except Exception:
            holes = obj
        n = _as_len(holes)
        if n > 0:
            return int(n), H, name

    raise RuntimeError("Kein nutzbares Feld gefunden: freeform/rectangle/U_shaped haben keine Bohrlöcher.")


def _validate_series(hours: np.ndarray, q_vals: np.ndarray) -> None:
    """Validiert 1-h Raster, 8760 Werte, keine Gaps. Stunden 1..8760 oder 0..8759 akzeptiert."""
    if hours.ndim != 1 or q_vals.ndim != 1 or hours.size != q_vals.size:
        raise RuntimeError("Spalten passen nicht zusammen (unterschiedliche Länge).")
    n = hours.size
    if n != 8760:
        raise RuntimeError(f"Erwarte 8760 Stunden, gefunden: {n}")
    diffs = np.diff(hours.astype(float))
    if not np.allclose(diffs, 1.0, atol=1e-6):
        raise RuntimeError("Die Stunden-Spalte ist nicht durchgehend im 1-h Raster (Δ=1).")


def _normalize_length(hours: np.ndarray, q_vals: np.ndarray):
    """
    Normalisiert auf exakt 8760 Zeilen.
    8760 -> OK
    8761 -> trimme heuristisch (siehe Code)
    >8761 -> kürze auf 8760
    <8760 -> Fehler
    """
    n = int(hours.size)
    if n == 8760:
        return hours, q_vals, None
    if n == 8761:
        try:
            h0 = int(round(float(hours[0]))); hN = int(round(float(hours[-1])))
        except Exception:
            h0, hN = hours[0], hours[-1]
        if h0 == 0 and hN == 8760:
            return hours[1:], q_vals[1:], "trim_first_0_to_8760"
        if h0 == 1 and hN == 8761:
            return hours[:-1], q_vals[:-1], "trim_last_1_to_8761"
        return hours[:8760], q_vals[:8760], "trim_last_default"
    if n > 8761:
        return hours[:8760], q_vals[:8760], "trim_to_8760"
    raise RuntimeError(f"Erwarte 8760 Stunden, gefunden: {n}")


def _compute_qprime(q_W: np.ndarray, n_bh: int, H_m: float) -> np.ndarray:
    denom = max(1.0, n_bh * H_m)
    return q_W.astype(float) / denom  # W/m, Vorzeichen bleibt


def _write_csv(hours: np.ndarray, qprime: np.ndarray, out_path: Path) -> None:
    # Semikolon als Trenner; Dezimalpunkt; keine Tausendertrennzeichen
    with out_path.open("w", encoding="utf-8") as f:
        f.write("hour;W_per_m\n")
        for h, wpm in zip(hours, qprime):
            try:
                h_int = int(round(float(h)))
            except Exception:
                h_int = int(h)
            f.write(f"{h_int};{wpm:.6f}\n")


def _write_json_meta(hours, qprime, n_bh, H, use_col, excel_name, out_path_json: Path, kind: str = "building", field_type: str = "rectangle_field") -> None:
    E_kWh_building = float(qprime.sum() * (n_bh * H) * 3600.0 / 3.6e6)  # Sum(q' * L_total * dt)
    meta = {
        "version": "v1",
        "dt_s": 3600,
        "unit": "W_per_m",
        "source_excel": excel_name,
        "profile_kind": kind,  # "building" oder "solar_only"
        "column_used": "B" if use_col.upper() == "B" else "C",
        "field": {"type": field_type, "n_bh": n_bh, "H_m": H, "L_total_m": float(n_bh * H)},
        "series_length": int(len(qprime)),
        "stats": {
            "E_kWh_building_estimate": E_kWh_building,  # noch kein COP berücksichtigt
            "W_per_m_min": float(np.min(qprime)),
            "W_per_m_max": float(np.max(qprime)),
        },
    }
    with out_path_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_noninteractive(excel_path: Path, use_col: str = "B",
                       write_json: bool = True, write_solar_only: bool = False,
                       field_choice: str | None = None) -> tuple[Path, Path | None, Path | None, Path | None]:
    # Hauptprofil (B/C)
    hours, q_building = _read_excel_series(excel_path, use_col)
    hours, q_building, _ = _normalize_length(hours, q_building)
    _validate_series(hours, q_building)
    n_bh, H, ftype = _field_params_from_project(field_choice)
    qprime = _compute_qprime(q_building, n_bh, H)

    out_csv = excel_path.parent / "qprime_profile_from_excel.csv"
    _write_csv(hours, qprime, out_csv)

    out_json = excel_path.parent / "qprime_profile_from_excel.json"
    _write_json_meta(hours, qprime, n_bh, H, use_col, excel_path.name, out_json, kind="building", field_type=ftype)

    # Optional: Solar-only (D)
    out_csv_solar = None
    out_json_solar = None
    if write_solar_only:
        hours_s, q_solar = _read_excel_solar_only(excel_path)
        hours_s, q_solar, _ = _normalize_length(hours_s, q_solar)
        _validate_series(hours_s, q_solar)
        if hours_s.shape != hours.shape or not np.allclose(hours_s, hours):
            raise RuntimeError("Stundenachsen von Hauptprofil und Solar-only unterscheiden sich.")
        qprime_re = _compute_qprime(q_solar, n_bh, H)
        out_csv_solar = excel_path.parent / "qprime_recharge_only.csv"
        _write_csv(hours_s, qprime_re, out_csv_solar)
        out_json_solar = excel_path.parent / "qprime_recharge_only.json"
        _write_json_meta(hours_s, qprime_re, n_bh, H, use_col, excel_path.name, out_json_solar, kind="solar_only", field_type=ftype)

    # Konsole: Kurzreport
    E_kWh = float(qprime.sum() * (n_bh * H) * 3600.0 / 3.6e6)
    print(f"[OK] CSV geschrieben: {out_csv.name}  |  Jahresenergie (Gebäude): {E_kWh:,.0f} kWh".replace(",","X").replace(".",",").replace("X","."))
    print(f"    Feld: n_bh={n_bh}, H={H:.2f} m  →  L_total={n_bh*H:.0f} m")
    print(f"    q' [W/m]: min={np.min(qprime):.3f}  max={np.max(qprime):.3f}")
    if out_csv_solar is not None:
        E_s_kWh = float(np.sum(qprime_re) * (n_bh * H) * 3600.0 / 3.6e6)
        print(f"[OK] Solar-only CSV geschrieben: {out_csv_solar.name}  |  Jahresenergie (Solar, Vorzeichen beachtet): {E_s_kWh:,.0f} kWh".replace(",","X").replace(".",",").replace("X","."))
    return out_csv, out_csv_solar, out_json, out_json_solar


# ---------------- Tkinter UI ----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Excel → q′(t) [W/m] (pyg)")
        self.geometry("980x360")
        self.resizable(True, True)

        self.excel_path = tk.StringVar(value=str(Path("./Wärmelastprofil2.xlsx").resolve()))
        self.use_col = tk.StringVar(value="B")      # B=mit Rückspeisung, C=ohne
        # JSON wird immer geschrieben (Metadaten inkl. L_ref)
        self.write_json = tk.BooleanVar(value=True)
        self.write_solar_only = tk.BooleanVar(value=False)
        # Feldwahl (freeform/rectangle/ushaped)
        self.var_field = tk.StringVar(value=self._default_field_choice())

        self._build()

    def _build(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        r=0
        ttk.Label(frm, text="Excel-Datei (Ergebnisseite mit Spalten A/B/C/D):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.excel_path, width=58).grid(row=r, column=1, **pad)
        ttk.Button(frm, text="Durchsuchen…", command=self._browse).grid(row=r, column=2, **pad); r+=1

        ttk.Label(frm, text="Gebäudelast wählen (Hauptprofil):").grid(row=r, column=0, sticky="w", **pad)
        rb1 = ttk.Radiobutton(frm, text="Mit Rückspeisung (Spalte B)", variable=self.use_col, value="B")
        rb2 = ttk.Radiobutton(frm, text="Ohne Rückspeisung (Spalte C)", variable=self.use_col, value="C")
        rb1.grid(row=r, column=1, sticky="w", **pad); rb2.grid(row=r, column=2, sticky="w", **pad); r+=1

        ttk.Checkbutton(frm, text="Zusätzlich 'Solar-only' (Spalte D) exportieren",
                        variable=self.write_solar_only).grid(row=r, column=0, columnspan=3, sticky="w", **pad); r+=1
        # Feldwahl
        ttk.Label(frm, text="Feld für Umrechnung (W → W/m):").grid(row=r, column=0, sticky="w", **pad)
        self.cmb_field = ttk.Combobox(frm, textvariable=self.var_field, values=self._available_field_choices(), state="readonly", width=22)
        self.cmb_field.grid(row=r, column=1, sticky="w", **pad)
        self.lbl_field_info = ttk.Label(frm, text="–", foreground="#333")
        self.lbl_field_info.grid(row=r, column=2, sticky="w", **pad)
        try:
            self.cmb_field.bind('<<ComboboxSelected>>', lambda e: self._update_field_info())
        except Exception:
            pass
        r+=1
        ttk.Label(frm, text="JSON-Metadaten werden immer erzeugt (inkl. L_ref).", foreground="#333").grid(row=r, column=0, columnspan=3, sticky="w", **pad); r+=1

        btnfrm = ttk.Frame(frm); btnfrm.grid(row=r, column=0, columnspan=3, pady=(16,0))
        ttk.Button(btnfrm, text="Profil(e) erzeugen", command=self._run).grid(row=0, column=0, padx=10)
        ttk.Button(btnfrm, text="Schließen", command=self.destroy).grid(row=0, column=1, padx=10)

        note = ("Ausgaben im Excel-Ordner:\n"
                "  • qprime_profile_from_excel.csv + .json (mit L_ref)\n"
                "  • optional: qprime_recharge_only.csv + .json (mit L_ref)\n"
                "Voraussetzung: 8760 Zeilen, 1-h Raster. COP wird hier NICHT angewendet.")
        ttk.Label(frm, text=note, foreground="#333").grid(row=r+1, column=0, columnspan=3, sticky="w", padx=5, pady=8)

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Excel wählen",
            filetypes=[("Excel-Dateien", "*.xlsx;*.xlsm;*.xls"), ("Alle Dateien", "*.*")],
            initialdir=str(Path(".").resolve())
        )
        if p:
            self.excel_path.set(p)

    def _available_field_choices(self) -> list[str]:
        out = []
        try:
            # Reihenfolge der Anzeige: freeform, rectangle, ushaped (falls verfügbar)
            n, _, _ = None, None, None
            # Freeform
            try:
                ff = getattr(bf, 'freeform_field', None)
                if ff is not None:
                    n = len(ff)
            except TypeError:
                try:
                    n = len(list(ff))
                except Exception:
                    n = 0
            if n and int(n) > 0:
                out.append('freeform')
            # Rectangle/U
            for nm in ('rectangle_field','U_shaped_field'):
                obj = getattr(bf, nm, None)
                if obj is None:
                    continue
                try:
                    holes = obj.to_boreholes() if hasattr(obj,'to_boreholes') else obj
                    count = len(holes)
                except Exception:
                    try:
                        count = len(list(obj))
                    except Exception:
                        count = 0
                if count and int(count) > 0:
                    out.append('rectangle' if nm=='rectangle_field' else 'ushaped')
        except Exception:
            pass
        if not out:
            out = ['rectangle','ushaped','freeform']
        return out

    def _default_field_choice(self) -> str:
        choices = self._available_field_choices()
        # Vorzug: freeform → rectangle → ushaped
        for pref in ('freeform','rectangle','ushaped'):
            if pref in choices:
                return pref
        return choices[0] if choices else 'rectangle'

    def _update_field_info(self):
        try:
            # N/H anzeigen für die gewählte Option
            nm = self.var_field.get().strip().lower()
            internal = {'freeform':'freeform_field','rectangle':'rectangle_field','ushaped':'U_shaped_field'}.get(nm,nm)
            n_bh, H, ftype = _field_params_from_project(internal)
            self.lbl_field_info.config(text=f"n_bh={n_bh}, H={H:.2f} m, L_total={int(n_bh*H)} m, src={ftype}")
        except Exception as e:
            self.lbl_field_info.config(text=f"Feldinfo: {e}")

    def _run(self):
        try:
            p = Path(self.excel_path.get()).resolve()
            use = self.use_col.get()
            write_solar = bool(self.write_solar_only.get())
            # Feldwahl aus Combobox
            field_choice_ui = (self.var_field.get() or '').strip().lower()
            field_choice = {'freeform':'freeform_field','rectangle':'rectangle_field','ushaped':'U_shaped_field'}.get(field_choice_ui, None)
            out_csv, out_csv_solar, out_json, out_json_solar = run_noninteractive(
                p, use_col=use, write_json=True, write_solar_only=write_solar, field_choice=field_choice
            )
            msg = f"CSV/JSON erstellt:\n{out_csv}\n{out_csv.with_suffix('.json')}"
            if out_csv_solar:
                msg += f"\nSolar-only CSV/JSON erstellt:\n{out_csv_solar}\n{out_csv_solar.with_suffix('.json')}"
            messagebox.showinfo("Fertig", msg)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))


def main():
    if len(sys.argv) > 1:
        # CLI: excel_to_qprime.py <ExcelPfad> [B|C] [--solar]
        excel = Path(sys.argv[1]).resolve()
        use = "B"
        write_solar = False
        if len(sys.argv) >= 3 and sys.argv[2].upper() in ("B","C"):
            use = sys.argv[2].upper()
            extra_args = sys.argv[3:]
        else:
            extra_args = sys.argv[2:]
        for arg in extra_args:
            if arg == "--solar":
                write_solar = True
        run_noninteractive(excel, use_col=use, write_json=True, write_solar_only=write_solar)
    else:
        try:
            app = App()
            app.mainloop()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
