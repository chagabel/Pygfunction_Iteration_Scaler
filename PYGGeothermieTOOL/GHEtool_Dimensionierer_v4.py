#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GHEtool_Dimensionierer_v4 – L4 (hourly) mit Auto‑Defaults aus Data_Base.py

Änderungen ggü. v3:
- Lädt beim Start (und wenn der DB‑Pfad geändert wird) wieder automatisch
  Projekt‑Defaults aus Data_Base.py:
  * Bodenparameter: k_s, T_g, optional ρc (oder aus alpha/difusivity berechnet)
  * Komfort‑Voreinstellungen für Geometriefelder (falls vorhanden): Bx, By, D, r
    sowie fixe Tiefe H für den Tab „Tiefe fix“. Alle Werte bleiben editierbar.
- Rechnet ausschließlich mit GHEtool‑Sizing L4 (hourly), identisch zu v3.

Hinweis zur Nutzung:
- Für Erdlast‑CSV: „CSV ist Gebäudelast“ AUS lassen; SCOP/SEER = 0.
- Für Gebäude‑CSV: „CSV ist Gebäudelast“ AN und SCOP/SEER setzen.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import queue
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple, Callable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

# ----------------------- GHEtool-Import (neu/alt) -----------------------
try:
    from GHEtool import (
        Borefield,
        GroundConstantTemperature,
        HourlyGeothermalLoad,
        HourlyBuildingLoad,
    )
    GHETOOL_STYLE = "new"
except Exception:
    try:
        from GHEtool import Borefield, GroundData as GroundConstantTemperature  # type: ignore
        try:
            from GHEtool import HourlyGeothermalLoad  # type: ignore
        except Exception:
            HourlyGeothermalLoad = None  # type: ignore
        try:
            from GHEtool import HourlyBuildingLoad  # type: ignore
        except Exception:
            HourlyBuildingLoad = None  # type: ignore
        GHETOOL_STYLE = "old"
    except Exception as e:
        raise ImportError(
            "GHEtool konnte nicht importiert werden. Installiere: pip install GHEtool\n"
            f"Originalfehler: {e}"
        )


# -------------------------------- Utilities ------------------------------------
def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Kann Modul {module_name} aus {path} nicht laden.")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _load_hourly_csv(path_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    seps = [';', ',', '\t']
    decs = [',', '.']
    last_err: Optional[Exception] = None
    for sep in seps:
        for dec in decs:
            try:
                df = pd.read_csv(path_csv, sep=sep, decimal=dec)
                cols = {str(c).lower().strip(): c for c in df.columns}
                if 'hour' in cols and 'q_extraction_k_w' in cols:
                    # Robust gegen Tippfehler: erlauben sowohl q_extraction_kW als auch q_extraction_k_w
                    cols['q_extraction_kw'] = cols.pop('q_extraction_k_w')
                if 'hour' in cols and 'q_extraction_kw' in cols and 'q_injection_kw' in cols:
                    df = df.sort_values(by=[cols['hour']]).reset_index(drop=True)
                    q_ex = pd.to_numeric(df[cols['q_extraction_kw']], errors='coerce').fillna(0.0).to_numpy(float)
                    q_in = pd.to_numeric(df[cols['q_injection_kw']], errors='coerce').fillna(0.0).to_numpy(float)
                    return np.maximum(0.0, q_ex), np.maximum(0.0, q_in)
            except Exception as e:
                last_err = e
    raise ValueError(
        "CSV-Format nicht erkannt. Erwartet: hour;Q_extraction_kW;Q_injection_kW. "
        f"Letzter Fehler: {last_err}"
    )


def _convert_building_to_ground(q_heat: np.ndarray, q_cool: np.ndarray,
                                scop: float, seer: float, use_injection: bool) -> Tuple[np.ndarray, np.ndarray]:
    q_ex = q_heat.copy()
    if scop and scop > 0:
        fac = max(0.0, 1.0 - 1.0 / float(scop))
        q_ex = q_ex * fac
    q_in = q_cool.copy() if use_injection else np.zeros_like(q_cool)
    if use_injection and seer and seer > 0:
        fac = 1.0 + 1.0 / float(seer)
        q_in = q_in * fac
    return q_ex, q_in


def _build_load_from_ground_arrays(q_ex_kW: np.ndarray, q_in_kW: np.ndarray, years: int):
    q_ex = np.asarray(q_ex_kW, dtype=float).reshape(-1)
    q_in = np.asarray(q_in_kW, dtype=float).reshape(-1)
    try:
        if HourlyGeothermalLoad is not None:
            return HourlyGeothermalLoad(extraction=q_ex, injection=q_in, simulation_period=int(years))
    except Exception:
        pass
    last_err = None
    if HourlyBuildingLoad is not None:
        for kwargs in (
            dict(extraction=q_ex, injection=q_in, simulation_period=int(years)),
            dict(heating=q_ex,   cooling=q_in,    simulation_period=int(years)),
            dict(cooling=q_in,   heating=q_ex,    simulation_period=int(years)),
        ):
            try:
                return HourlyBuildingLoad(**kwargs)
            except Exception as e:
                last_err = e
                continue
        for args in ((q_ex, q_in, int(years)), (q_ex, q_in)):
            try:
                return HourlyBuildingLoad(*args)
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Konnte kein Hourly*-Load-Objekt erzeugen. Letzter Fehler: {last_err}")


def _make_borefield(load, k_s: float, T_g: float, rho_c: Optional[float], years: int,
                    Tmin: float, Tmax: float,
                    use_constant_Rb: bool, Rb_value_mK_per_W: float):
    ground = None
    for args in ((k_s, T_g, rho_c) if rho_c else (k_s, T_g), (k_s, T_g)):
        try:
            ground = GroundConstantTemperature(*args)  # type: ignore
            break
        except Exception:
            ground = None
    if ground is None:
        raise RuntimeError("GroundConstantTemperature konnte nicht erstellt werden.")

    bf = None
    for kwargs in (
        dict(load=load, ground_data=ground, simulation_period=years),
        dict(load=load, ground_data=ground),
        dict(simulation_period=years),
        dict(),
    ):
        try:
            bf = Borefield(**kwargs)  # type: ignore
            break
        except Exception:
            bf = None
    if bf is None:
        raise RuntimeError("Borefield konnte nicht erstellt werden.")

    try:
        if hasattr(bf, "ground_data") and getattr(bf, "ground_data") is None:
            setattr(bf, "ground_data", ground)
    except Exception:
        pass
    for setter in ("set_ground_parameters", "set_ground_data", "set_ground"):
        if hasattr(bf, setter):
            try:
                getattr(bf, setter)(ground)  # type: ignore
                break
            except Exception:
                pass

    if hasattr(bf, "simulation_period"):
        try:
            bf.simulation_period = years  # type: ignore
        except Exception:
            pass

    set_min = getattr(bf, "set_min_avg_fluid_temperature", None) or getattr(bf, "set_min_ground_temperature", None)
    set_max = getattr(bf, "set_max_avg_fluid_temperature", None) or getattr(bf, "set_max_ground_temperature", None)
    if set_min is None or set_max is None:
        raise RuntimeError("GHEtool-Version ohne Temperaturgrenzen-Setter.")
    set_min(float(Tmin)); set_max(float(Tmax))

    if hasattr(bf, "calculation_setup"):
        try:
            bf.calculation_setup(use_constant_Rb=bool(use_constant_Rb))  # type: ignore
        except Exception:
            pass
    if use_constant_Rb and hasattr(bf, "Rb"):
        try:
            bf.Rb = float(Rb_value_mK_per_W)  # type: ignore
        except Exception:
            pass

    try:
        if hasattr(bf, "load") and getattr(bf, "load") is None:
            setattr(bf, "load", load)
    except Exception:
        pass
    if hasattr(bf, "set_load"):
        try:
            bf.set_load(load)  # type: ignore
        except Exception:
            pass

    return bf


def _create_rect(borefield, nx: int, ny: int, Bx: float, By: float, H: float, D: float, r_b: float) -> None:
    try:
        borefield.create_rectangular_borefield(nx, ny, Bx, By, H, D, r_b)
        return
    except Exception:
        pass
    try:
        borefield.create_rectangular_borefield(nx, ny, Bx, By, H)
        return
    except Exception:
        pass
    raise


def _rectangular_search_L4(bf, Bx: float, By: float, H_fix: float, D: float, r_b: float,
                           nx_max: int, ny_max: int,
                           progress_cb: Optional[Callable[[int, int], None]] = None) -> Tuple[bool, int, int, float]:
    total = nx_max * ny_max
    done = 0
    best = None
    for nx in range(1, nx_max + 1):
        for ny in range(1, ny_max + 1):
            done += 1
            if progress_cb and (done % 3 == 0 or done == total):
                progress_cb(done, total)
            try:
                _create_rect(bf, nx, ny, Bx, By, H_fix, D, r_b)
                H_req = float(bf.size(100.0, L4_sizing=True))
            except Exception:
                continue
            N = nx * ny
            if H_req <= H_fix:
                if best is None or N < best[0] or (N == best[0] and H_req < best[3]):
                    best = (N, nx, ny, H_req)
    if best is None:
        return (False, 0, 0, float("nan"))
    _, nx_b, ny_b, H_b = best
    return (True, nx_b, ny_b, H_b)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GHEtool Dimensionierer v4 – L4 (hourly) + Defaults aus Data_Base")
        self.geometry("900x780")

        self.path_csv = tk.StringVar()
        self.path_db = tk.StringVar(value=str(Path("Data_Base.py")))
        self.k_s = tk.DoubleVar(value=2.0)
        self.Tg = tk.DoubleVar(value=10.0)
        self.rho_c = tk.StringVar(value="")
        self.years = tk.IntVar(value=50)
        self.Tmin = tk.DoubleVar(value=0.0)
        self.Tmax = tk.DoubleVar(value=16.0)

        self.scop = tk.DoubleVar(value=0.0)
        self.seer = tk.DoubleVar(value=0.0)
        self.use_injection = tk.BooleanVar(value=True)

        self.use_const_Rb = tk.BooleanVar(value=True)
        self.const_Rb_val = tk.DoubleVar(value=0.2)

        self.std_Bx = tk.DoubleVar(value=6.0)
        self.std_By = tk.DoubleVar(value=6.0)
        self.std_nx = tk.IntVar(value=5)
        self.std_ny = tk.IntVar(value=5)
        self.std_D = tk.DoubleVar(value=1.5)
        self.std_rb = tk.DoubleVar(value=0.075)

        self.fx_Bx = tk.DoubleVar(value=6.0)
        self.fx_By = tk.DoubleVar(value=6.0)
        self.fx_H = tk.DoubleVar(value=100.0)
        self.fx_D = tk.DoubleVar(value=1.5)
        self.fx_rb = tk.DoubleVar(value=0.075)
        self.fx_nxmax = tk.IntVar(value=20)
        self.fx_nymax = tk.IntVar(value=20)

        nb = ttk.Notebook(self)
        self.tab_std = ttk.Frame(nb)
        self.tab_fix = ttk.Frame(nb)
        nb.add(self.tab_std, text="Standard (H_req – L4)")
        nb.add(self.tab_fix, text="Tiefe fix (Layouts – L4)")
        nb.pack(fill="both", expand=True)

        self._build_common_header()
        self._build_tab_standard()
        self._build_tab_fixed()

        frm_prog = ttk.Frame(self, padding=(10, 4))
        frm_prog.pack(fill="x")
        self.progress = ttk.Progressbar(frm_prog, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", expand=True)
        self.status = tk.StringVar(value="Bereit.")
        ttk.Label(frm_prog, textvariable=self.status).pack(anchor="w", pady=(3, 6))

        self.txt = tk.Text(self, height=14, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._q: "queue.Queue" = queue.Queue()
        self._running = False
        self.after(120, self._poll)
        # Merker für letzte Berechnung (für DB-Übernahme)
        self._last_res: Optional[dict] = None

        # Auto‑Defaults aus Data_Base.py laden
        self._load_defaults_from_db()

    # ---------- UI ----------
    def _build_common_header(self):
        frm = ttk.LabelFrame(self, text="Dateien, Bodendaten & L4", padding=10)
        frm.pack(fill="x", padx=10, pady=8)

        row = 0
        ttk.Label(frm, text="Hourly load CSV:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.path_csv, width=60).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse…", command=self._pick_csv).grid(row=row, column=2); row += 1

        ttk.Label(frm, text="Data_Base.py:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.path_db, width=60).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse…", command=self._pick_db).grid(row=row, column=2); row += 1

        ttk.Label(frm, text="λ Boden [W/mK]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.k_s, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Tg [°C]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.Tg, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="ρc [J/m³K] (optional):").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.rho_c, width=14).grid(row=row, column=1, sticky="w"); row += 1

        ttk.Label(frm, text="Simulationsjahre:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.years, width=10).grid(row=row, column=1, sticky="w"); row += 1

        ttk.Label(frm, text="Grenzen avg fluid Tmin/Tmax [°C]:").grid(row=row, column=0, sticky="w")
        lim = ttk.Frame(frm); lim.grid(row=row, column=1, sticky="w")
        ttk.Entry(lim, textvariable=self.Tmin, width=8).pack(side="left")
        ttk.Label(lim, text=" / ").pack(side="left")
        ttk.Entry(lim, textvariable=self.Tmax, width=8).pack(side="left")
        row += 1

        sub = ttk.Frame(frm); sub.grid(row=row, column=0, columnspan=3, sticky="we", pady=(6, 0)); row += 1
        ttk.Label(sub, text="SCOP (Heizen):").grid(row=0, column=0, sticky="w")
        ttk.Entry(sub, textvariable=self.scop, width=8).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(sub, text="SEER (Kühlen):").grid(row=0, column=2, sticky="w")
        ttk.Entry(sub, textvariable=self.seer, width=8).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Checkbutton(sub, text="Rückspeisung verwenden", variable=self.use_injection).grid(row=0, column=4, sticky="w")

        sub2 = ttk.Frame(frm); sub2.grid(row=row, column=0, columnspan=3, sticky="we", pady=(6, 0)); row += 1
        ttk.Checkbutton(sub2, text="Rb*: konstant", variable=self.use_const_Rb).grid(row=0, column=0, sticky="w")
        ttk.Label(sub2, text="Rb* [mK/W]:").grid(row=0, column=1, sticky="w")
        ttk.Entry(sub2, textvariable=self.const_Rb_val, width=8).grid(row=0, column=2, sticky="w")

        # Last-Skalierung (optional)
        scale = ttk.LabelFrame(frm, text="Last-Skalierung (vor Sizing)")
        scale.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        row += 1
        self.scale_mode = tk.StringVar(value='original')        # 'original' | 'target'
        self.scale_ex_kwh = tk.StringVar(value='')              # Ziel kWh für Extraction (Erdlast)
        self.scale_in_mode = tk.StringVar(value='same')         # 'original' | 'same' | 'target'
        self.scale_in_kwh = tk.StringVar(value='')              # Ziel kWh für Injection
        self.scale_ref = tk.StringVar(value='ground')           # 'ground' | 'building'

        ttk.Radiobutton(scale, text="Original (keine Skalierung)", variable=self.scale_mode, value='original').grid(row=0, column=0, sticky='w', padx=(6,6), pady=(4,2))
        ttk.Radiobutton(scale, text="Ziel Jahresenergie (Extraction)", variable=self.scale_mode, value='target').grid(row=0, column=1, sticky='w', padx=(6,6), pady=(4,2))
        ttk.Label(scale, text="E_ex Ziel [kWh/a]:").grid(row=0, column=2, sticky='e')
        ttk.Entry(scale, textvariable=self.scale_ex_kwh, width=12).grid(row=0, column=3, sticky='w')

        ttk.Label(scale, text="Injection-Skalierung:").grid(row=1, column=0, sticky='w', padx=(6,6))
        ttk.Radiobutton(scale, text="original", variable=self.scale_in_mode, value='original').grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(scale, text="gleicher Faktor", variable=self.scale_in_mode, value='same').grid(row=1, column=2, sticky='w')
        ttk.Radiobutton(scale, text="Ziel [kWh/a]", variable=self.scale_in_mode, value='target').grid(row=1, column=3, sticky='w')
        ttk.Entry(scale, textvariable=self.scale_in_kwh, width=12).grid(row=1, column=4, sticky='w')
        ttk.Label(scale, text="Bezug:").grid(row=2, column=0, sticky='w', padx=(6,6))
        ttk.Radiobutton(scale, text="Erdlast (nach COP)", variable=self.scale_ref, value='ground').grid(row=2, column=1, sticky='w')
        ttk.Radiobutton(scale, text="Gebäudelast (vor COP)", variable=self.scale_ref, value='building').grid(row=2, column=2, sticky='w')

        frm.columnconfigure(1, weight=1)

        # Button: Ergebniswerte nach Data_Base.py übertragen
        ttk.Button(self, text="In Data_Base.py übernehmen", command=self._apply_to_db).pack(anchor="e", padx=12, pady=(2, 0))

    def _build_tab_standard(self):
        frm = ttk.LabelFrame(self.tab_std, text="Standard – H_req (L4)", padding=10)
        frm.pack(fill="x", padx=10, pady=8)
        row = 0
        ttk.Label(frm, text="Bx [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_Bx, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="By [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_By, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="nx").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_nx, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="ny").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_ny, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Burial depth D [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_D, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Borehole radius r_b [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.std_rb, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Button(frm, text="H_req berechnen (L4)", command=self._on_size_standard).grid(row=row, column=0, columnspan=2, pady=8)

    def _build_tab_fixed(self):
        frm = ttk.LabelFrame(self.tab_fix, text="Tiefe fix – Layoutsuche (L4)", padding=10)
        frm.pack(fill="x", padx=10, pady=8)
        row = 0
        ttk.Label(frm, text="Bx [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_Bx, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="By [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_By, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Fixe Tiefe H_fix [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_H, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Burial depth D [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_D, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="Borehole radius r_b [m]").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_rb, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="nx_max").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_nxmax, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Label(frm, text="ny_max").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fx_nymax, width=10).grid(row=row, column=1, sticky="w"); row += 1
        ttk.Button(frm, text="Layout finden (≤ H_fix, L4)", command=self._on_size_fixed).grid(row=row, column=0, columnspan=2, pady=8)

    # ---------- Defaults aus Data_Base ----------
    def _load_defaults_from_db(self):
        try:
            mod = _import_module_from_path("project_Data_Base_defaults_v4", Path(self.path_db.get()).expanduser())
        except Exception:
            return
        # Bodenparameter
        try:
            if hasattr(mod, "k_s"): self.k_s.set(float(getattr(mod, "k_s")))
        except Exception: pass
        try:
            if hasattr(mod, "T_g"): self.Tg.set(float(getattr(mod, "T_g")))
        except Exception: pass
        # rho_c direkt oder aus alpha/difusivity ableiten
        try:
            if hasattr(mod, "rho_c"):
                self.rho_c.set(f"{float(getattr(mod, 'rho_c')):.0f}")
            else:
                alpha = None
                if hasattr(mod, "alpha"): alpha = float(getattr(mod, "alpha"))
                elif hasattr(mod, "difusivity"): alpha = float(getattr(mod, "difusivity"))
                if alpha and float(self.k_s.get())>0:
                    rho_c = float(self.k_s.get()) / alpha
                    self.rho_c.set(f"{rho_c:.0f}")
        except Exception: pass
        # Komfort: Geometrie‑Defaults vorbelegen (editierbar)
        for attr, var in (("Bx", self.std_Bx), ("By", self.std_By), ("D", self.std_D)):
            try:
                if hasattr(mod, attr): var.set(float(getattr(mod, attr)))
            except Exception: pass
        try:
            if hasattr(mod, "r"): self.std_rb.set(float(getattr(mod, "r")))
        except Exception: pass
        # gleiche Defaults auch für „Tiefe fix“
        try:
            if hasattr(mod, "Bx"): self.fx_Bx.set(float(getattr(mod, "Bx")))
            if hasattr(mod, "By"): self.fx_By.set(float(getattr(mod, "By")))
            if hasattr(mod, "D"):  self.fx_D.set(float(getattr(mod, "D")))
            if hasattr(mod, "r"):  self.fx_rb.set(float(getattr(mod, "r")))
            if hasattr(mod, "H"):  self.fx_H.set(float(getattr(mod, "H")))
        except Exception: pass

    # ---------- File pickers ----------
    def _pick_csv(self):
        p = filedialog.askopenfilename(title="Choose hourly load CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p: self.path_csv.set(p)
    def _pick_db(self):
        p = filedialog.askopenfilename(title="Choose Data_Base.py", filetypes=[("Python", "*.py"), ("All files", "*.*")])
        if p:
            self.path_db.set(p)
            self._load_defaults_from_db()

    # ---------- Worker orchestration (identisch v3) ----------
    def _start_thread(self, target: Callable[[], None]):
        if self._running: return
        if not self.path_csv.get().strip():
            messagebox.showerror("Fehler", "Bitte eine CSV wählen.")
            return
        self._running = True
        self.status.set("Rechne…")
        self.progress.configure(mode="determinate"); self.progress["value"] = 0
        threading.Thread(target=target, daemon=True).start()

    def _on_size_standard(self):
        self._start_thread(self._worker_standard)
    def _on_size_fixed(self):
        self._start_thread(self._worker_fixed)

    def _worker_standard(self):
        try:
            b_ex, b_in = _load_hourly_csv(Path(self.path_csv.get()).expanduser())
            is_building = (float(self.scop.get()) > 0 or float(self.seer.get()) > 0)
            # Warnungen/Validierung
            # 1) Gebäudelast-Skalierung gewählt aber SCOP/SEER=0 → wirkungslos
            if self.scale_mode.get() == 'target' and self.scale_ref.get() == 'building' and not is_building:
                proceed = messagebox.askokcancel(
                    'Hinweis – Gebäudelast vor COP',
                    'Skalierung mit Bezug "Gebäudelast (vor COP)" gewählt, aber SCOP/SEER = 0.\n'
                    'Die Skalierung vor COP wäre wirkungslos.\n\n'
                    'OK = Bezug automatisch auf "Erdlast (nach COP)" umstellen\nAbbrechen = zurück zur Eingabe')
                if not proceed:
                    self._q.put(("error", "Abgebrochen durch Nutzer (Skalierungsbezug ungeeignet).")); return
                try:
                    self.scale_ref.set('ground')
                except Exception:
                    pass
            # 2) Rückspeisung-Check auf Gebäudeseite
            try:
                if bool(self.use_injection.get()) and float(np.sum(b_in)) <= 1e-9 and is_building:
                    messagebox.showwarning('Hinweis – Rückspeisung', 'Rückspeisung ist aktiviert, die Gebäude‑CSV enthält jedoch keine Kühl-/Injection‑Energie. Injection=0.')
            except Exception:
                pass
            # Last-Skalierung – wahlweise vor (Gebäude) oder nach (Erdlast) COP-Umrechnung
            # 1) ggf. Gebäudelast skalieren
            E_ex0_b = float(np.sum(b_ex))
            E_in0_b = float(np.sum(b_in))
            if is_building and self.scale_mode.get() == 'target' and self.scale_ref.get() == 'building':
                fac_ex_b = 1.0; fac_in_b = 1.0
                try:
                    target_ex = float(self.scale_ex_kwh.get().strip()) if self.scale_ex_kwh.get().strip() else 0.0
                    if target_ex > 0 and E_ex0_b > 0:
                        fac_ex_b = target_ex / E_ex0_b
                    mode_in = self.scale_in_mode.get()
                    if mode_in == 'same':
                        fac_in_b = fac_ex_b
                    elif mode_in == 'target':
                        target_in = float(self.scale_in_kwh.get().strip()) if self.scale_in_kwh.get().strip() else 0.0
                        if target_in > 0 and E_in0_b > 0:
                            fac_in_b = target_in / E_in0_b
                except Exception:
                    fac_ex_b = 1.0; fac_in_b = 1.0
                if fac_ex_b != 1.0:
                    b_ex = b_ex * fac_ex_b
                if fac_in_b != 1.0:
                    b_in = b_in * fac_in_b
            # 2) Gebäude -> Erdlast
            if is_building:
                q_ex, q_in = _convert_building_to_ground(b_ex, b_in, float(self.scop.get()), float(self.seer.get()), bool(self.use_injection.get()))
            else:
                q_ex, q_in = b_ex, b_in
                if not bool(self.use_injection.get()):
                    q_in = np.zeros_like(q_in)
            # 2b) Rückspeisung-Check auf Erdlastseite
            try:
                if bool(self.use_injection.get()) and float(np.sum(q_in)) <= 1e-9:
                    messagebox.showwarning('Hinweis – Rückspeisung', 'Rückspeisung ist aktiviert, die (Erdlast‑)CSV enthält jedoch keine Injection. Injection=0.')
            except Exception:
                pass
            # 3) ggf. Erdlast skalieren
            E_ex0 = float(np.sum(q_ex))
            E_in0 = float(np.sum(q_in))
            fac_ex = 1.0
            fac_in = 1.0
            try:
                if self.scale_mode.get() == 'target' and self.scale_ref.get() == 'ground':
                    target_ex = float(self.scale_ex_kwh.get().strip()) if self.scale_ex_kwh.get().strip() else 0.0
                    if target_ex > 0 and E_ex0 > 0:
                        fac_ex = target_ex / E_ex0
                # Injection:
                mode_in = self.scale_in_mode.get()
                if mode_in == 'same' and self.scale_ref.get() == 'ground':
                    fac_in = fac_ex
                elif mode_in == 'target' and self.scale_ref.get() == 'ground':
                    target_in = float(self.scale_in_kwh.get().strip()) if self.scale_in_kwh.get().strip() else 0.0
                    if target_in > 0 and E_in0 > 0:
                        fac_in = target_in / E_in0
                # 'original' -> fac_in=1.0
            except Exception:
                fac_ex = 1.0; fac_in = 1.0
            if fac_ex != 1.0:
                q_ex = q_ex * fac_ex
            if fac_in != 1.0:
                q_in = q_in * fac_in
            E_ex = float(np.sum(q_ex)); E_in = float(np.sum(q_in))
            peak_ex = float(np.max(q_ex)) if q_ex.size else 0.0
            peak_in = float(np.max(q_in)) if q_in.size else 0.0
            load = _build_load_from_ground_arrays(q_ex, q_in, int(self.years.get()))
            rho_c_val = None
            try:
                rho_c_val = float(self.rho_c.get().strip()) if self.rho_c.get().strip() else None
            except Exception:
                rho_c_val = None
            bf = _make_borefield(load, float(self.k_s.get()), float(self.Tg.get()), rho_c_val, int(self.years.get()), float(self.Tmin.get()), float(self.Tmax.get()), bool(self.use_const_Rb.get()), float(self.const_Rb_val.get()))
            _create_rect(bf, int(self.std_nx.get()), int(self.std_ny.get()), float(self.std_Bx.get()), float(self.std_By.get()), 100.0, float(self.std_D.get()), float(self.std_rb.get()))
            H_req = float(bf.size(100.0, L4_sizing=True))
            N = int(self.std_nx.get()) * int(self.std_ny.get())
            res = {
                "mode": "Standard L4",
                "nx": int(self.std_nx.get()),
                "ny": int(self.std_ny.get()),
                "N": N,
                "Bx": float(self.std_Bx.get()),
                "By": float(self.std_By.get()),
                "H_req_m": H_req,
                "Tmin_C": float(self.Tmin.get()),
                "Tmax_C": float(self.Tmax.get()),
                "years": int(self.years.get()),
                "k_s_WmK": float(self.k_s.get()),
                "Tg_C": float(self.Tg.get()),
                "rho_c_Jm3K": rho_c_val,
                "SCOP": float(self.scop.get()),
                "SEER": float(self.seer.get()),
                "use_injection": bool(self.use_injection.get()),
                "Rb_value_mK_per_W": float(self.const_Rb_val.get()) if self.use_const_Rb.get() else None,
                "GHEtool_style": GHETOOL_STYLE,
                # Zusatz: Last-Report
                "E_ex_kWh": E_ex,
                "E_in_kWh": E_in,
                "scale_ex": fac_ex,
                "scale_in": fac_in,
                "peak_ex_kW": peak_ex,
                "peak_in_kW": peak_in,
                "scale_ref": self.scale_ref.get(),
                "E_ex_build_kWh": float(np.sum(b_ex)) if is_building else None,
                "E_in_build_kWh": float(np.sum(b_in)) if is_building else None,
            }
            out_base = Path(self.path_csv.get()).with_suffix("")
            json_path = out_base.parent / f"{out_base.name}__ghe_standard_v4_L4.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            self._q.put(("result_std", res, json_path))
        except Exception as e:
            self._q.put(("error", str(e)))

    def _worker_fixed(self):
        try:
            b_ex, b_in = _load_hourly_csv(Path(self.path_csv.get()).expanduser())
            is_building = (float(self.scop.get()) > 0 or float(self.seer.get()) > 0)
            # Warnungen/Validierung
            if self.scale_mode.get() == 'target' and self.scale_ref.get() == 'building' and not is_building:
                proceed = messagebox.askokcancel(
                    'Hinweis – Gebäudelast vor COP',
                    'Skalierung mit Bezug "Gebäudelast (vor COP)" gewählt, aber SCOP/SEER = 0.\n'
                    'Die Skalierung vor COP wäre wirkungslos.\n\n'
                    'OK = Bezug automatisch auf "Erdlast (nach COP)" umstellen\nAbbrechen = zurück zur Eingabe')
                if not proceed:
                    self._q.put(("error", "Abgebrochen durch Nutzer (Skalierungsbezug ungeeignet).")); return
                try:
                    self.scale_ref.set('ground')
                except Exception:
                    pass
            try:
                if bool(self.use_injection.get()) and float(np.sum(b_in)) <= 1e-9 and is_building:
                    messagebox.showwarning('Hinweis – Rückspeisung', 'Rückspeisung ist aktiviert, die Gebäude‑CSV enthält jedoch keine Kühl-/Injection‑Energie. Injection=0.')
            except Exception:
                pass
            # Gebäude-Skalierung vor Umrechnung (optional)
            E_ex0_b = float(np.sum(b_ex)); E_in0_b = float(np.sum(b_in))
            if is_building and self.scale_mode.get() == 'target' and self.scale_ref.get() == 'building':
                fac_ex_b = 1.0; fac_in_b = 1.0
                try:
                    target_ex = float(self.scale_ex_kwh.get().strip()) if self.scale_ex_kwh.get().strip() else 0.0
                    if target_ex > 0 and E_ex0_b > 0:
                        fac_ex_b = target_ex / E_ex0_b
                    mode_in = self.scale_in_mode.get()
                    if mode_in == 'same':
                        fac_in_b = fac_ex_b
                    elif mode_in == 'target':
                        target_in = float(self.scale_in_kwh.get().strip()) if self.scale_in_kwh.get().strip() else 0.0
                        if target_in > 0 and E_in0_b > 0:
                            fac_in_b = target_in / E_in0_b
                except Exception:
                    fac_ex_b = 1.0; fac_in_b = 1.0
                if fac_ex_b != 1.0:
                    b_ex = b_ex * fac_ex_b
                if fac_in_b != 1.0:
                    b_in = b_in * fac_in_b
            # Umrechnung in Erdlast
            if is_building:
                q_ex, q_in = _convert_building_to_ground(b_ex, b_in, float(self.scop.get()), float(self.seer.get()), bool(self.use_injection.get()))
            else:
                q_ex, q_in = b_ex, b_in
                if not bool(self.use_injection.get()):
                    q_in = np.zeros_like(q_in)
            try:
                if bool(self.use_injection.get()) and float(np.sum(q_in)) <= 1e-9:
                    messagebox.showwarning('Hinweis – Rückspeisung', 'Rückspeisung ist aktiviert, die (Erdlast‑)CSV enthält jedoch keine Injection. Injection=0.')
            except Exception:
                pass
            # Erdlast-Skalierung nach Umrechnung (optional)
            E_ex0 = float(np.sum(q_ex)); E_in0 = float(np.sum(q_in))
            fac_ex = 1.0; fac_in = 1.0
            try:
                if self.scale_mode.get() == 'target' and self.scale_ref.get() == 'ground':
                    target_ex = float(self.scale_ex_kwh.get().strip()) if self.scale_ex_kwh.get().strip() else 0.0
                    if target_ex > 0 and E_ex0 > 0:
                        fac_ex = target_ex / E_ex0
                mode_in = self.scale_in_mode.get()
                if mode_in == 'same' and self.scale_ref.get() == 'ground':
                    fac_in = fac_ex
                elif mode_in == 'target' and self.scale_ref.get() == 'ground':
                    target_in = float(self.scale_in_kwh.get().strip()) if self.scale_in_kwh.get().strip() else 0.0
                    if target_in > 0 and E_in0 > 0:
                        fac_in = target_in / E_in0
            except Exception:
                fac_ex = 1.0; fac_in = 1.0
            if fac_ex != 1.0:
                q_ex = q_ex * fac_ex
            if fac_in != 1.0:
                q_in = q_in * fac_in
            E_ex = float(np.sum(q_ex)); E_in = float(np.sum(q_in))
            peak_ex = float(np.max(q_ex)) if q_ex.size else 0.0
            peak_in = float(np.max(q_in)) if q_in.size else 0.0
            load = _build_load_from_ground_arrays(q_ex, q_in, int(self.years.get()))
            rho_c_val = None
            try:
                rho_c_val = float(self.rho_c.get().strip()) if self.rho_c.get().strip() else None
            except Exception:
                rho_c_val = None
            bf = _make_borefield(load, float(self.k_s.get()), float(self.Tg.get()), rho_c_val, int(self.years.get()), float(self.Tmin.get()), float(self.Tmax.get()), bool(self.use_const_Rb.get()), float(self.const_Rb_val.get()))
            feasible, nx, ny, Hreq = _rectangular_search_L4(bf, float(self.fx_Bx.get()), float(self.fx_By.get()), float(self.fx_H.get()), float(self.fx_D.get()), float(self.fx_rb.get()), int(self.fx_nxmax.get()), int(self.fx_nymax.get()), progress_cb=lambda d, t: self._q.put(("progress", d, t)))
            N = nx * ny
            res = {
                "mode": "Tiefe fix L4",
                "feasible": bool(feasible),
                "nx": int(nx),
                "ny": int(ny),
                "N": int(N),
                "Bx": float(self.fx_Bx.get()),
                "By": float(self.fx_By.get()),
                "H_fix_m": float(self.fx_H.get()),
                "H_req_best_m": float(Hreq) if Hreq == Hreq else None,
                "Tmin_C": float(self.Tmin.get()),
                "Tmax_C": float(self.Tmax.get()),
                "years": int(self.years.get()),
                "k_s_WmK": float(self.k_s.get()),
                "Tg_C": float(self.Tg.get()),
                "rho_c_Jm3K": rho_c_val,
                "SCOP": float(self.scop.get()),
                "SEER": float(self.seer.get()),
                "use_injection": bool(self.use_injection.get()),
                "Rb_value_mK_per_W": float(self.const_Rb_val.get()) if self.use_const_Rb.get() else None,
                "GHEtool_style": GHETOOL_STYLE,
                # Zusatz
                "E_ex_kWh": E_ex,
                "E_in_kWh": E_in,
                "scale_ex": fac_ex,
                "scale_in": fac_in,
                "peak_ex_kW": peak_ex,
                "peak_in_kW": peak_in,
                "scale_ref": self.scale_ref.get(),
                "E_ex_build_kWh": float(np.sum(b_ex)) if is_building else None,
                "E_in_build_kWh": float(np.sum(b_in)) if is_building else None,
            }
            out_base = Path(self.path_csv.get()).with_suffix("")
            json_path = out_base.parent / f"{out_base.name}__ghe_fixed_depth_v4_L4.json"
            csv_path = out_base.parent / f"{out_base.name}__ghe_fixed_depth_v4_L4.csv"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["feasible", "nx", "ny", "N", "Bx", "By", "H_fix_m", "H_req_best_m", "k_s_WmK", "Tg_C", "rho_c_Jm3K", "Tmin_C", "Tmax_C", "years", "SCOP", "SEER", "use_injection", "Rb_value_mK_per_W"])
                w.writerow([res["feasible"], res["nx"], res["ny"], res["N"], res["Bx"], res["By"], res["H_fix_m"], res["H_req_best_m"], res["k_s_WmK"], res["Tg_C"], res["rho_c_Jm3K"] or "", res["Tmin_C"], res["Tmax_C"], res["years"], res["SCOP"], res["SEER"], int(res["use_injection"]), res["Rb_value_mK_per_W"] or ""])            
            self._q.put(("result_fix", res, csv_path, json_path))
        except Exception as e:
            self._q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                kind, *payload = self._q.get_nowait()
                if kind == "progress":
                    done, total = payload
                    pct = max(0, min(100, int(done / max(1, total) * 100)))
                    self.progress.configure(mode="determinate"); self.progress["value"] = pct
                    self.status.set(f"Suche… {pct}%")
                elif kind == "result_std":
                    res, json_path = payload
                    self.progress["value"] = 100
                    self.status.set("Fertig (Standard L4).")
                    self._show_standard(res, json_path)
                    try:
                        self._last_res = dict(res)
                        self._last_res["_mode_ui"] = "standard"
                    except Exception:
                        pass
                    self._running = False
                elif kind == "result_fix":
                    res, csv_path, json_path = payload
                    self.progress["value"] = 100
                    self.status.set("Fertig (Tiefe fix L4).")
                    self._show_fixed(res, csv_path, json_path)
                    try:
                        self._last_res = dict(res)
                        self._last_res["_mode_ui"] = "fixed"
                    except Exception:
                        pass
                    self._running = False
                elif kind == "error":
                    self.progress["value"] = 0
                    self.status.set("Fehler.")
                    self._running = False
                    messagebox.showerror("Fehler", payload[0])
        except queue.Empty:
            pass
        self.after(120, self._poll)

    # ---------- Output ----------
    def _show_standard(self, res: dict, json_path: Path):
        lines = []
        lines.append("=== Standard-GHEtool (v4 – L4) ===")
        lines.append(f"Layout: {res['nx']} × {res['ny']}  (N={res['N']})  Bx/By = {res['Bx']:.2f}/{res['By']:.2f} m")
        lines.append(f"Erforderliche Tiefe: H_req = {res['H_req_m']:.1f} m")
        # Lastreport
        try:
            qavg = (res['E_ex_kWh']*1000.0)/(res['N']*res['H_req_m']*8760.0) if res.get('E_ex_kWh') and res['N']>0 and res['H_req_m']>0 else None
            lines.append(f"Lasten (nach Skalierung): E_ex={res.get('E_ex_kWh',0):,.0f} kWh/a, E_in={res.get('E_in_kWh',0):,.0f} kWh/a".replace(',', 'X').replace('.', ',').replace('X','.'))
            lines.append(f"Faktoren: ex={res.get('scale_ex',1.0):.4f}, in={res.get('scale_in',1.0):.4f} | Peaks: ex={res.get('peak_ex_kW',0):.2f} kW, in={res.get('peak_in_kW',0):.2f} kW")
            if qavg is not None:
                lines.append(f"q′_avg ≈ {qavg:.3f} W/m (bezogen auf N·H_req)")
        except Exception:
            pass
        lines.append(f"Grenzen avg fluid: Tmin/Tmax = {res['Tmin_C']:.1f}/{res['Tmax_C']:.1f} °C – Jahre: {res['years']}")
        lines.append(f"Boden: λ = {res['k_s_WmK']:.2f} W/mK, Tg = {res['Tg_C']:.1f} °C, ρc = {res['rho_c_Jm3K'] if res['rho_c_Jm3K'] else '–'}")
        lines.append(f"Umrechnung: SCOP = {res['SCOP']:.2f}, SEER = {res['SEER']:.2f}, Rückspeisung={'an' if res['use_injection'] else 'aus'}")
        lines.append(f"Rb*: {res['Rb_value_mK_per_W'] if res['Rb_value_mK_per_W'] is not None else 'n/a'} mK/W")
        lines.append(f"GHEtool: {res['GHEtool_style']}")
        lines.append("")
        lines.append(f"JSON: {json_path}")
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(lines))

    def _show_fixed(self, res: dict, csv_path: Path, json_path: Path):
        lines = []
        lines.append("=== Tiefe fix – Layoutsuche (v4 – L4) ===")
        lines.append(f"Fixe Tiefe: H_fix = {res['H_fix_m']:.1f} m")
        lines.append(f"Ergebnis: Machbarkeit = {'JA' if res['feasible'] else 'NEIN'}")
        if res['feasible']:
            lines.append(f"Rechteckfeld: {res['nx']} × {res['ny']}  (N={res['N']}) – Bx/By = {res['Bx']:.2f}/{res['By']:.2f} m")
            if res['H_req_best_m'] is not None:
                lines.append(f"Errechnete Tiefe für bestes Layout: H_req = {res['H_req_best_m']:.1f} m")
        # Lastreport
        try:
            qavg = (res['E_ex_kWh']*1000.0)/(max(1,res['N'])*max(1.0,res['H_fix_m'])*8760.0) if res.get('E_ex_kWh') else None
            lines.append(f"Lasten (nach Skalierung): E_ex={res.get('E_ex_kWh',0):,.0f} kWh/a, E_in={res.get('E_in_kWh',0):,.0f} kWh/a".replace(',', 'X').replace('.', ',').replace('X','.'))
            lines.append(f"Faktoren: ex={res.get('scale_ex',1.0):.4f}, in={res.get('scale_in',1.0):.4f} | Peaks: ex={res.get('peak_ex_kW',0):.2f} kW, in={res.get('peak_in_kW',0):.2f} kW")
            if qavg is not None:
                lines.append(f"q′_avg ≈ {qavg:.3f} W/m (bezogen auf N·H_fix)")
        except Exception:
            pass
        lines.append(f"Grenzen avg fluid: Tmin/Tmax = {res['Tmin_C']:.1f}/{res['Tmax_C']:.1f} °C – Jahre: {res['years']}")
        lines.append(f"Boden: λ = {res['k_s_WmK']:.2f} W/mK, Tg = {res['Tg_C']:.1f} °C, ρc = {res['rho_c_Jm3K'] if res['rho_c_Jm3K'] else '–'}")
        lines.append(f"Umrechnung: SCOP = {res['SCOP']:.2f}, SEER = {res['SEER']:.2f}, Rückspeisung={'an' if res['use_injection'] else 'aus'}")
        lines.append(f"Rb*: {res['Rb_value_mK_per_W'] if res['Rb_value_mK_per_W'] is not None else 'n/a'} mK/W")
        lines.append(f"GHEtool: {res['GHEtool_style']}")
        lines.append("")
        lines.append(f"CSV: {csv_path}")
        lines.append(f"JSON: {json_path}")
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(lines))

    # ---------- Data_Base.py übernehmen ----------
    def _apply_to_db(self):
        if not getattr(self, '_last_res', None):
            messagebox.showwarning('Hinweis', 'Bitte zuerst eine Berechnung durchführen.')
            return
        res = self._last_res or {}
        mode = str(res.get('mode',''))
        try:
            if mode.startswith('Standard'):
                nx = int(res.get('nx', int(self.std_nx.get())))
                ny = int(res.get('ny', int(self.std_ny.get())))
                Bx = float(res.get('Bx', float(self.std_Bx.get())))
                By = float(res.get('By', float(self.std_By.get())))
                H  = float(res.get('H_req_m', float(self.fx_H.get())))
                D  = float(self.std_D.get())
                r_b= float(self.std_rb.get())
            else:
                nx = int(res.get('nx', 0))
                ny = int(res.get('ny', 0))
                Bx = float(res.get('Bx', float(self.fx_Bx.get())))
                By = float(res.get('By', float(self.fx_By.get())))
                H  = float(res.get('H_fix_m', float(self.fx_H.get())))
                D  = float(self.fx_D.get())
                r_b= float(self.fx_rb.get())
        except Exception as e:
            messagebox.showerror('Fehler', f'Werte konnten nicht gelesen werden: {e}')
            return

        def _patch_db_file(path: Path, updates: dict) -> bool:
            try:
                txt = path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                messagebox.showerror('Fehler', f'Lesefehler {path}: {e}')
                return False
            pats = {
                'rows_sonden': r'^\s*rows_sonden\s*=\s*([0-9]+)',
                'Columns_sonden': r'^\s*Columns_sonden\s*=\s*([0-9]+)',
                'Bx': r'^\s*Bx\s*=\s*([0-9eE\.,\+\-]+)',
                'By': r'^\s*By\s*=\s*([0-9eE\.,\+\-]+)',
                'H': r'^\s*H\s*=\s*([0-9eE\.,\+\-]+)',
                'D': r'^\s*D\s*=\s*([0-9eE\.,\+\-]+)',
                'r': r'^\s*r\s*=\s*([0-9eE\.,\+\-]+)',
            }
            new_txt = txt
            for key, pat in pats.items():
                if key not in updates:
                    continue
                val = updates[key]
                rep = f"{key} = {float(val):.6g}" if isinstance(val, float) else f"{key} = {val}"
                new_txt = re.sub(pat, rep, new_txt, flags=re.MULTILINE)
            try:
                path.write_text(new_txt, encoding='utf-8')
                return True
            except Exception as e:
                messagebox.showerror('Fehler', f'Schreibfehler {path}: {e}')
                return False

        updates = {'rows_sonden': nx, 'Columns_sonden': ny, 'Bx': Bx, 'By': By, 'H': H, 'D': D, 'r': r_b}
        db_path = Path('Data_Base.py').resolve()
        ok = _patch_db_file(db_path, updates)
        if ok:
            info = (f"Übernommen nach Data_Base.py:\n"
                    f"rows_sonden={nx}, Columns_sonden={ny}\n"
                    f"Bx={Bx:.3f} m, By={By:.3f} m, H={H:.2f} m\n"
                    f"D={D:.2f} m, r={r_b:.3f} m")
            try:
                messagebox.showinfo('Gespeichert', info)
            except Exception:
                pass
        else:
            messagebox.showerror('Fehler', 'Konnte Data_Base.py nicht aktualisieren.')


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
