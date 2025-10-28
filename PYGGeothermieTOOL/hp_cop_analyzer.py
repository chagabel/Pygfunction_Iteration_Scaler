#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hp_cop_analyzer.py – COP-Analyse aus q′(t) [W/m]

Funktion
- Liest ein stündliches Linienlast-Profil q′(t) [W/m] (8760/h) aus CSV/XLSX
  – exakt das Exportformat aus dem Longterm-Checker (hour;W_per_m).
- Rechnet die Quellentemperaturen (T_b, EWT≈T_in, LWT≈T_out) für YEARS Jahre
  über pygfunction + Rohrmodell (wie in v13 – CJ standardmäßig).
- Berechnet zeitaufgelöst den Carnot-COP (Heizen) und einen realen COP mit
  Gütegrad η.
- Zeigt zwei Plots (COP_Carnot, COP_real) und Median/arithm. Mittel über die
  Heizstunden (Q_ground>0).

Hinweis
- COP-Berechnung erfolgt nur für Heizstunden (Q_ground>0). Für Rückspeisung
  (Q_ground≤0) werden COP-Werte als NaN ignoriert.
"""
from __future__ import annotations

import os
import csv
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Backend robust setzen (Spyder/Inline vermeiden)
try:
    if 'MPLBACKEND' in os.environ:
        mb = os.environ.get('MPLBACKEND','')
        if 'inline' in mb.lower() or mb.startswith('module://'):
            os.environ.pop('MPLBACKEND', None)
            os.environ['MPLBACKEND'] = 'TkAgg'
except Exception:
    pass
import matplotlib.pyplot as plt
try:
    plt.ion()
except Exception:
    pass

import pygfunction as gt
import Data_Base as bd
import borefields as bf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def _read_qprime_csv_or_xlsx(path: Path) -> np.ndarray:
    """Liest q′(t) [W/m] (8760 Zeilen). Unterstützt CSV (auto delimiter) und Excel.
    CSV: erwartet Spalte 'W_per_m'.
    Excel: nimmt 'profile_prev_year' oder 1. Blatt und sucht Spalte 'W_per_m'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profil nicht gefunden: {p}")
    if p.suffix.lower() in ('.xlsx', '.xlsm', '.xls'):
        import pandas as pd
        try:
            df = pd.read_excel(p, sheet_name='profile_prev_year')
        except Exception:
            df = pd.read_excel(p)
        col = None
        for c in df.columns:
            if str(c).strip().lower() == 'w_per_m':
                col = c; break
        if col is None:
            # Fallback: zweite Spalte
            col = df.columns[1] if df.shape[1] >= 2 else df.columns[0]
        q = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    else:
        rows = []
        with p.open('r', encoding='utf-8-sig') as fh:
            first = fh.readline(); delim = ';' if ';' in first and ',' not in first else ','; fh.seek(0)
            rdr = csv.DictReader(fh, delimiter=delim)
            wkey = None
            for k in (rdr.fieldnames or []):
                if str(k).strip().lower() == 'w_per_m': wkey = k; break
            if wkey is None:
                raise RuntimeError(f"CSV muss 'W_per_m' enthalten. Header: {rdr.fieldnames}")
            for row in rdr:
                try:
                    v = float((row.get(wkey,'') or '').strip())
                except Exception:
                    v = 0.0
                rows.append(v)
        q = np.array(rows, dtype=float)
    if q.size != 8760:
        raise RuntimeError(f"Erwarte 8760 Zeilen, gefunden: {q.size}")
    return q


def _choose_field(name: str):
    """Gibt (name, list[Borehole]) zurück.
    - rectangle/ushaped: konvertiert pygfunction.Borefield → Liste von Boreholes
    - freeform: nutzt borefields.freeform_field oder liest CSV aus ./autoborehole
    """
    nm = (name or '').lower().strip()
    # Freeform aus CSV/Liste
    if nm in ('freeform','freiform','polygon','auto'):
        try:
            holes = list(getattr(bf, 'freeform_field', []) or [])
            if not holes:
                # manuelles Laden aus CSV (Fallback)
                from pygfunction.boreholes import Borehole
                csvp = Path(os.getcwd()) / 'autoborehole' / 'borefield_polygon_points.csv'
                if not csvp.exists():
                    raise FileNotFoundError('autoborehole/borefield_polygon_points.csv nicht gefunden')
                with csvp.open('r', encoding='utf-8') as fh:
                    rdr = csv.DictReader(fh)
                    xk = yk = None
                    if rdr.fieldnames:
                        for k in rdr.fieldnames:
                            lk = (k or '').strip().lower()
                            if lk == 'x_m': xk = k
                            if lk == 'y_m': yk = k
                    assert xk and yk, 'CSV muss Spalten x_m,y_m enthalten.'
                    for row in rdr:
                        try:
                            x = float(str(row.get(xk,'0')).replace(',', '.'))
                            y = float(str(row.get(yk,'0')).replace(',', '.'))
                            holes.append(Borehole(float(bd.H), float(bd.D), float(bd.r), x, y))
                        except Exception:
                            continue
            return 'freeform_field', holes
        except Exception:
            # Fallback auf Rechteck
            nm = 'rectangle'

    # Rechteck/U: Borefield → Boreholes
    try:
        if nm in ('ushaped','u_shaped','u-shaped','u'):
            obj = getattr(bf, 'U_shaped_field', None)
            holes = obj.to_boreholes() if obj is not None and hasattr(obj, 'to_boreholes') else list(obj)
            return 'U_shaped_field', holes
        else:
            obj = getattr(bf, 'rectangle_field', None)
            holes = obj.to_boreholes() if obj is not None and hasattr(obj, 'to_boreholes') else list(obj)
            return 'rectangle_field', holes
    except Exception:
        # Letzte Rettung: freie Liste (leer)
        return 'rectangle_field', []


def _build_pipe_model(pipe_type: str, borehole, m_flow_bh: float):
    """Kopiert die Rohrmodell-Logik aus v13 (vereinfachtes Pendant)."""
    if (pipe_type or 'utube').lower() == 'coaxial':
        r_in_in   = float(getattr(bd,'r_in_in'))
        r_in_out  = float(getattr(bd,'r_in_out'))
        r_out_in  = float(getattr(bd,'r_out_in'))
        r_out_out = float(getattr(bd,'r_out_out'))
        k_p_in = float(getattr(bd,'k_p'))
        k_p_out = k_p_in
        R_p_in  = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in_in,  r_in_out,  k_p_in)
        R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(r_out_in, r_out_out, k_p_out)
        h_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_bh, r_in_in, bd.fluid_viscosity, bd.fluid_density,
            bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon)
        R_f_in = 1.0/(h_in*2.0*np.pi*r_in_in)
        h_a_in, h_a_out = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            m_flow_bh, r_in_out, r_out_in, bd.fluid_viscosity, bd.fluid_density,
            bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon)
        R_f_out_in  = 1.0/(h_a_in *2.0*np.pi*r_in_out)
        R_f_out_out = 1.0/(h_a_out*2.0*np.pi*r_out_in)
        R_ff = R_f_in + R_p_in + R_f_out_in
        R_fp = R_p_out + R_f_out_out
        pipe = gt.pipes.Coaxial(bd.pos_coaxial, np.array([r_out_in, r_in_in]), np.array([r_out_out, r_in_out]),
                                borehole, bd.k_s, bd.k_g, R_ff, R_fp)
        Rb_eff = pipe.effective_borehole_thermal_resistance(m_flow_bh, bd.fluid_isobaric_heatcapacity)
        return pipe, Rb_eff
    else:
        R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(bd.rp_in, bd.rp_out, bd.k_p)
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_bh, bd.rp_in, bd.fluid_viscosity, bd.fluid_density,
            bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon)
        R_f = 1.0/(h_f*2.0*np.pi*bd.rp_in)
        tmp = gt.pipes.SingleUTube(bd.pos_single, bd.rp_in, bd.rp_out, borehole, bd.k_s, bd.k_g, R_f+R_p)
        Rb_eff = tmp.effective_borehole_thermal_resistance(m_flow_bh, bd.fluid_isobaric_heatcapacity)
        pipe = gt.pipes.SingleUTube(bd.pos_single, bd.rp_in, bd.rp_out, borehole, bd.k_s, bd.k_g, Rb_eff)
        return pipe, Rb_eff


def simulate_cop(qprime_wpm_year: np.ndarray,
                 years: int,
                 field_name: str = 'rectangle',
                 folding_mode: str = 'CJ',
                 gf_method: str = 'equivalent',
                 boundary: str = 'UBWT',
                 pipe_type: str = 'utube',
                 dt: int = 3600,
                 T_supply_hot_C: float = 60.0,
                 dT_cond_K: float = 2.0,
                 dT_evap_K: float = 5.0,
                 eta_carnot: float = 0.5,
                 progress: Optional[callable] = None,
                 cancel_evt: Optional[threading.Event] = None):
    """Rechnet YEARS*8760 Zeitschritte und liefert COP-Zeitreihen + Temperaturen.
    Returns dict mit: COP_carnot, COP_real, T_b, T_in, T_out, q_ground, n_steps_year
    """
    # Feld
    fname, field = _choose_field(field_name)
    n_bh = int(len(list(field))) if not isinstance(field, (list, tuple)) else len(field)
    H = float(bd.H); L_total = max(1.0, n_bh * H)

    # Serien
    qprime_year = np.asarray(qprime_wpm_year, dtype=float)  # W/m
    if qprime_year.size != 8760:
        raise RuntimeError('Profil muss 8760 Werte haben.')
    n_steps_year = int(np.round(8760.0 * 3600.0 / dt))
    if n_steps_year != 8760:
        raise RuntimeError('dt muss 3600 s teilen (hourly).')

    qprime_series = np.tile(qprime_year, years)
    total_steps = years * n_steps_year
    Q_ground_series = qprime_series * L_total  # W

    # Rohrmodell / Massenstrom pro Sonde
    m_flow_bh = max(1e-9, float(bd.flow) / max(1, n_bh))
    borehole0 = field[0] if n_bh > 0 else None
    Pipe, _ = _build_pipe_model(pipe_type, borehole0, m_flow_bh)
    cp = float(bd.fluid_isobaric_heatcapacity)

    # Ausgabe-Arrays
    T_b   = np.zeros(total_steps)
    T_in  = np.zeros(total_steps)
    T_out = np.zeros(total_steps)

    def _prog(i, label):
        if progress:
            try:
                progress(i, total_steps, label)
            except Exception:
                pass

    # Faltung
    if folding_mode.lower() == 'hourly':
        time_full = dt * np.arange(1, total_steps + 1)
        if progress: progress(0, 1, 'g-function hourly …')
        g_full = gt.gfunction.gFunction(field, alpha=bd.difusivity, time=time_full,
                                        method=gf_method, boundary_condition=boundary, options=bd.options).gFunc
        if cancel_evt and cancel_evt.is_set():
            raise RuntimeError('Abgebrochen')
        dg = np.diff(g_full, prepend=0.0)
        inj_W = -qprime_series * L_total
        conv = np.convolve(inj_W, dg, mode='full')[:total_steps]
        T_b[:] = conv/(2.0*np.pi*bd.k_s)/(H*n_bh) + bd.T_g
        for i in range(total_steps):
            if cancel_evt and cancel_evt.is_set(): raise RuntimeError('Abgebrochen')
            Q_bh = (qprime_series[i]*L_total)/max(1,n_bh)
            T_in[i]  = Pipe.get_inlet_temperature(Q_bh, T_b[i], m_flow_bh, cp)
            T_out[i] = Pipe.get_outlet_temperature(T_in[i], T_b[i], m_flow_bh, cp)
            _prog(i, 'hourly')
    else:
        # Aggregatoren CJ/Liu/MLAA
        if folding_mode.lower() == 'liu':
            LoadAgg = gt.load_aggregation.Liu(dt, years*8760*dt)
        elif folding_mode.lower() == 'mlaa':
            LoadAgg = gt.load_aggregation.MLAA(dt, years*8760*dt)
        else:
            LoadAgg = gt.load_aggregation.ClaessonJaved(dt, years*8760*dt)
        time_req = LoadAgg.get_times_for_simulation()
        if progress: progress(0, 1, f'g-function {folding_mode} …')
        gfunc = gt.gfunction.gFunction(field, alpha=bd.difusivity, time=time_req,
                                       boundary_condition=boundary, options=bd.options, method=gf_method)
        if cancel_evt and cancel_evt.is_set():
            raise RuntimeError('Abgebrochen')
        LoadAgg.initialize(gfunc.gFunc/(2.0*np.pi*bd.k_s))
        for i in range(total_steps):
            if cancel_evt and cancel_evt.is_set(): raise RuntimeError('Abgebrochen')
            LoadAgg.next_time_step(int((i+1)*dt))
            qprime = float(qprime_series[i])
            LoadAgg.set_current_load(qprime)
            dT_b = LoadAgg.temporal_superposition(); dT_b = float(np.ravel(dT_b)[0])
            T_b[i] = bd.T_g - dT_b
            Q_bh = (qprime*L_total)/max(1,n_bh)
            T_in[i]  = Pipe.get_inlet_temperature(Q_bh, T_b[i], m_flow_bh, cp)
            T_out[i] = Pipe.get_outlet_temperature(T_in[i], T_b[i], m_flow_bh, cp)
            _prog(i, folding_mode)

    # COP-Berechnung (nur für Heizstunden: Q_ground>0)
    Th = (T_supply_hot_C + dT_cond_K) + 273.15
    Tc = (T_in - dT_evap_K) + 273.15
    dT = np.maximum(Th - Tc, 1.0)  # Sicherheitsuntergrenze 1 K
    cop_carnot = Th / dT
    cop_real   = eta_carnot * cop_carnot
    heater_mask = (Q_ground_series > 0.0)
    cop_carnot[~heater_mask] = np.nan
    cop_real[~heater_mask]   = np.nan

    return dict(COP_carnot=cop_carnot, COP_real=cop_real,
                T_b=T_b, T_in=T_in, T_out=T_out,
                q_ground=Q_ground_series, n_steps_year=n_steps_year)


# -----------------------------
# GUI
# -----------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('HP COP Analyzer – q′(t) → COP(t)')
        self.geometry('980x600')

        self.path_profile = tk.StringVar()
        self.years   = tk.IntVar(value=1)
        self.dt      = tk.IntVar(value=3600)
        self.fold    = tk.StringVar(value='CJ')
        self.bound   = tk.StringVar(value='UBWT')
        self.pipe    = tk.StringVar(value='utube')
        self.field   = tk.StringVar(value='rectangle')

        self.Ts      = tk.DoubleVar(value=60.0)
        self.dTcond  = tk.DoubleVar(value=2.0)
        self.dTevap  = tk.DoubleVar(value=5.0)
        self.eta     = tk.DoubleVar(value=0.50)

        self._cancel_evt: Optional[threading.Event] = None

        self._build_ui()
        try:
            self.protocol('WM_DELETE_WINDOW', self._on_close)
        except Exception:
            pass

    def _build_ui(self):
        pad = dict(padx=8, pady=6)
        # Quelle/Datei
        frm_file = ttk.LabelFrame(self, text='Profilquelle – q′[W/m], 8760/h')
        frm_file.pack(fill='x', padx=10, pady=8)
        self.use_synth = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_file, text='Synthetisches Heizprofil verwenden (CSV/Excel deaktivieren)',
                        variable=self.use_synth, command=self._toggle_source).grid(row=0, column=0, columnspan=2, sticky='w', **pad)
        self.e_path = ttk.Entry(frm_file, textvariable=self.path_profile, width=72)
        self.e_path.grid(row=1, column=0, **pad)
        self.b_path = ttk.Button(frm_file, text='…', command=self._pick)
        self.b_path.grid(row=1, column=1, **pad)

        # Parameter
        frm_p = ttk.LabelFrame(self, text='Parameter')
        frm_p.pack(fill='x', padx=10, pady=8)
        r=0
        ttk.Label(frm_p, text='YEARS:').grid(row=r, column=0, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.years, width=8).grid(row=r, column=1, sticky='w', **pad)
        ttk.Label(frm_p, text='dt [s]:').grid(row=r, column=2, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.dt, width=8).grid(row=r, column=3, sticky='w', **pad)
        ttk.Label(frm_p, text='FOLDING:').grid(row=r, column=4, sticky='e', **pad)
        ttk.Combobox(frm_p, textvariable=self.fold, values=['CJ','Liu','MLAA','hourly'], width=10).grid(row=r, column=5, sticky='w', **pad)
        r+=1
        ttk.Label(frm_p, text='BOUNDARY:').grid(row=r, column=0, sticky='e', **pad)
        ttk.Combobox(frm_p, textvariable=self.bound, values=['UBWT','UHTR'], width=10).grid(row=r, column=1, sticky='w', **pad)
        ttk.Label(frm_p, text='PIPE:').grid(row=r, column=2, sticky='e', **pad)
        ttk.Combobox(frm_p, textvariable=self.pipe, values=['utube','coaxial'], width=10).grid(row=r, column=3, sticky='w', **pad)
        ttk.Label(frm_p, text='FIELD:').grid(row=r, column=4, sticky='e', **pad)
        ttk.Combobox(frm_p, textvariable=self.field, values=['rectangle','ushaped','freeform'], width=12).grid(row=r, column=5, sticky='w', **pad)
        r+=1
        ttk.Label(frm_p, text='T_supply [°C]:').grid(row=r, column=0, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.Ts, width=8).grid(row=r, column=1, sticky='w', **pad)
        ttk.Label(frm_p, text='ΔT_cond [K]:').grid(row=r, column=2, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.dTcond, width=8).grid(row=r, column=3, sticky='w', **pad)
        ttk.Label(frm_p, text='ΔT_evap [K]:').grid(row=r, column=4, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.dTevap, width=8).grid(row=r, column=5, sticky='w', **pad)
        r+=1
        ttk.Label(frm_p, text='η (Carnot-Güte):').grid(row=r, column=0, sticky='e', **pad)
        ttk.Entry(frm_p, textvariable=self.eta, width=8).grid(row=r, column=1, sticky='w', **pad)
        r+=1
        # Erklärung
        ttk.Label(
            frm_p,
            text=(
                "COP-Formel (Heizen):  COP_Carnot = T_h/(T_h−T_c);  COP_real = η · COP_Carnot.\n"
                "T_h = (T_supply + ΔT_cond) + 273.15;  T_c = (EWT − ΔT_evap) + 273.15.\n"
                "EWT ≈ T_in aus dem Rohrmodell. ΔT_cond/ΔT_evap sind Wärmetauscher-Pinchs (typ. 2–5 K).\n"
                "Hinweis: COP wird nur für Heizstunden (Q_ground>0) berechnet; Rückspeisung wird ignoriert."
            ),
            foreground='#444', wraplength=900, justify='left'
        ).grid(row=r, column=0, columnspan=6, sticky='w', padx=8, pady=(8,4))

        # Fortschritt
        frm_prog = ttk.LabelFrame(self, text='Fortschritt')
        frm_prog.pack(fill='x', padx=10, pady=8)
        self.pvar = tk.IntVar(value=0)
        self.prog = ttk.Progressbar(frm_prog, orient='horizontal', mode='determinate', maximum=100, variable=self.pvar)
        self.prog.pack(fill='x', padx=8, pady=6)
        self.status = tk.StringVar(value='Bereit.')
        ttk.Label(frm_prog, textvariable=self.status).pack(anchor='w', padx=8, pady=(0,6))

        # Buttons
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill='x', padx=10, pady=8)
        ttk.Button(frm_btn, text='Start', command=self._start).pack(side='left', padx=8)
        ttk.Button(frm_btn, text='Abbrechen', command=self._cancel).pack(side='left')

    def _pick(self):
        p = filedialog.askopenfilename(title='Profil wählen',
                                       filetypes=[('CSV','*.csv'),('Excel','*.xlsx *.xlsm *.xls'),('Alle','*.*')])
        if p: self.path_profile.set(p)

    def _toggle_source(self):
        try:
            if self.use_synth.get():
                # CSV-Eingaben deaktivieren
                try:
                    self.e_path.state(('disabled',))
                    self.b_path.state(('disabled',))
                except Exception:
                    self.e_path.configure(state='disabled'); self.b_path.configure(state='disabled')
            else:
                try:
                    self.e_path.state(('!disabled',))
                    self.b_path.state(('!disabled',))
                except Exception:
                    self.e_path.configure(state='normal'); self.b_path.configure(state='normal')
        except Exception:
            pass

    def _start(self):
        try:
            years = max(1, int(self.years.get()))
            dt    = int(self.dt.get())
            # Profilquelle
            if self.use_synth.get():
                # Baue synthetisches Gebäudelast-Profil (W) und wandle in q′ [W/m] um
                from synthetic_heating_profile import build_heating_only_profile
                # Feld bestimmen, um L_total für Division zu erhalten
                _, field = _choose_field(self.field.get())
                try:
                    n_bh = len(field)
                except TypeError:
                    n_bh = len(list(field))
                H = float(bd.H); L_total = max(1.0, n_bh * H)
                Q, _info = build_heating_only_profile(bd.time, bd.dt, getattr(bd, 'Bedarf', 20000))
                if Q.size != 8760:
                    raise RuntimeError('Synthetisches Profil muss 8760 Stunden liefern.')
                q = (Q / L_total).astype(float)
            else:
                p = Path(self.path_profile.get()).resolve()
                if not p.exists():
                    messagebox.showerror('Fehler', 'Profildatei nicht gefunden.'); return
                q = _read_qprime_csv_or_xlsx(p)
            years = max(1, int(self.years.get()))
            dt    = int(self.dt.get())
            if dt != 3600:
                messagebox.showwarning('Hinweis', 'Dieses Modul erwartet dt=3600 s (hourly). Es wird fortgefahren.')
            fold  = self.fold.get(); bound = self.bound.get(); pipe = self.pipe.get(); field = self.field.get()
            Ts = float(self.Ts.get()); dTc = float(self.dTcond.get()); dTe = float(self.dTevap.get()); eta = float(self.eta.get())
        except Exception as e:
            messagebox.showerror('Eingabefehler', str(e)); return

        self._cancel_evt = threading.Event()
        def _prog(i,total,label):
            try:
                pct = int((i+1)/max(1,total)*100)
                self.pvar.set(pct); self.status.set(f"{label} – {pct}%"); self.update_idletasks()
            except Exception:
                pass

        def worker():
            try:
                pack = simulate_cop(q, years, field, fold, 'equivalent', bound, pipe, dt,
                                    Ts, dTc, dTe, eta, _prog, self._cancel_evt)
                if self._cancel_evt.is_set():
                    self.status.set('Abgebrochen.')
                    return
                # Plots im UI-Thread
                def _finish():
                    copC = pack['COP_carnot']; copR = pack['COP_real']
                    def stats(a: np.ndarray) -> Tuple[float,float,int]:
                        v = a[~np.isnan(a)]
                        if v.size==0: return float('nan'), float('nan'), 0
                        return float(np.nanmedian(v)), float(np.nanmean(v)), int(v.size)
                    medC, meanC, nC = stats(copC)
                    medR, meanR, nR = stats(copR)
                    # Plot 1: Carnot
                    plt.figure(figsize=(12,4.0))
                    plt.plot(copC, lw=0.8, label=f"Median={medC:.2f} | Mittel={meanC:.2f} | N={nC}")
                    plt.title(f"Carnot-COP – T_supply={Ts:.1f}°C, ΔT_cond={dTc:.1f}K, ΔT_evap={dTe:.1f}K")
                    plt.xlabel('Zeit [h]'); plt.ylabel('COP_Carnot [–]'); plt.grid(True, linestyle='--', alpha=0.35); plt.legend(); plt.tight_layout()
                    # Plot 2: Real
                    plt.figure(figsize=(12,4.0))
                    plt.plot(copR, lw=0.8, color='tab:orange', label=f"Median={medR:.2f} | Mittel={meanR:.2f} | N={nR}")
                    plt.title(f"Realer COP – η={eta:.2f}")
                    plt.xlabel('Zeit [h]'); plt.ylabel('COP_real [–]'); plt.grid(True, linestyle='--', alpha=0.35); plt.legend(); plt.tight_layout()
                    try:
                        plt.draw(); plt.show(block=False); plt.pause(0.1)
                    except Exception:
                        plt.show()
                    self.status.set('Fertig.')
                self.after(0, _finish)
            except Exception as e:
                self.status.set(f'Fehler: {e}')
                messagebox.showerror('Fehler', str(e))

        threading.Thread(target=worker, daemon=True).start()
        messagebox.showinfo('Start', 'Simulation gestartet. COP-Plots erscheinen in Matplotlib; Fortschritt unten.')

    def _cancel(self):
        if self._cancel_evt:
            self._cancel_evt.set(); self.status.set('Breche ab … (wirksam nach aktuellem Schritt)')

    def _on_close(self):
        try:
            self._cancel()
        except Exception:
            pass
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


def main():
    App().mainloop()


if __name__ == '__main__':
    main()
