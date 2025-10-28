#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v13 – Integrierte GUI + Rechner (vereint v10‑Logik mit einer robusten GUI)

Wesentliches:
- Genau EIN Anteil wird iteriert: Gebäudelast ODER Solar (Rückspeisung).
- Der andere Anteil bleibt original oder wird einmalig auf Ziel‑kWh/a skaliert.
- UBWT/UHTR, Rohrtyp (U‑Rohr/Coaxial), Zeitfaltung (CJ/Liu/MLAA/hourly).
- Fortschritt in GUI (Prozent) + Abbrechen‑Button. Während g‑Function keine
  echte Prozent möglich → Anzeige „busy“.
- Plots/Exporte wie v10 (Matplotlib/Excel/CSV), unverändert außer GUI‑Steuerung.

Nutzung:
- Direkt starten: python longterm_profile_scaler_v13.py
  → GUI öffnet sich. Einstellungen vornehmen, „Start“. Abbrechen jederzeit möglich.
"""

from __future__ import annotations

import csv
import sys
import os
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional
import json

import numpy as np
# Sicherstellen, dass kein Inline-Backend aktiv ist (z. B. via Spyder/MPLBACKEND)
try:
    if 'MPLBACKEND' in os.environ:
        mb = os.environ.get('MPLBACKEND','')
        if 'inline' in mb.lower() or mb.startswith('module://'):
            os.environ.pop('MPLBACKEND', None)
            os.environ['MPLBACKEND'] = 'TkAgg'
except Exception:
    pass
import matplotlib.pyplot as plt
# Interaktiver Modus, damit Fenster sofort erscheinen (auch aus Tk-Callbacks)
try:
    plt.ion()
except Exception:
    pass
import pygfunction as gt

import Data_Base as bd
import borefields as bf
import synthetic_heating_profile as shp

# ==========================
# Hilfsfunktionen (profil/zeit)
# ==========================
def month_indices_for_years(years: int, dt_s: int):
    hours_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)*24
    steps_per_hour = int(round(3600/dt_s))
    if abs(steps_per_hour - 3600/dt_s) > 1e-9:
        raise RuntimeError('Dieses Raster erwartet dt, das 1 h teilt (z. B. 3600 s).')
    steps_per_month = hours_per_month*steps_per_hour
    out=[]; start=0
    for y in range(years):
        for m, spm in enumerate(steps_per_month):
            out.append((y, m, start, start+int(spm)))
            start += int(spm)
    return out


def _read_qprime_csv(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV nicht gefunden: {path}')
    rows=[]
    with open(path,'r',encoding='utf-8-sig') as fh:
        first=fh.readline(); delim=';' if ';' in first and ',' not in first else ','; fh.seek(0)
        reader=csv.DictReader(fh, delimiter=delim)
        wkey=None
        for k in (reader.fieldnames or []):
            kk=(k or '').strip().lower()
            if kk in ('w_per_m','w_per_m[w/m]','w_per_m_w/m'): wkey=k; break
        if wkey is None:
            wkey='W_per_m'
            if wkey not in (reader.fieldnames or []):
                raise RuntimeError(f"CSV muss 'W_per_m' enthalten. Header: {reader.fieldnames}")
        for row in reader:
            val=row.get(wkey,'').strip()
            if val:
                try: rows.append(float(val))
                except Exception: continue
    qprime=np.array(rows,dtype=float)
    if qprime.size!=8760: raise RuntimeError(f'CSV hat {qprime.size} Zeilen, erwartet 8760.')
    return qprime


def _read_profile_meta_for_csv(csv_path: str) -> dict | None:
    """Liest die JSON-Metadaten zur gegebenen q′-CSV.
    Erwartet gleichnamige .json im selben Ordner (z. B. qprime_profile_from_excel.json).
    Rückgabe: dict oder None, wenn nicht gefunden/lesbar.
    """
    try:
        base = os.path.splitext(csv_path)[0]
        meta_path = base + ".json"
        if not os.path.exists(meta_path):
            # hart kodierte Fallbacks (Standardnamen)
            alt = None
            if os.path.basename(csv_path) == "qprime_profile_from_excel.csv":
                alt = os.path.join(os.path.dirname(csv_path), "qprime_profile_from_excel.json")
            elif os.path.basename(csv_path) == "qprime_recharge_only.csv":
                alt = os.path.join(os.path.dirname(csv_path), "qprime_recharge_only.json")
            if alt and os.path.exists(alt):
                meta_path = alt
            else:
                return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _energy_kwh(series_W: np.ndarray, dt_s: int) -> float:
    return float(np.sum(series_W)*dt_s/3.6e6)


def choose_field_by_name(name: str):
    name=(name or '').lower().strip()
    if name in ('ushaped','u_shaped','u-shaped','u'):
        return 'U_shaped_field', bf.U_shaped_field
    if name in ('freeform','freiform','polygon','auto'):
        # Versuche, Freiform-Punkte aus autoborehole/borefield_polygon_points.csv zu laden
        try:
            import csv, os
            from pygfunction.boreholes import Borehole
            H=float(bd.H); D=float(getattr(bd,'D',1.5)); r=float(getattr(bd,'r',0.075))
            path = os.path.join(os.getcwd(), 'autoborehole', 'borefield_polygon_points.csv')
            if not os.path.exists(path):
                # alternative: letzter Metaordner? (für Einfachheit: Fehlermeldung)
                raise FileNotFoundError('autoborehole/borefield_polygon_points.csv nicht gefunden.')
            pts=[]
            with open(path,'r',encoding='utf-8') as fh:
                rdr = csv.DictReader(fh)
                xk=None; yk=None
                for k in (rdr.fieldnames or []):
                    lk=(k or '').strip().lower()
                    if lk=='x_m': xk=k
                    if lk=='y_m': yk=k
                if not xk or not yk:
                    raise RuntimeError('CSV muss Spalten x_m,y_m enthalten.')
                for row in rdr:
                    try:
                        x=float((row.get(xk,'') or '0').replace(',','.'))
                        y=float((row.get(yk,'') or '0').replace(',','.'))
                        pts.append((x,y))
                    except Exception:
                        continue
            # Erzeuge Boreholes
            holes=[Borehole(H, D, r, x, y) for (x,y) in pts]
            return 'freeform_field', holes
        except Exception as e:
            # Fallback auf Rechteck
            return 'rectangle_field', bf.rectangle_field
    return 'rectangle_field', bf.rectangle_field


# ==========================
# Konfiguration
# ==========================
@dataclass
class Config:
    # Modell/Modi
    GF_METHOD: str = 'equivalent'           # 'equivalent' | 'similarities'
    FOLDING_MODE: str = 'CJ'                # 'CJ' | 'Liu' | 'MLAA' | 'hourly'
    BOUNDARY: str = 'UBWT'                  # 'UBWT' | 'UHTR'
    PIPE_TYPE: str = 'utube'                # 'utube' | 'coaxial'
    SHOW_PROGRESS: bool = True
    USE_EXTRA_SOLAR_PROFILE: bool = True

    # Geometrie/Zeiten
    FIELD_NAME: str = 'rectangle'
    YEARS: int = 50
    dt: int = 3600

    # Profile
    PROFILE_SOURCE: str = 'csv'
    CSV_PATH: str = 'qprime_profile_from_excel.csv'
    CSV_PATH_RECHARGE: str = 'qprime_recharge_only.csv'

    # Iteration (genau eine Seite iteriert)
    ITERATION_TARGET: str = 'last'          # 'last' | 'solar'
    LAST_SCALING_MODE: str = 'iterate'      # 'iterate' | 'fixed' | 'original'
    SOLAR_SCALING_MODE: str = 'original'    # 'iterate' | 'fixed' | 'original'
    START_ANNUAL_KWH_LAST: float = 450000.0
    FINE_STEP_PCT_LAST: float = 2.0
    MAX_ITERS_LAST: int = 5
    START_ANNUAL_KWH_SOLAR: float = 150000.0
    FINE_STEP_PCT_SOLAR: float = 2.0
    MAX_ITERS_SOLAR: int = 5

    # Grenzen Last (Min)
    USE_LAST_MIN_TFAVG: bool = True
    USE_LAST_MIN_EWT: bool = False
    USE_LAST_MIN_TB: bool = False
    USE_LAST_MONTH_MEAN_TB: bool = False
    # Neu: Monatsmittel-Grenzen (Last)
    USE_LAST_MONTH_MEAN_EWT: bool = False
    USE_LAST_MONTH_MEAN_TFAVG: bool = False
    LAST_MIN_TFAVG_C: float = 0.0
    LAST_MIN_EWT_C: float = -5.0
    LAST_MIN_TB_C: float = 0.0
    LAST_MONTH_MEAN_TB_C: float = 0.0
    # Neu: Default-Grenzen Monatsmittel (konservativ)
    LAST_MONTH_MEAN_EWT_C: float = 0.0
    LAST_MONTH_MEAN_TFAVG_C: float = 0.0

    # Grenzen Solar (Max)
    USE_SOLAR_MAX_TFAVG: bool = True
    USE_SOLAR_MAX_EWT: bool = False
    USE_SOLAR_MAX_TB: bool = False
    USE_SOLAR_MONTH_MEAN_TB: bool = False
    # Neu: Monatsmittel-Grenzen (Solar)
    USE_SOLAR_MONTH_MEAN_EWT: bool = False
    USE_SOLAR_MONTH_MEAN_TFAVG: bool = False
    SOLAR_MAX_TFAVG_C: float = 20.0
    SOLAR_MAX_EWT_C: float = 20.0
    SOLAR_MAX_TB_C: float = 25.0
    SOLAR_MONTH_MEAN_TB_C: float = 20.0
    # Neu: Default-Grenzen Monatsmittel (Solar)
    SOLAR_MONTH_MEAN_EWT_C: float = 20.0
    SOLAR_MONTH_MEAN_TFAVG_C: float = 20.0

    # Plots
    SHOW_PLOT_TEMP_SERIES: bool = True
    SHOW_PLOT_MONTHLY_MEANS: bool = True
    SHOW_PLOT_TARGET_PROFILE_PREV: bool = True
    SHOW_PLOT_TARGET_PROFILE_VIOL: bool = True
    SHOW_PLOT_GROUND_PROFILE: bool = True
    SHOW_PLOT_VIOLATION_DETAIL: bool = True

    # Neu: Drift-Kriterium und Plot
    USE_DRIFT_MIN_MONTH_MEAN_TB: bool = False   # Abbruch, wenn mittlere jährliche Abkühlrate (min Monatsmittel Tb) zu groß
    DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR: float = 0.10  # [K/Jahr] (zulässige maximale Abkühlrate)
    SHOW_PLOT_DRIFT_MIN_TB: bool = True
    # Neu: Drift-Kriterium – Jahresmittel T_b (für Solar-Ermittlung)
    USE_DRIFT_ANNUAL_MEAN_TB: bool = False
    DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR: float = 0.00

    # COP
    USE_COP: bool = False
    COP_VALUE: float = 3.0

    # Export
    EXPORT_EXCEL_LAST: bool = True
    EXPORT_EXCEL_SOLAR: bool = False
    EXCEL_FILENAME_LAST: str = 'heating_profile_W_per_m_prev_year_last.xlsx'
    EXCEL_FILENAME_SOLAR: str = 'solar_profile_W_per_m_prev_year.xlsx'


# ==========================
# Physik: Rohrmodelle
# ==========================
def build_pipe_model(config: Config, borehole, m_flow_bh: float):
    if config.PIPE_TYPE.lower()=='coaxial':
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


# ==========================
# Kernsimulation eines Jahresprofils über YEARS
# ==========================
def simulate(config: Config,
             Q_build_year_W: np.ndarray,
             qprime_solar_year: np.ndarray,
             progress: Optional[Callable[[int,int,str], None]] = None,
             cancel_evt: Optional[threading.Event] = None):
    # Feld
    field_name, field = choose_field_by_name(config.FIELD_NAME)
    try: n_bh = len(field)
    except TypeError: n_bh = len(list(field))
    H=float(bd.H); L_total=max(1.0, n_bh*H)
    dt=config.dt; n_steps_year=int(np.round(8760.0*3600.0/dt)); total_steps=config.YEARS*n_steps_year

    # COP korrekt: nur auf positive Gebäudelasten anwenden; negative (Einspeisung) bleiben unverändert
    Q_pos = np.clip(Q_build_year_W, 0.0, None)
    Q_neg = np.clip(Q_build_year_W, None, 0.0)
    if config.USE_COP:
        Q_pos_ground = Q_pos*(1.0 - 1.0/max(1e-9, config.COP_VALUE))
    else:
        Q_pos_ground = Q_pos
    Q_build_ground = Q_pos_ground + Q_neg

    # Serien wiederholen
    Q_build_series = np.tile(Q_build_ground, config.YEARS)
    qprime_solar_series = np.tile(qprime_solar_year, config.YEARS)
    qprime_eff_series = Q_build_series/L_total - qprime_solar_series

    # Rohrmodell
    m_flow_bh=max(1e-9, float(bd.flow)/max(1,n_bh))
    borehole0 = next(iter(field))
    Pipe, _ = build_pipe_model(config, borehole0, m_flow_bh)
    cp=float(bd.fluid_isobaric_heatcapacity)

    # Ausgabereihen
    T_b=np.zeros(total_steps); T_in=np.zeros(total_steps); T_out=np.zeros(total_steps); T_favg=np.zeros(total_steps)

    def _prog(i,label):
        if progress: progress(i,total_steps,label)

    if config.FOLDING_MODE.lower()=='hourly':
        time_full = dt*np.arange(1,total_steps+1)
        if progress: progress(0,1,'g-function hourly …')
        g_full = gt.gfunction.gFunction(field, alpha=bd.difusivity, time=time_full,
                                        method=config.GF_METHOD, boundary_condition=config.BOUNDARY, options=bd.options).gFunc
        if cancel_evt and cancel_evt.is_set():
            raise RuntimeError('Abgebrochen')
        dg=np.diff(g_full,prepend=0.0)
        inj_W = -qprime_eff_series*L_total
        conv = np.convolve(inj_W, dg, mode='full')[:total_steps]
        T_b = conv/(2.0*np.pi*bd.k_s)/(H*n_bh) + bd.T_g
        for i in range(total_steps):
            if cancel_evt and cancel_evt.is_set(): raise RuntimeError('Abgebrochen')
            Q_bh=(qprime_eff_series[i]*L_total)/max(1,n_bh)
            T_in[i]=Pipe.get_inlet_temperature(Q_bh, T_b[i], m_flow_bh, cp)
            T_out[i]=Pipe.get_outlet_temperature(T_in[i], T_b[i], m_flow_bh, cp)
            T_favg[i]=0.5*(T_in[i]+T_out[i])
            _prog(i,'hourly')
    else:
        # Aggregatoren
        if config.FOLDING_MODE.lower()=='liu':
            LoadAgg=gt.load_aggregation.Liu(dt, config.YEARS*8760*dt)
        elif config.FOLDING_MODE.lower()=='mlaa':
            LoadAgg=gt.load_aggregation.MLAA(dt, config.YEARS*8760*dt)
        else:
            LoadAgg=gt.load_aggregation.ClaessonJaved(dt, config.YEARS*8760*dt)
        time_req=LoadAgg.get_times_for_simulation()
        if progress: progress(0,1,f'g-function {config.FOLDING_MODE} …')
        gfunc=gt.gfunction.gFunction(field, alpha=bd.difusivity, time=time_req,
                                     boundary_condition=config.BOUNDARY, options=bd.options, method=config.GF_METHOD)
        if cancel_evt and cancel_evt.is_set(): raise RuntimeError('Abgebrochen')
        LoadAgg.initialize(gfunc.gFunc/(2.0*np.pi*bd.k_s))
        for i in range(total_steps):
            if cancel_evt and cancel_evt.is_set(): raise RuntimeError('Abgebrochen')
            LoadAgg.next_time_step(int((i+1)*dt))
            qprime=float(qprime_eff_series[i])
            LoadAgg.set_current_load(qprime)
            dT_b=LoadAgg.temporal_superposition(); dT_b=float(np.ravel(dT_b)[0])
            T_b[i]=bd.T_g - dT_b
            Q_bh=(qprime*L_total)/max(1,n_bh)
            T_in[i]=Pipe.get_inlet_temperature(Q_bh, T_b[i], m_flow_bh, cp)
            T_out[i]=Pipe.get_outlet_temperature(T_in[i], T_b[i], m_flow_bh, cp)
            T_favg[i]=0.5*(T_in[i]+T_out[i])
            _prog(i,config.FOLDING_MODE)

    return dict(T_b=T_b,T_in=T_in,T_out=T_out,T_f_avg=T_favg,
                qprime_eff=qprime_eff_series,
                Q_build_series=Q_build_series,
                Q_solar_series=qprime_solar_series*L_total,
                n_steps_year=n_steps_year)


# ==========================
# Iterationssteuerung und Plots
# ==========================
def run_iteration(config: Config,
                  progress: Optional[Callable[[int,int,str], None]] = None,
                  iter_status: Optional[Callable[[float,float,float,int], None]] = None,
                  cancel_evt: Optional[threading.Event] = None):
    # Feldgröße
    field_name, field = choose_field_by_name(config.FIELD_NAME)
    try: n_bh=len(field)
    except TypeError: n_bh=len(list(field))
    H=float(bd.H); L_total=max(1.0, n_bh*H); dt=config.dt

    # Profile (Basis)
    # Gebäudelast – unskaliert aus CSV ("original"). Für synthetic bleibt Skalierung auf Ziel sinnvoll.
    def build_build_original():
        steps=int(np.round(8760.0*3600.0/dt))
        if config.PROFILE_SOURCE.lower()=='synthetic':
            # Kein echtes "original" vorhanden → verwende Startwert als Basis
            Q, _ = shp.build_heating_only_profile(bd.time, bd.dt, config.START_ANNUAL_KWH_LAST)
            if Q.size!=steps: raise RuntimeError('Synthetisches Profil auf anderes dt einstellen')
            return Q
        qprime=_read_qprime_csv(config.CSV_PATH)
        # Energie-invariant: JSON-Metadaten auslesen → L_ref verwenden
        meta = _read_profile_meta_for_csv(config.CSV_PATH)
        if meta and isinstance(meta.get('field'), dict):
            n_ref = float(meta['field'].get('n_bh', n_bh))
            H_ref = float(meta['field'].get('H_m', H))
            L_ref = max(1.0, n_ref*H_ref)
        else:
            L_ref = L_total
        Q_hour=qprime*L_ref
        return np.repeat(Q_hour, int(round(3600/dt)))

    def build_solar_original():
        qprime_s = _read_qprime_csv(config.CSV_PATH_RECHARGE)
        # Energie-invariant skalieren: q′_solar_scaled = q′_csv * (L_ref / L_total)
        meta = _read_profile_meta_for_csv(config.CSV_PATH_RECHARGE)
        if meta and isinstance(meta.get('field'), dict):
            n_ref = float(meta['field'].get('n_bh', n_bh))
            H_ref = float(meta['field'].get('H_m', H))
            L_ref = max(1.0, n_ref*H_ref)
        else:
            L_ref = L_total
        scale = L_ref / L_total
        return np.repeat(qprime_s * scale, int(round(3600/dt)))

    qprime_solar_base = build_solar_original() if config.USE_EXTRA_SOLAR_PROFILE else np.zeros(int(8760*3600/dt))
    Q_build_base      = build_build_original()
    # Energies: Netto, Positiv-Anteil, Solar
    E_build_base      = _energy_kwh(Q_build_base, dt)
    E_build_pos_base  = _energy_kwh(np.clip(Q_build_base, 0.0, None), dt)
    # qprime_solar_base wurde bereits energieinvariant skaliert ⇒ Energie = Sum(q′*L_ref)
    E_solar_base      = _energy_kwh(qprime_solar_base*L_total, dt)

    def scale_build_to(E):
        # Skaliere so, dass die POSITIVE Gebäudelastenergie dem Ziel E entspricht
        if E_build_pos_base > 0:
            return Q_build_base * (float(E) / E_build_pos_base)
        # Fallback (keine positiven Anteile): skaliere auf Nettoenergie
        return Q_build_base if E_build_base <= 0 else Q_build_base * (float(E) / E_build_base)
    def scale_solar_to(E):
        return qprime_solar_base if E_solar_base<=0 else qprime_solar_base*(float(E)/E_solar_base)

    # Startserien je nach Modi
    if config.LAST_SCALING_MODE=='original':
        Q_build_year=Q_build_base.copy()
    elif config.LAST_SCALING_MODE=='fixed':
        Q_build_year=scale_build_to(config.START_ANNUAL_KWH_LAST)
    else:  # iterate initial
        Q_build_year=scale_build_to(config.START_ANNUAL_KWH_LAST)

    if not config.USE_EXTRA_SOLAR_PROFILE:
        qprime_solar_year=np.zeros_like(qprime_solar_base)
    elif config.SOLAR_SCALING_MODE=='original':
        qprime_solar_year=qprime_solar_base.copy()
    elif config.SOLAR_SCALING_MODE=='fixed':
        qprime_solar_year=scale_solar_to(config.START_ANNUAL_KWH_SOLAR)
    else:
        qprime_solar_year=scale_solar_to(config.START_ANNUAL_KWH_SOLAR)

    if config.ITERATION_TARGET=='last':
        E_iter=config.START_ANNUAL_KWH_LAST; step=1.0+config.FINE_STEP_PCT_LAST/100.0; maxit=config.MAX_ITERS_LAST
    else:
        E_iter=config.START_ANNUAL_KWH_SOLAR; step=1.0+config.FINE_STEP_PCT_SOLAR/100.0; maxit=config.MAX_ITERS_SOLAR

    prev=None; prevE=None; res=None
    viol_info=None  # wird gesetzt, wenn ein Grenzwert verletzt wurde
    last_energies=None  # für Plots/Export
    for it in range(1, maxit+1):
        if config.ITERATION_TARGET=='last':
            Q_build_year=scale_build_to(E_iter)
        else:
            qprime_solar_year=scale_solar_to(E_iter)

        # Iterations-Energien (kWh/a) für GUI – immer anzeigen
        # Anzeige-Bilanz nach Nutzerwunsch:
        # Gebäude = Summe der positiven Gebäudelasten (ohne COP)
        # W_el     = Gebäude / COP (nur wenn COP aktiv)
        # Solar    = externe Solar-CSV + interne Einspeisung aus negativen Gebäudelasten
        # Boden    = Gebäude − W_el − Solar
        Q_pos = np.clip(Q_build_year, 0.0, None)
        Q_neg = np.clip(Q_build_year, None, 0.0)
        E_build_pos_kWh = _energy_kwh(Q_pos, dt)                          # Gebäudelast (nur positive Anteile)
        E_wel_kWh = (E_build_pos_kWh/max(1e-9, config.COP_VALUE)) if config.USE_COP else 0.0
        # Anzeigegrößen
        E_solar_ext_kWh = _energy_kwh(qprime_solar_year*L_total, dt)      # Solar (Rückspeisung, extern)
        E_solar_int_kWh = _energy_kwh(-Q_neg, dt)                          # interne Rückspeisung (aus neg. Gebäudelast)
        E_solar_kWh = E_solar_ext_kWh + E_solar_int_kWh
        E_net_kWh   = E_build_pos_kWh - E_wel_kWh - E_solar_kWh            # Effektive Bodenlast (Jahresbilanz)
        # Merke für Plots/Export
        last_energies = dict(
            E_build_pos_kWh=float(E_build_pos_kWh),
            E_wel_kWh=float(E_wel_kWh),
            E_solar_ext_kWh=float(E_solar_ext_kWh),
            E_solar_int_kWh=float(E_solar_int_kWh),
            E_solar_tot_kWh=float(E_solar_kWh),
            E_net_kWh=float(E_net_kWh),
            E_profile_kWh=float(E_build_pos_kWh - E_solar_int_kWh),
        )
        if iter_status:
            try:
                iter_status(E_build_pos_kWh/1000.0, E_solar_kWh/1000.0, E_net_kWh/1000.0, E_wel_kWh/1000.0, it)
            except Exception:
                pass

        # Simulation eines Laufs
        if progress: progress(0,1,f'Iteration {it:02d} …')
        res=simulate(config, Q_build_year, qprime_solar_year, progress, cancel_evt)
        if cancel_evt and cancel_evt.is_set():
            raise RuntimeError('Abgebrochen')

        # Grenzen prüfen
        violated=False
        # Merke die zuerst verletzte Bedingung für Report
        def _set_violation(side: str, kind: str, value: float, limit: float, extra: Optional[str]=None):
            nonlocal viol_info
            if viol_info is None:
                viol_info=dict(side=side, kind=kind, value=float(value), limit=float(limit), extra=extra or '', iter=it)
        if config.ITERATION_TARGET=='last':
            if config.USE_LAST_MIN_TFAVG:
                v=float(np.nanmin(res['T_f_avg']))
                if v < config.LAST_MIN_TFAVG_C:
                    violated=True; _set_violation('last','min_Tfavg',v,config.LAST_MIN_TFAVG_C)
            if config.USE_LAST_MIN_EWT:
                v=float(np.nanmin(res['T_in']))
                if v < config.LAST_MIN_EWT_C:
                    violated=True; _set_violation('last','min_EWT',v,config.LAST_MIN_EWT_C)
            if config.USE_LAST_MIN_TB:
                v=float(np.nanmin(res['T_b']))
                if v < config.LAST_MIN_TB_C:
                    violated=True; _set_violation('last','min_Tb',v,config.LAST_MIN_TB_C)
            if config.USE_LAST_MONTH_MEAN_TB:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_b'][s:e]))
                        if mm < config.LAST_MONTH_MEAN_TB_C:
                            violated=True; _set_violation('last','min_month_mean_Tb',mm,config.LAST_MONTH_MEAN_TB_C,f'steps={s}:{e}')
                            break
            if not violated and config.USE_LAST_MONTH_MEAN_EWT:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_in'][s:e]))
                        if mm < config.LAST_MONTH_MEAN_EWT_C:
                            violated=True; _set_violation('last','min_month_mean_EWT',mm,config.LAST_MONTH_MEAN_EWT_C,f'steps={s}:{e}')
                            break
            if not violated and config.USE_LAST_MONTH_MEAN_TFAVG:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_f_avg'][s:e]))
                        if mm < config.LAST_MONTH_MEAN_TFAVG_C:
                            violated=True; _set_violation('last','min_month_mean_Tfavg',mm,config.LAST_MONTH_MEAN_TFAVG_C,f'steps={s}:{e}')
                            break
        else:
            if config.USE_SOLAR_MAX_TFAVG:
                v=float(np.nanmax(res['T_f_avg']))
                if v > config.SOLAR_MAX_TFAVG_C:
                    violated=True; _set_violation('solar','max_Tfavg',v,config.SOLAR_MAX_TFAVG_C)
            if config.USE_SOLAR_MAX_EWT:
                v=float(np.nanmax(res['T_in']))
                if v > config.SOLAR_MAX_EWT_C:
                    violated=True; _set_violation('solar','max_EWT',v,config.SOLAR_MAX_EWT_C)
            if config.USE_SOLAR_MAX_TB:
                v=float(np.nanmax(res['T_b']))
                if v > config.SOLAR_MAX_TB_C:
                    violated=True; _set_violation('solar','max_Tb',v,config.SOLAR_MAX_TB_C)
            if config.USE_SOLAR_MONTH_MEAN_TB:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_b'][s:e]))
                        if mm > config.SOLAR_MONTH_MEAN_TB_C:
                            violated=True; _set_violation('solar','max_month_mean_Tb',mm,config.SOLAR_MONTH_MEAN_TB_C,f'steps={s}:{e}')
                            break
            if not violated and config.USE_SOLAR_MONTH_MEAN_EWT:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_in'][s:e]))
                        if mm > config.SOLAR_MONTH_MEAN_EWT_C:
                            violated=True; _set_violation('solar','max_month_mean_EWT',mm,config.SOLAR_MONTH_MEAN_EWT_C,f'steps={s}:{e}')
                            break
            if not violated and config.USE_SOLAR_MONTH_MEAN_TFAVG:
                for (_,_,s,e) in month_indices_for_years(config.YEARS, dt):
                    if s<e:
                        mm=float(np.mean(res['T_f_avg'][s:e]))
                        if mm > config.SOLAR_MONTH_MEAN_TFAVG_C:
                            violated=True; _set_violation('solar','max_month_mean_Tfavg',mm,config.SOLAR_MONTH_MEAN_TFAVG_C,f'steps={s}:{e}')
                            break

        # Unabhängig vom Ziel: Drift-Kriterium (min Monatsmittel T_b)
        if not violated and config.USE_DRIFT_MIN_MONTH_MEAN_TB:
            idx = month_indices_for_years(config.YEARS, dt)
            # pro Jahr: Minimum der Monatsmittel Tb
            per_year_min = []
            for y in range(config.YEARS):
                months = [k for k,(yy,_,_,_) in enumerate(idx) if yy==y]
                vals = []
                for k in months:
                    _,_,s,e = idx[k]
                    if e> s:
                        vals.append(float(np.mean(res['T_b'][s:e])))
                per_year_min.append(float(np.nanmin(vals)) if vals else float('nan'))
            drift = None
            if config.YEARS >= 2 and np.isfinite(per_year_min[0]) and np.isfinite(per_year_min[-1]):
                drift = (per_year_min[-1] - per_year_min[0]) / float(config.YEARS - 1)
            else:
                # Schnellcheck (1 Jahr): konservative Schätzung aus Halbjahres-Minima
                months_first = [k for k,(yy,mm,_,_) in enumerate(idx) if yy==0 and mm < 6]
                months_last  = [k for k,(yy,mm,_,_) in enumerate(idx) if yy==0 and mm >= 6]
                def min_mm(midx):
                    arr=[]
                    for k in midx:
                        _,_,s,e = idx[k]
                        if e> s:
                            arr.append(float(np.mean(res['T_b'][s:e])))
                    return float(np.nanmin(arr)) if arr else float('nan')
                a = min_mm(months_first); b = min_mm(months_last)
                if np.isfinite(a) and np.isfinite(b):
                    drift = (b - a)  # über ~1 Jahr
            if drift is not None and np.isfinite(drift):
                if drift < -float(config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR):
                    violated=True; _set_violation('last','drift_min_month_mean_Tb', drift, -float(config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR), extra=f"per_year_min={per_year_min}")

        # Unabhängig vom Ziel: Drift-Kriterium (Jahresmittel T_b)
        if not violated and config.USE_DRIFT_ANNUAL_MEAN_TB:
            # Jahresmittel je Jahr
            idx = month_indices_for_years(config.YEARS, dt)
            per_year_mean = []
            for y in range(config.YEARS):
                starts_ends = [(s,e) for (yy,_,s,e) in idx if yy==y]
                if not starts_ends:
                    per_year_mean.append(float('nan')); continue
                s0 = starts_ends[0][0]; eN = starts_ends[-1][1]
                per_year_mean.append(float(np.mean(res['T_b'][s0:eN])) if eN>s0 else float('nan'))
            drift = None
            if config.YEARS >= 2 and np.isfinite(per_year_mean[0]) and np.isfinite(per_year_mean[-1]):
                drift = (per_year_mean[-1] - per_year_mean[0]) / float(config.YEARS - 1)
            if drift is not None and np.isfinite(drift):
                if drift < -float(config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR):
                    violated=True; _set_violation('last','drift_annual_mean_Tb', drift, -float(config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR), extra=f"per_year_mean={per_year_mean}")

        if violated:
            break
        prev=res; prevE=E_iter; E_iter*=step

    return dict(res=res, prev=prev, prevE=prevE, field_name=field_name, n_bh=n_bh, H=H, L_total=L_total, dt=dt, violation=viol_info, energies=last_energies)


def plots_and_exports(config: Config, pack):
    res = pack['res']; prev=pack['prev']; L_total=pack['L_total']; dt=pack['dt']
    hours=(dt*np.arange(1,res['T_in'].size+1))/3600.0

    # Energies (für Infoboxen), aus run_iteration mitgegeben
    energies = pack.get('energies') or {}
    def _get_kwh(key: str) -> float:
        try:
            return float(energies.get(key, 0.0))
        except Exception:
            return 0.0
    E_build_mwh = _get_kwh('E_build_pos_kWh')/1000.0
    E_wel_mwh   = _get_kwh('E_wel_kWh')/1000.0
    E_solar_mwh = _get_kwh('E_solar_tot_kWh')/1000.0
    E_net_mwh   = _get_kwh('E_net_kWh')/1000.0
    E_prof_mwh  = _get_kwh('E_profile_kWh')/1000.0

    def _fmt(n: float) -> str:
        return f"{float(n):,.3f}".replace(',', 'X').replace('.', ',').replace('X','.')

    def _info_text() -> str:
        return "\n".join([
            f"Jahreslast Profil: {_fmt(E_prof_mwh)} MWh/a",
            f"Last Gebäude: {_fmt(E_build_mwh)} MWh/a",
            f"Solar (gesamt): {_fmt(E_solar_mwh)} MWh/a",
            f"W_el (COP={'aus' if not config.USE_COP else f'{config.COP_VALUE:.2f}'}): {_fmt(E_wel_mwh)} MWh/a",
            f"Boden (effektiv): {_fmt(E_net_mwh)} MWh/a",
        ])

    # Suffix für Titel, falls COP aktiv ist
    cop_suf = (f" | COP={config.COP_VALUE:.2f} (Faktor={1.0-1.0/max(1e-9,config.COP_VALUE):.3f})" if config.USE_COP else "")

    if config.SHOW_PLOT_TEMP_SERIES:
        plt.figure(figsize=(12,3.8))
        plt.plot(hours, res['T_in'], lw=0.9, label='T_in')
        plt.plot(hours, res['T_f_avg'], lw=0.9, label='T_f,avg')
        plt.plot(hours, res['T_b'], lw=0.9, label='T_b')
        plt.title('Temperaturen – Gesamtzeit'+cop_suf); plt.xlabel('Zeit [h]'); plt.ylabel('Temperatur [°C]')
        plt.grid(True, linestyle='--', alpha=0.35); 
        leg = plt.legend()
        if config.USE_COP:
            try: leg.set_title(f"COP={config.COP_VALUE:.2f}")
            except Exception: pass
        plt.tight_layout()
        # Info-Box
        ax=plt.gca();
        ax.annotate(_info_text(), xy=(0.98,0.98), xycoords='axes fraction', ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8), fontsize=9, zorder=10)

    if config.SHOW_PLOT_MONTHLY_MEANS:
        idx=month_indices_for_years(config.YEARS, dt)
        def mm(a): return np.array([float(np.mean(a[s:e])) if e>s else np.nan for (_,_,s,e) in idx])
        x=np.arange(len(idx)); mm_tin=mm(res['T_in']); mm_tb=mm(res['T_b']); mm_tfavg=mm(res['T_f_avg'])
        plt.figure(figsize=(12,3.8))
        plt.plot(x, mm_tin, '-', lw=1.2, label='Monatsmittel T_in'); plt.scatter(x, mm_tin, s=8)
        plt.plot(x, mm_tfavg, '-', lw=1.2, label='Monatsmittel T_f,avg'); plt.scatter(x, mm_tfavg, s=8)
        plt.plot(x, mm_tb,  '-', lw=1.2, label='Monatsmittel T_b');  plt.scatter(x, mm_tb,  s=8)
        plt.title('Monatsmittel'+cop_suf); plt.xlabel('Monat (kumuliert)'); plt.ylabel('Temperatur [°C]')
        plt.grid(True, linestyle='--', alpha=0.35); 
        leg = plt.legend()
        if config.USE_COP:
            try: leg.set_title(f"COP={config.COP_VALUE:.2f}")
            except Exception: pass
        plt.tight_layout()
        ax=plt.gca(); ax.annotate(_info_text(), xy=(0.98,0.98), xycoords='axes fraction',
                                  ha='right', va='top', bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8),
                                  fontsize=9, zorder=10)
        ax=plt.gca(); ax.annotate(_info_text(), xy=(0.98,0.98), xycoords='axes fraction',
                                  ha='right', va='top', bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8),
                                  fontsize=9, zorder=10)

    # Drift-Plot: Jahresmittel und min Monatsmittel T_b pro Jahr
    if config.SHOW_PLOT_DRIFT_MIN_TB:
        idx=month_indices_for_years(config.YEARS, dt)
        per_year_min=[]; per_year_mean=[]
        for y in range(config.YEARS):
            months=[k for k,(yy,_,_,_) in enumerate(idx) if yy==y]
            vals=[]
            for k in months:
                _,_,s,e=idx[k]
                if e>s:
                    vals.append(float(np.mean(res['T_b'][s:e])))
            per_year_min.append(float(np.nanmin(vals)) if vals else np.nan)
            if months:
                s0=idx[months[0]][2]; eN=idx[months[-1]][3]
                per_year_mean.append(float(np.mean(res['T_b'][s0:eN])) if eN>s0 else np.nan)
            else:
                per_year_mean.append(np.nan)
        years=np.arange(1, config.YEARS+1)
        plt.figure(figsize=(10,3.6))
        # Plot Jahresmittel (vordergründig)
        plt.plot(years, per_year_mean, '-o', lw=1.6, color='tab:blue', label='Jahresmittel T_b')
        # Plot min Monatsmittel (hinterlegt)
        plt.plot(years, per_year_min, '-o', lw=1.0, alpha=0.45, color='tab:gray', label='min Monatsmittel T_b')
        # Allowed-Linien
        title='Drift – '
        if config.USE_DRIFT_ANNUAL_MEAN_TB and config.YEARS>=2 and np.isfinite(per_year_mean[0]) and np.isfinite(per_year_mean[-1]):
            drift_mean=(per_year_mean[-1]-per_year_mean[0])/float(config.YEARS-1)
            base_mean=per_year_mean[0]
            allowed_mean=base_mean + (-abs(config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR))*(years-1)
            plt.plot(years, allowed_mean, '--', color='red', label=f'zul. Jahresmittel-Drift −{config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR:.3f} K/Jahr')
            title += f'Jahresmittel {drift_mean:.4f} K/Jahr'
        if config.USE_DRIFT_MIN_MONTH_MEAN_TB and config.YEARS>=2 and np.isfinite(per_year_min[0]) and np.isfinite(per_year_min[-1]):
            drift_min=(per_year_min[-1]-per_year_min[0])/float(config.YEARS-1)
            base_min=per_year_min[0]
            allowed_min=base_min + (-abs(config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR))*(years-1)
            plt.plot(years, allowed_min, ':', color='orange', label=f'zul. min-Monatsmittel-Drift −{config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR:.3f} K/Jahr')
            if 'Jahresmittel' not in title:
                title += f'min-Monatsmittel {drift_min:.4f} K/Jahr'
            else:
                title += f' | min-Monatsmittel {drift_min:.4f} K/Jahr'
        if title=='Drift – ':
            title='Drift (Jahres- und min-Monatsmittel)'
        plt.title(title)
        plt.xlabel('Jahr'); plt.ylabel('Temperatur [°C]'); plt.grid(True, linestyle='--', alpha=0.35); plt.legend(); plt.tight_layout()

    def plot_series(series_W_year, title):
        n=res['n_steps_year']; t=np.arange(n)*(dt/3600.0)
        plt.figure(figsize=(12,4.0))
        lbl = f"E≈{_energy_kwh(series_W_year[:n],dt):,.0f} kWh/a".replace(',', 'X').replace('.', ',').replace('X','.')
        if config.USE_COP:
            lbl += f" | COP={config.COP_VALUE:.2f}"
        plt.plot(t, series_W_year[:n]/1000.0, lw=1.0, label=lbl)
        plt.axhline(0.0, linestyle='--', alpha=0.5); plt.title(title)
        plt.xlabel('Zeit im Jahr [h]'); plt.ylabel('Leistung [kW]'); plt.grid(True, linestyle='--', alpha=0.35)
        leg = plt.legend()
        if config.USE_COP:
            try: leg.set_title(f"COP={config.COP_VALUE:.2f}")
            except Exception: pass
        plt.tight_layout()
        ax = plt.gca(); ax.annotate(_info_text(), xy=(0.98,0.98), xycoords='axes fraction',
                                    ha='right', va='top', bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8),
                                    fontsize=9, zorder=10)

    # Effektives Profil an Boden (Vorjahr/Verletzungsjahr)
    if config.SHOW_PLOT_TARGET_PROFILE_PREV and prev is not None:
        Qg_prev = prev['Q_build_series'][:res['n_steps_year']]  # bereits Boden-komponente (COP korrekt, intern + extern in Simulation behandelt)
        Qeff_prev = Qg_prev - prev['Q_solar_series'][:res['n_steps_year']]
        title = 'Profil an Boden (effektiv) – Vorjahr' + cop_suf
        plot_series(Qeff_prev, title)
    if config.SHOW_PLOT_TARGET_PROFILE_VIOL:
        Qg_now = res['Q_build_series'][:res['n_steps_year']]
        Qeff_now = Qg_now - res['Q_solar_series'][:res['n_steps_year']]
        title = 'Profil an Boden (effektiv) – Verletzungsjahr' + cop_suf
        plot_series(Qeff_now, title)

    if config.SHOW_PLOT_GROUND_PROFILE:
        # Vorjahr
        if prev is not None:
            Qg_prev = prev['Q_build_series'][:res['n_steps_year']]
            Qeff_prev = Qg_prev - prev['Q_solar_series'][:res['n_steps_year']]
            title = 'Bodenlast – Vorjahr (effektiv)' + cop_suf
            plot_series(Qeff_prev, title)
        # Verletzungsjahr
        Qg_now = res['Q_build_series'][:res['n_steps_year']]
        Qeff_now = Qg_now - res['Q_solar_series'][:res['n_steps_year']]
        title = 'Bodenlast – Verletzungsjahr (effektiv)' + cop_suf
        plot_series(Qeff_now, title)

    # Helper: aktives Kriterium aus Config ableiten (für den Fall ohne Verletzung)
    def _choose_active_kind_and_limit() -> tuple[str, float, str]:
        # returns (kind, limit, side)
        if config.ITERATION_TARGET=='last':
            if config.USE_LAST_MIN_TFAVG: return 'min_Tfavg', config.LAST_MIN_TFAVG_C, 'last'
            if config.USE_LAST_MIN_EWT:   return 'min_EWT',   config.LAST_MIN_EWT_C,   'last'
            if config.USE_LAST_MIN_TB:    return 'min_Tb',    config.LAST_MIN_TB_C,    'last'
            if config.USE_LAST_MONTH_MEAN_TFAVG: return 'min_month_mean_Tfavg', config.LAST_MONTH_MEAN_TFAVG_C, 'last'
            if config.USE_LAST_MONTH_MEAN_EWT:   return 'min_month_mean_EWT',   config.LAST_MONTH_MEAN_EWT_C,   'last'
            if config.USE_LAST_MONTH_MEAN_TB:    return 'min_month_mean_Tb',    config.LAST_MONTH_MEAN_TB_C,    'last'
            return 'min_Tfavg', float('nan'), 'last'
        else:
            if config.USE_SOLAR_MAX_TFAVG: return 'max_Tfavg', config.SOLAR_MAX_TFAVG_C, 'solar'
            if config.USE_SOLAR_MAX_EWT:   return 'max_EWT',   config.SOLAR_MAX_EWT_C,   'solar'
            if config.USE_SOLAR_MAX_TB:    return 'max_Tb',    config.SOLAR_MAX_TB_C,    'solar'
            if config.USE_SOLAR_MONTH_MEAN_TFAVG: return 'max_month_mean_Tfavg', config.SOLAR_MONTH_MEAN_TFAVG_C, 'solar'
            if config.USE_SOLAR_MONTH_MEAN_EWT:   return 'max_month_mean_EWT',   config.SOLAR_MONTH_MEAN_EWT_C,   'solar'
            if config.USE_SOLAR_MONTH_MEAN_TB:    return 'max_month_mean_Tb',    config.SOLAR_MONTH_MEAN_TB_C,    'solar'
            return 'max_Tfavg', float('nan'), 'solar'

    # Detailplot des (verletzten ODER aktiven) Kriteriums – Monats-/Zeitfenster
    make_detail = config.SHOW_PLOT_VIOLATION_DETAIL and (pack.get('violation') is not None or True)
    if make_detail:
        # benutze echte Verletzung, sonst leiten wir Kriterium/Limit aus Config ab und markieren das Extrem
        if pack.get('violation') is not None:
            viol = pack['violation']
            kind = viol.get('kind','')
            side = viol.get('side','')
            limit = float(viol.get('limit', np.nan))
            value = float(viol.get('value', np.nan))
            violated = True
            # Spezialfall: Drift-Kriterium – eigener Plot
            if kind.startswith('drift_min_month_mean_Tb') or kind.startswith('drift_annual_mean_Tb'):
                idx=month_indices_for_years(config.YEARS, dt)
                per_year_min=[]; per_year_mean=[]
                for y in range(config.YEARS):
                    months=[k for k,(yy,_,_,_) in enumerate(idx) if yy==y]
                    vals=[]
                    for k in months:
                        _,_,s,e = idx[k]
                        if e> s:
                            vals.append(float(np.mean(res['T_b'][s:e])))
                    per_year_min.append(float(np.nanmin(vals)) if vals else np.nan)
                    if months:
                        s0=idx[months[0]][2]; eN=idx[months[-1]][3]
                        per_year_mean.append(float(np.mean(res['T_b'][s0:eN])) if eN>s0 else np.nan)
                    else:
                        per_year_mean.append(np.nan)
                years=np.arange(1, config.YEARS+1)
                plt.figure(figsize=(10,3.6))
                plt.plot(years, per_year_mean, '-o', lw=1.6, color='tab:blue', label='Jahresmittel T_b')
                plt.plot(years, per_year_min,  '-o', lw=1.0, alpha=0.45, color='tab:gray', label='min Monatsmittel T_b')
                title='Drift '
                if kind.startswith('drift_annual_mean_Tb') and config.YEARS>=2 and np.isfinite(per_year_mean[0]) and np.isfinite(per_year_mean[-1]):
                    drift=(per_year_mean[-1]-per_year_mean[0]) / float(config.YEARS-1)
                    base=per_year_mean[0]
                    allowed=base + (-abs(config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR))*(years-1)
                    plt.plot(years, allowed, '--', color='red', label=f'zulässige Drift −{config.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR:.3f} K/Jahr')
                    title += f'(Jahresmittel): {drift:.4f} K/Jahr | Limit {limit:.4f} K/Jahr'
                elif kind.startswith('drift_min_month_mean_Tb') and config.YEARS>=2 and np.isfinite(per_year_min[0]) and np.isfinite(per_year_min[-1]):
                    drift=(per_year_min[-1]-per_year_min[0]) / float(config.YEARS-1)
                    base=per_year_min[0]
                    allowed=base + (-abs(config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR))*(years-1)
                    plt.plot(years, allowed, '--', color='red', label=f'zulässige Drift −{config.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR:.3f} K/Jahr')
                    title += f'(min Monatsmittel): {drift:.4f} K/Jahr | Limit {limit:.4f} K/Jahr'
                plt.title(title)
                plt.xlabel('Jahr'); plt.ylabel('Temperatur [°C]'); plt.grid(True, linestyle='--', alpha=0.35); plt.legend(); plt.tight_layout()
                return
        else:
            kind, limit, side = _choose_active_kind_and_limit()
            violated = False
            # Extremwert bestimmen
            data_map = {
                'min_Tfavg': res['T_f_avg'], 'max_Tfavg': res['T_f_avg'],
                'min_EWT': res['T_in'],      'max_EWT': res['T_in'],
                'min_Tb': res['T_b'],        'max_Tb': res['T_b'],
                'min_month_mean_Tb': res['T_b'], 'max_month_mean_Tb': res['T_b'],
                'min_month_mean_EWT': res['T_in'], 'max_month_mean_EWT': res['T_in'],
                'min_month_mean_Tfavg': res['T_f_avg'], 'max_month_mean_Tfavg': res['T_f_avg'],
            }
            a = data_map.get(kind, res['T_b'])
            if 'min_' in kind:
                value = float(np.nanmin(a))
            elif 'max_' in kind:
                value = float(np.nanmax(a))
            else:
                value = float(np.nanmin(a))

        idx=month_indices_for_years(config.YEARS, dt)
        idx = month_indices_for_years(config.YEARS, dt)
        
        # Hilfsfunktionen
        def _mm(a):
            return np.array([float(np.mean(a[s:e])) if e>s else np.nan for (_,_,s,e) in idx])

        # Energien stammen aus pack['energies']; Info-Box nutzt _info_text()

        # Zu plottende Serie bestimmen
        series_map = {
            'min_Tfavg': res['T_f_avg'], 'max_Tfavg': res['T_f_avg'],
            'min_EWT': res['T_in'],      'max_EWT': res['T_in'],
            'min_Tb': res['T_b'],        'max_Tb': res['T_b'],
            'min_month_mean_Tb': res['T_b'], 'max_month_mean_Tb': res['T_b'],
            'min_month_mean_EWT': res['T_in'], 'max_month_mean_EWT': res['T_in'],
            'min_month_mean_Tfavg': res['T_f_avg'], 'max_month_mean_Tfavg': res['T_f_avg'],
        }
        a = series_map.get(kind, res['T_b'])

        # Monat ermitteln
        month_label = 'n/a'
        s=e=None
        if pack.get('violation') is not None and 'month_mean' in kind and isinstance(viol.get('extra'), str) and 'steps=' in viol['extra']:
            try:
                se = viol['extra'].split('steps=')[1]
                s,e = [int(x) for x in se.split(':')]
            except Exception:
                s=e=None
        if s is None or e is None:
            # Finde Index der Extremstelle – bei Monatsmittel-Kriterien auf Monatsmittel arbeiten
            if 'month_mean' in kind:
                mm_vals_all = _mm(a)
                i_mm = int(np.nanargmin(mm_vals_all) if 'min_' in kind else np.nanargmax(mm_vals_all))
                yy, mm_i, ss, ee = idx[i_mm]
                s, e = ss, ee
                month_label = f'J{yy+1}-M{mm_i+1}'
                # Setze den Extremwert passend zum Monatsmittel
                value = float(mm_vals_all[i_mm])
            else:
                i = int(np.nanargmin(a) if 'min_' in kind else np.nanargmax(a))
                for (yy,mm,ss,ee) in idx:
                    if ss <= i < ee:
                        s,e = ss,ee; month_label=f'J{yy+1}-M{mm+1}'; break
        else:
            # Finde Label aus s,e
            for (yy,mm,ss,ee) in idx:
                if ss==s and ee==e:
                    month_label=f'J{yy+1}-M{mm+1}'; break

        if 'month_mean' in kind:
            # Monatsmittelplot
            mm_vals = _mm(a)
            # Wenn wir s,e erst jetzt ermittelt haben (oder aus active criterion), stelle sicher, dass 'value' die Monatsmittel darstellt
            try:
                if s is not None and e is not None:
                    # finde den Monatsindex m_id
                    m_id = [k for k,(_,_,ss,ee) in enumerate(idx) if ss==s and ee==e]
                    if m_id:
                        value = float(mm_vals[m_id[0]])
            except Exception:
                pass
            x = np.arange(len(mm_vals))
            plt.figure(figsize=(12,3.8))
            plt.plot(x, mm_vals, '-', lw=1.2, label='Monatsmittel')
            plt.scatter(x, mm_vals, s=10)
            if s is not None and e is not None:
                # markiere den verletzten Monat
                m_id = [k for k,(_,_,ss,ee) in enumerate(idx) if ss==s and ee==e]
                if m_id:
                    mi=m_id[0]
                    plt.scatter([mi],[mm_vals[mi]], color='red', s=35, zorder=3, label='Extremwert')
            if np.isfinite(limit):
                plt.axhline(limit, color='red', linestyle='--', label=f'Grenze {limit:.2f} °C')
            title = (f"Grenzverletzung ({'Last' if side=='last' else 'Solar'}): {kind} – {month_label}"
                     if violated else
                     f"Kein Verstoß – aktiv: {kind} – {month_label}") + cop_suf
            plt.title(title)
            plt.xlabel('Monat (kumuliert)'); plt.ylabel('Temperatur [°C]')
            plt.grid(True, linestyle='--', alpha=0.35)
            plt.legend()
            ax=plt.gca();
            ax.annotate(_info_text()+"\n" + (
                f"Verletzung: {value:.3f} °C vs. Grenze {limit:.3f} °C" if violated and np.isfinite(limit)
                else (f"Kein Verstoß, Extremwert: {value:.3f} °C" + (f" – Grenze {limit:.3f} °C" if np.isfinite(limit) else ""))
            ), xy=(0.98,0.98), xycoords='axes fraction', ha='right', va='top',
            bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8), fontsize=9, zorder=10)
            plt.tight_layout()
        else:
            # Stundenauflösung im Verletzungsmonat
            s = s or 0; e = e or res['n_steps_year']
            t = (np.arange(e-s)*(dt/3600.0))
            plt.figure(figsize=(12,3.8))
            plt.plot(t, a[s:e], lw=1.0, label='Temperatur')
            if np.isfinite(limit):
                plt.axhline(limit, color='red', linestyle='--', label=f'Grenze {limit:.2f} °C')
            plt.title((f"Grenzverletzung ({'Last' if side=='last' else 'Solar'}): {kind} – {month_label}"
                      if violated else
                      f"Kein Verstoß – aktiv: {kind} – {month_label}") + cop_suf)
            plt.xlabel('Zeit im Monat [h]'); plt.ylabel('Temperatur [°C]')
            plt.grid(True, linestyle='--', alpha=0.35); plt.legend()
            ax=plt.gca();
            ax.annotate(_info_text()+"\n" + (
                f"Verletzung: {value:.3f} °C vs. Grenze {limit:.3f} °C" if violated and np.isfinite(limit)
                else (f"Kein Verstoß, Extremwert: {value:.3f} °C" + (f" – Grenze {limit:.3f} °C" if np.isfinite(limit) else ""))
            ), xy=(0.98,0.98), xycoords='axes fraction', ha='right', va='top',
            bbox=dict(boxstyle='round', fc='white', ec='#444', alpha=0.85, lw=0.8), fontsize=9, zorder=10)
            plt.tight_layout()

    # Exporte Vorjahr
    if prev is not None:
        try:
            import pandas as pd
            hours_idx=np.arange(1,res['n_steps_year']+1,dtype=int)
            Qg_prev = prev['Q_build_series'][:res['n_steps_year']]
            qprime_eff_prev = (Qg_prev - prev['Q_solar_series'][:res['n_steps_year']])/pack['L_total']
            if config.EXPORT_EXCEL_LAST and config.ITERATION_TARGET=='last':
                with pd.ExcelWriter(config.EXCEL_FILENAME_LAST, engine='xlsxwriter') as w:
                    pd.DataFrame({'hour':hours_idx, 'W_per_m':qprime_eff_prev}).to_excel(w, index=False, sheet_name='profile_prev_year')
            if config.EXPORT_EXCEL_SOLAR and config.ITERATION_TARGET=='solar':
                with pd.ExcelWriter(config.EXCEL_FILENAME_SOLAR, engine='xlsxwriter') as w:
                    pd.DataFrame({'hour':hours_idx, 'W_per_m':qprime_eff_prev}).to_excel(w, index=False, sheet_name='profile_prev_year')
        except Exception as e:
            print(f"[export] Excel-Export fehlgeschlagen: {e}")

    try:
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        plt.show()


# ==========================
# GUI
# ==========================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('v13 – Longterm-Scaler (GUI)')
        self.geometry('1020x820')
        self.cfg = Config()
        self._cancel_evt: Optional[threading.Event] = None

        nb=ttk.Notebook(self); nb.pack(fill='both', expand=True)
        self.t_main=ttk.Frame(nb); self.t_iter=ttk.Frame(nb); self.t_crit=ttk.Frame(nb); self.t_plot=ttk.Frame(nb); self.t_path=ttk.Frame(nb)
        nb.add(self.t_main, text='Modell & Modi'); nb.add(self.t_iter, text='Iteration'); nb.add(self.t_crit, text='Grenzen'); nb.add(self.t_plot, text='Plots & Export'); nb.add(self.t_path, text='Dateipfade')

        self._make_main(); self._make_iter(); self._make_crit(); self._make_plot(); self._make_path()

        # Progress + Buttons
        bar=ttk.Frame(self); bar.pack(fill='x', pady=6)
        self.pvar=tk.IntVar(value=0); self.prog=ttk.Progressbar(bar, mode='determinate', maximum=100, variable=self.pvar)
        self.prog.pack(fill='x', padx=8)
        self.status=tk.StringVar(value='Bereit.')
        ttk.Label(bar, textvariable=self.status).pack(anchor='w', padx=8, pady=4)
        # Iterations-Energieanzeige (MWh/a)
        lf = ttk.LabelFrame(self, text='Energien aktuelle Iteration (MWh/a)')
        lf.pack(fill='x', padx=8, pady=(0,8))
        self.e_build=tk.StringVar(value='–'); self.e_solar=tk.StringVar(value='–'); self.e_net=tk.StringVar(value='–'); self.e_wel=tk.StringVar(value='–')
        self.lbl_build = ttk.Label(lf, text='Gebäude:'); self.lbl_build.grid(row=0,column=0,sticky='e',padx=8,pady=4)
        ttk.Label(lf, textvariable=self.e_build).grid(row=0,column=1,sticky='w')
        self.lbl_solar = ttk.Label(lf, text='Solar (Rückspeisung):'); self.lbl_solar.grid(row=0,column=2,sticky='e',padx=8)
        ttk.Label(lf, textvariable=self.e_solar).grid(row=0,column=3,sticky='w')
        self.lbl_net = ttk.Label(lf, text='Boden (effektiv):'); self.lbl_net.grid(row=0,column=4,sticky='e',padx=8)
        ttk.Label(lf, textvariable=self.e_net).grid(row=0,column=5,sticky='w')
        self.lbl_wel = ttk.Label(lf, text='W_el (COP):'); self.lbl_wel.grid(row=1,column=0,sticky='e',padx=8,pady=4)
        ttk.Label(lf, textvariable=self.e_wel).grid(row=1,column=1,sticky='w')

        btns=ttk.Frame(self); btns.pack(fill='x', pady=(0,8))
        ttk.Button(btns, text='Start', command=self._start).pack(side='left', padx=8)
        ttk.Button(btns, text='Abbrechen', command=self._cancel).pack(side='left')
        try:
            self.protocol('WM_DELETE_WINDOW', self._on_close)
        except Exception:
            pass

    # ---------- Tabs ----------
    def _make_main(self):
        f=self.t_main; pad=dict(padx=8,pady=6)
        self.var_gf=tk.StringVar(value=self.cfg.GF_METHOD)
        self.var_fold=tk.StringVar(value=self.cfg.FOLDING_MODE)
        self.var_bound=tk.StringVar(value=self.cfg.BOUNDARY)
        self.var_pipe=tk.StringVar(value=self.cfg.PIPE_TYPE)
        self.var_target=tk.StringVar(value=self.cfg.ITERATION_TARGET)
        self.var_prog=tk.BooleanVar(value=self.cfg.SHOW_PROGRESS)
        self.var_solar=tk.BooleanVar(value=self.cfg.USE_EXTRA_SOLAR_PROFILE)
        self.var_usecop=tk.BooleanVar(value=self.cfg.USE_COP)
        self.var_cop=tk.DoubleVar(value=self.cfg.COP_VALUE)
        self.var_field=tk.StringVar(value=self.cfg.FIELD_NAME)
        # Neu: Simulationsjahre
        self.var_years=tk.IntVar(value=self.cfg.YEARS)

        row=0
        ttk.Label(f,text='GF_METHOD:').grid(row=row,column=0,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_gf,values=['equivalent','similarities'],width=16).grid(row=row,column=1,**pad)
        ttk.Label(f,text='FOLDING_MODE:').grid(row=row,column=2,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_fold,values=['CJ','Liu','MLAA','hourly'],width=16).grid(row=row,column=3,**pad); row+=1
        ttk.Label(f,text='BOUNDARY:').grid(row=row,column=0,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_bound,values=['UBWT','UHTR'],width=16).grid(row=row,column=1,**pad)
        ttk.Label(f,text='PIPE_TYPE:').grid(row=row,column=2,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_pipe,values=['utube','coaxial'],width=16).grid(row=row,column=3,**pad); row+=1
        ttk.Label(f,text='ITERATION_TARGET:').grid(row=row,column=0,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_target,values=['last','solar'],width=16).grid(row=row,column=1,**pad)
        ttk.Checkbutton(f,text='SHOW_PROGRESS',variable=self.var_prog).grid(row=row,column=2,sticky='w',**pad)
        ttk.Checkbutton(f,text='USE_EXTRA_SOLAR_PROFILE',variable=self.var_solar,command=self._toggle_solar).grid(row=row,column=3,sticky='w',**pad); row+=1
        ttk.Label(f,text='FELD:').grid(row=row,column=0,sticky='e',**pad)
        ttk.Combobox(f,textvariable=self.var_field,values=['rectangle','ushaped','freeform'],width=16).grid(row=row,column=1,**pad); row+=1
        ttk.Checkbutton(f,text='USE_COP (nur Gebäudelast)',variable=self.var_usecop).grid(row=row,column=0,sticky='w',**pad)
        ttk.Label(f,text='COP_VALUE:').grid(row=row,column=1,sticky='e',**pad)
        ttk.Entry(f,textvariable=self.var_cop,width=10).grid(row=row,column=2,sticky='w',**pad); row+=1
        ttk.Label(f,text='Simulationsjahre:').grid(row=row,column=0,sticky='e',**pad)
        ttk.Entry(f,textvariable=self.var_years,width=10).grid(row=row,column=1,sticky='w',**pad); row+=1

        help_txt=("GF_METHOD: Feldreduktionsmethode. FOLDING_MODE: Zeitfaltung. BOUNDARY: UBWT/UHTR. "
                  "PIPE_TYPE: U‑Rohr/Coaxial. ITERATION_TARGET: welches Profil iteriert wird. "
                  "USE_EXTRA_SOLAR_PROFILE: Solar komplett ein/aus.")
        ttk.Label(f,text=help_txt,foreground='#444',wraplength=920,justify='left').grid(row=row,column=0,columnspan=4,sticky='w',padx=8,pady=(6,6))

    def _make_iter(self):
        f=self.t_iter; pad=dict(padx=8,pady=6)
        # Last
        self.var_last_mode=tk.StringVar(value=self.cfg.LAST_SCALING_MODE)
        self.var_last_start=tk.DoubleVar(value=self.cfg.START_ANNUAL_KWH_LAST)
        self.var_last_step=tk.DoubleVar(value=self.cfg.FINE_STEP_PCT_LAST)
        self.var_last_max=tk.IntVar(value=self.cfg.MAX_ITERS_LAST)
        lf1=ttk.LabelFrame(f,text='Gebäudelast (Last)')
        lf1.pack(fill='x',padx=8,pady=6)
        self.rb_last_iter=ttk.Radiobutton(lf1,text='iterate',variable=self.var_last_mode,value='iterate',command=self._sync_iter)
        self.rb_last_iter.grid(row=0,column=0,sticky='w',padx=8,pady=4)
        ttk.Radiobutton(lf1,text='fixed',variable=self.var_last_mode,value='fixed',command=self._sync_iter).grid(row=0,column=1,sticky='w',padx=8,pady=4)
        ttk.Radiobutton(lf1,text='original',variable=self.var_last_mode,value='original',command=self._sync_iter).grid(row=0,column=2,sticky='w',padx=8,pady=4)
        ttk.Label(lf1,text='START_ANNUAL_KWH_LAST:').grid(row=1,column=0,sticky='e'); ttk.Entry(lf1,textvariable=self.var_last_start,width=14).grid(row=1,column=1,sticky='w')
        ttk.Label(lf1,text='FINE_STEP_PCT_LAST:').grid(row=1,column=2,sticky='e'); ttk.Entry(lf1,textvariable=self.var_last_step,width=8).grid(row=1,column=3,sticky='w')
        ttk.Label(lf1,text='MAX_ITERS_LAST:').grid(row=1,column=4,sticky='e'); ttk.Entry(lf1,textvariable=self.var_last_max,width=8).grid(row=1,column=5,sticky='w')
        # Solar
        self.var_sol_mode=tk.StringVar(value=self.cfg.SOLAR_SCALING_MODE)
        self.var_sol_start=tk.DoubleVar(value=self.cfg.START_ANNUAL_KWH_SOLAR)
        self.var_sol_step=tk.DoubleVar(value=self.cfg.FINE_STEP_PCT_SOLAR)
        self.var_sol_max=tk.IntVar(value=self.cfg.MAX_ITERS_SOLAR)
        lf2=ttk.LabelFrame(f,text='Solar (Rückspeisung)')
        lf2.pack(fill='x',padx=8,pady=6)
        self.rb_sol_iter=ttk.Radiobutton(lf2,text='iterate',variable=self.var_sol_mode,value='iterate',command=self._sync_iter)
        self.rb_sol_iter.grid(row=0,column=0,sticky='w',padx=8,pady=4)
        ttk.Radiobutton(lf2,text='fixed',variable=self.var_sol_mode,value='fixed',command=self._sync_iter).grid(row=0,column=1,sticky='w',padx=8,pady=4)
        ttk.Radiobutton(lf2,text='original',variable=self.var_sol_mode,value='original',command=self._sync_iter).grid(row=0,column=2,sticky='w',padx=8,pady=4)
        ttk.Label(lf2,text='START_ANNUAL_KWH_SOLAR:').grid(row=1,column=0,sticky='e'); ttk.Entry(lf2,textvariable=self.var_sol_start,width=14).grid(row=1,column=1,sticky='w')
        ttk.Label(lf2,text='FINE_STEP_PCT_SOLAR:').grid(row=1,column=2,sticky='e'); ttk.Entry(lf2,textvariable=self.var_sol_step,width=8).grid(row=1,column=3,sticky='w')
        ttk.Label(lf2,text='MAX_ITERS_SOLAR:').grid(row=1,column=4,sticky='e'); ttk.Entry(lf2,textvariable=self.var_sol_max,width=8).grid(row=1,column=5,sticky='w')
        ttk.Label(f,text=("Genau EINE Seite auf 'iterate'. 'fixed' skaliert einmalig; 'original' lässt CSV unverändert."),foreground='#444',wraplength=920,justify='left').pack(fill='x',padx=10,pady=(8,6))

    def _make_crit(self):
        f=self.t_crit; pad=dict(padx=8,pady=4)
        # Last (Min)
        self.v_l_tf=tk.BooleanVar(value=True); self.v_l_e=tk.BooleanVar(value=False); self.v_l_tb=tk.BooleanVar(value=False); self.v_l_mm=tk.BooleanVar(value=False)
        self.v_l_tf_c=tk.DoubleVar(value=0.0); self.v_l_e_c=tk.DoubleVar(value=-5.0); self.v_l_tb_c=tk.DoubleVar(value=0.0); self.v_l_mm_c=tk.DoubleVar(value=0.0)
        # Neu: Monatsmittel-Grenzen Last
        self.v_l_mm_e=tk.BooleanVar(value=False); self.v_l_mm_tf=tk.BooleanVar(value=False)
        self.v_l_mm_e_c=tk.DoubleVar(value=0.0);  self.v_l_mm_tf_c=tk.DoubleVar(value=0.0)
        lf1=ttk.LabelFrame(f,text='Grenzen – Last (Min)'); lf1.pack(fill='x',padx=8,pady=6)
        ttk.Checkbutton(lf1,text='min T_f,avg',variable=self.v_l_tf).grid(row=0,column=0,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_tf_c,width=8).grid(row=0,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf1,text='min T_in',variable=self.v_l_e).grid(row=0,column=2,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_e_c,width=8).grid(row=0,column=3,sticky='w',**pad)
        ttk.Checkbutton(lf1,text='min T_b',variable=self.v_l_tb).grid(row=1,column=0,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_tb_c,width=8).grid(row=1,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf1,text='Monatsmittel T_b (min)',variable=self.v_l_mm).grid(row=1,column=2,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_mm_c,width=8).grid(row=1,column=3,sticky='w',**pad)
        ttk.Checkbutton(lf1,text='Monatsmittel T_in (min)',variable=self.v_l_mm_e).grid(row=2,column=0,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_mm_e_c,width=8).grid(row=2,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf1,text='Monatsmittel T_f,avg (min)',variable=self.v_l_mm_tf).grid(row=2,column=2,sticky='w',**pad); ttk.Entry(lf1,textvariable=self.v_l_mm_tf_c,width=8).grid(row=2,column=3,sticky='w',**pad)
        # Solar (Max)
        self.v_s_tf=tk.BooleanVar(value=True); self.v_s_e=tk.BooleanVar(value=False); self.v_s_tb=tk.BooleanVar(value=False); self.v_s_mm=tk.BooleanVar(value=False)
        self.v_s_tf_c=tk.DoubleVar(value=20.0); self.v_s_e_c=tk.DoubleVar(value=20.0); self.v_s_tb_c=tk.DoubleVar(value=25.0); self.v_s_mm_c=tk.DoubleVar(value=20.0)
        # Neu: Monatsmittel-Grenzen Solar
        self.v_s_mm_e=tk.BooleanVar(value=False); self.v_s_mm_tf=tk.BooleanVar(value=False)
        self.v_s_mm_e_c=tk.DoubleVar(value=20.0); self.v_s_mm_tf_c=tk.DoubleVar(value=20.0)
        lf2=ttk.LabelFrame(f,text='Grenzen – Solar (Max)'); lf2.pack(fill='x',padx=8,pady=6)
        ttk.Checkbutton(lf2,text='max T_f,avg',variable=self.v_s_tf).grid(row=0,column=0,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_tf_c,width=8).grid(row=0,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf2,text='max T_in',variable=self.v_s_e).grid(row=0,column=2,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_e_c,width=8).grid(row=0,column=3,sticky='w',**pad)
        ttk.Checkbutton(lf2,text='max T_b',variable=self.v_s_tb).grid(row=1,column=0,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_tb_c,width=8).grid(row=1,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf2,text='Monatsmittel T_b (max)',variable=self.v_s_mm).grid(row=1,column=2,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_mm_c,width=8).grid(row=1,column=3,sticky='w',**pad)
        ttk.Checkbutton(lf2,text='Monatsmittel T_in (max)',variable=self.v_s_mm_e).grid(row=2,column=0,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_mm_e_c,width=8).grid(row=2,column=1,sticky='w',**pad)
        ttk.Checkbutton(lf2,text='Monatsmittel T_f,avg (max)',variable=self.v_s_mm_tf).grid(row=2,column=2,sticky='w',**pad); ttk.Entry(lf2,textvariable=self.v_s_mm_tf_c,width=8).grid(row=2,column=3,sticky='w',**pad)
        # Drift-Kriterien
        self.v_drift = tk.BooleanVar(value=False)
        self.v_drift_c = tk.DoubleVar(value=0.10)
        self.v_drift_ann = tk.BooleanVar(value=False)
        self.v_drift_ann_c = tk.DoubleVar(value=0.00)
        lf3 = ttk.LabelFrame(f, text='Langzeit-Drift')
        lf3.pack(fill='x', padx=8, pady=6)
        ttk.Checkbutton(lf3, text='max Drift min Monatsmittel T_b [K/Jahr]', variable=self.v_drift).grid(row=0, column=0, sticky='w', **pad)
        ttk.Entry(lf3, textvariable=self.v_drift_c, width=8).grid(row=0, column=1, sticky='w', **pad)
        ttk.Checkbutton(lf3, text='max Drift Jahresmittel T_b [K/Jahr]', variable=self.v_drift_ann).grid(row=1, column=0, sticky='w', **pad)
        ttk.Entry(lf3, textvariable=self.v_drift_ann_c, width=8).grid(row=1, column=1, sticky='w', **pad)
        ttk.Label(f,text=("Links: Grenzen für Last (Min), rechts: für Solar (Max). Aktiv ist der Satz, der zum Ziel passt."),foreground='#444',wraplength=920,justify='left').pack(fill='x',padx=10,pady=(8,6))

    def _make_plot(self):
        f=self.t_plot; pad=dict(padx=8,pady=6)
        self.p_temp=tk.BooleanVar(value=True); self.p_mm=tk.BooleanVar(value=True); self.p_prev=tk.BooleanVar(value=True); self.p_viol=tk.BooleanVar(value=True); self.p_ground=tk.BooleanVar(value=True)
        self.p_viol_detail=tk.BooleanVar(value=True)
        self.p_drift=tk.BooleanVar(value=True)
        ttk.Checkbutton(f,text='Temperaturen (Zeitreihen)',variable=self.p_temp).grid(row=0,column=0,sticky='w',**pad)
        ttk.Checkbutton(f,text='Monatsmittel (Linie+Punkte)',variable=self.p_mm).grid(row=1,column=0,sticky='w',**pad)
        ttk.Checkbutton(f,text='Profil Vorjahr (Ziel)',variable=self.p_prev).grid(row=2,column=0,sticky='w',**pad)
        ttk.Checkbutton(f,text='Profil Verletzungsjahr (Ziel)',variable=self.p_viol).grid(row=3,column=0,sticky='w',**pad)
        ttk.Checkbutton(f,text='Bodenprofil (effektiv)',variable=self.p_ground).grid(row=4,column=0,sticky='w',**pad)
        ttk.Checkbutton(f,text='Verletzungs‑Detail (Monat + Grenzlinie)',variable=self.p_viol_detail).grid(row=2,column=1,sticky='w',**pad)
        ttk.Checkbutton(f,text='Drift min Monatsmittel T_b',variable=self.p_drift).grid(row=3,column=1,sticky='w',**pad)
        self.exp_last=tk.BooleanVar(value=True); self.exp_solar=tk.BooleanVar(value=False)
        ttk.Checkbutton(f,text='Excel Export Vorjahr – Last',variable=self.exp_last).grid(row=0,column=1,sticky='w',**pad)
        ttk.Checkbutton(f,text='Excel Export Vorjahr – Solar',variable=self.exp_solar).grid(row=1,column=1,sticky='w',**pad)
        ttk.Label(f,text=("Profilplots zeigen das effektive Profil, das an pygfunction geht (Gebäude_ground − Solar)."),foreground='#444',wraplength=920,justify='left').grid(row=5,column=0,columnspan=2,sticky='w',padx=8,pady=(6,6))

    def _make_path(self):
        f=self.t_path; pad=dict(padx=8,pady=6)
        # Toggle: synthetisches Lastprofil (Testmodus) – CSV-Felder ausgrauen
        self.var_use_synth = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text='Synthetisches Lastprofil (Testmodus) – CSV-Pfade deaktivieren',
                        variable=self.var_use_synth, command=self._toggle_profile_source).grid(row=0,column=0,columnspan=3,sticky='w',**pad)

        # CSV-Pfade (werden bei Synthetic deaktiviert)
        self.p_last=tk.StringVar(value='qprime_profile_from_excel.csv')
        self.p_solar=tk.StringVar(value='qprime_recharge_only.csv')
        ttk.Label(f,text='CSV Gebäudelast (hour;W_per_m, 8760):').grid(row=1,column=0,sticky='e',**pad)
        self.e_last = ttk.Entry(f,textvariable=self.p_last, width=58)
        self.e_last.grid(row=1,column=1,sticky='we',**pad)
        self.b_last = ttk.Button(f,text='…',command=lambda:self._pick(self.p_last))
        self.b_last.grid(row=1,column=2,**pad)
        ttk.Label(f,text='CSV Solar-only (hour;W_per_m, 8760):').grid(row=2,column=0,sticky='e',**pad)
        self.e_solar = ttk.Entry(f,textvariable=self.p_solar,width=58)
        self.e_solar.grid(row=2,column=1,sticky='we',**pad)
        self.b_solar = ttk.Button(f,text='…',command=lambda:self._pick(self.p_solar))
        self.b_solar.grid(row=2,column=2,**pad)
        ttk.Label(f,text=("Wenn Solar deaktiviert ist, wird das Solar-Profil ignoriert. Bei synthetischem Profil wird Solar automatisch deaktiviert."),
                  foreground='#444',wraplength=920,justify='left').grid(row=3,column=0,columnspan=3,sticky='w',padx=8,pady=(6,6))

    # ---------- Helpers ----------
    def _pick(self,var):
        p=filedialog.askopenfilename(title='CSV wählen', filetypes=[('CSV','*.csv'),('Alle','*.*')])
        if p: var.set(p)

    def _toggle_solar(self):
        # Bei deaktiviertem Solar Ziel auf 'last'
        if not self.var_solar.get(): self.var_target.set('last')
        self._sync_iter()

    def _toggle_profile_source(self):
        """CSV-Pfade ein-/ausgrauen bei synthetischem Profil; Solar deaktivieren."""
        try:
            synth = bool(self.var_use_synth.get())
            state = 'disabled' if synth else '!disabled'
            for w in (self.e_last, self.b_last, self.e_solar, self.b_solar):
                try:
                    w.state((state,))
                except Exception:
                    try:
                        w.configure(state=('disabled' if synth else 'normal'))
                    except Exception:
                        pass
            if synth:
                try:
                    self.var_solar.set(False)
                    self._sync_iter()
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_iter(self):
        # Robust ohne Walrus-Operator: Attribute vorbereiten
        if not hasattr(self, 'var_last_mode'):
            self.var_last_mode = tk.StringVar(value='iterate')
        if not hasattr(self, 'var_sol_mode'):
            self.var_sol_mode = tk.StringVar(value='original')
        last_iter = (self.var_last_mode.get()=='iterate')
        sol_iter  = (self.var_sol_mode.get()=='iterate')
        # Exklusivität
        try:
            if self.var_solar.get():
                if self.var_last_mode.get()=='iterate':
                    # Gebäudelast iteriert → Solar darf nicht iterieren
                    self.rb_sol_iter.state(('disabled',)); self.rb_last_iter.state(('!disabled',))
                    self.var_target.set('last')
                elif self.var_sol_mode.get()=='iterate':
                    # Solar iteriert → Gebäudelast darf nicht iterieren
                    self.rb_last_iter.state(('disabled',)); self.rb_sol_iter.state(('!disabled',))
                    self.var_target.set('solar')
                else:
                    # Keiner iteriert: beide Buttons aktiv lassen
                    self.rb_last_iter.state(('!disabled',)); self.rb_sol_iter.state(('!disabled',))
            else:
                # Solar komplett aus → Ziel muss 'last' sein
                self.rb_last_iter.state(('!disabled',)); self.rb_sol_iter.state(('disabled',)); self.var_sol_mode.set('original')
                self.var_target.set('last')
        except Exception:
            pass

    def _start(self):
        # Build Config aus GUI
        cfg=Config()
        cfg.GF_METHOD=self.var_gf.get(); cfg.FOLDING_MODE=self.var_fold.get(); cfg.BOUNDARY=self.var_bound.get(); cfg.PIPE_TYPE=self.var_pipe.get()
        cfg.SHOW_PROGRESS=bool(self.var_prog.get()); cfg.USE_EXTRA_SOLAR_PROFILE=bool(self.var_solar.get())
        cfg.USE_COP=bool(self.var_usecop.get()); cfg.COP_VALUE=float(self.var_cop.get()) if self.var_usecop.get() else cfg.COP_VALUE
        cfg.FIELD_NAME=self.var_field.get()
        # Falls Rechteck/U gewählt, aber keine Bohrlöcher vorhanden (rows/cols=0), freundlich abbrechen.
        try:
            import borefields as _bf
            chosen = (cfg.FIELD_NAME or '').strip().lower()
            if chosen in ('rectangle','ushaped','u_shaped','u-shaped'):
                obj = getattr(_bf, 'rectangle_field' if chosen=='rectangle' else 'U_shaped_field', None)
                n = 0
                if obj is not None:
                    try:
                        n = len(obj)
                    except TypeError:
                        n = len(list(obj))
                if int(n) <= 0:
                    # Prüfe, ob Freeform verfügbar ist → Hinweis geben
                    ff = getattr(_bf, 'freeform_field', None)
                    try:
                        nff = len(ff) if ff is not None else 0
                    except TypeError:
                        nff = len(list(ff)) if ff is not None else 0
                    if nff and int(nff) > 0:
                        messagebox.showwarning('Hinweis', 'Das gewählte Rechteck/U‑Feld hat 0 Bohrlöcher (rows/cols=0).\nFreeform ist aktiv. Bitte wähle in der Feld‑Auswahl „freeform“ oder lege zuvor ein gültiges Rechteck/U‑Feld fest.')
                    else:
                        messagebox.showwarning('Hinweis', 'Das gewählte Feld hat 0 Bohrlöcher (rows/cols=0). Bitte Feldparameter prüfen oder „freeform“ wählen (falls vorhanden).')
                    return
        except Exception:
            pass
        # Profilquelle aus Toggle bestimmen
        try:
            cfg.PROFILE_SOURCE = 'synthetic' if bool(getattr(self,'var_use_synth',tk.BooleanVar(value=False)).get()) else 'csv'
        except Exception:
            cfg.PROFILE_SOURCE = 'csv'
        if cfg.PROFILE_SOURCE == 'synthetic':
            cfg.USE_EXTRA_SOLAR_PROFILE = False
        # Übernehme Simulationsjahre (>=1)
        try:
            cfg.YEARS=max(1,int(self.var_years.get()))
        except Exception:
            cfg.YEARS=max(1,int(cfg.YEARS))
        # Iteration: Exklusivität prüfen/erzwingen basierend auf den Radiobuttons
        cfg.ITERATION_TARGET=self.var_target.get()
        cfg.LAST_SCALING_MODE=getattr(self,'var_last_mode',tk.StringVar(value='iterate')).get()
        cfg.SOLAR_SCALING_MODE=getattr(self,'var_sol_mode',tk.StringVar(value='original')).get()

        def _apply_resolution(target: str, last_mode: str, sol_mode: str):
            cfg.ITERATION_TARGET=target
            cfg.LAST_SCALING_MODE=last_mode
            cfg.SOLAR_SCALING_MODE=sol_mode
            # UI zurückspiegeln (damit Nutzer sieht, was wirklich gilt)
            try:
                self.var_target.set(target)
                self.var_last_mode.set(last_mode)
                self.var_sol_mode.set(sol_mode)
            except Exception:
                pass

        if not cfg.USE_EXTRA_SOLAR_PROFILE:
            # Solar deaktiviert → immer Gebäudelast iterieren
            _apply_resolution('last', 'iterate', 'original')
        else:
            last_iter = (cfg.LAST_SCALING_MODE == 'iterate')
            sol_iter  = (cfg.SOLAR_SCALING_MODE == 'iterate')
            if last_iter and not sol_iter:
                _apply_resolution('last', 'iterate', cfg.SOLAR_SCALING_MODE)
            elif sol_iter and not last_iter:
                _apply_resolution('solar', cfg.LAST_SCALING_MODE, 'iterate')
            elif last_iter and sol_iter:
                # Konflikt: Nimm die Auswahl aus ITERATION_TARGET, schalte Gegenseite um
                if cfg.ITERATION_TARGET == 'solar':
                    _apply_resolution('solar', 'original', 'iterate')
                else:
                    _apply_resolution('last', 'iterate', 'original')
            else:  # keiner iteriert → richte dich nach ITERATION_TARGET und mache diese Seite 'iterate'
                if cfg.ITERATION_TARGET == 'solar':
                    _apply_resolution('solar', cfg.LAST_SCALING_MODE if cfg.LAST_SCALING_MODE!='iterate' else 'original', 'iterate')
                else:
                    _apply_resolution('last', 'iterate', cfg.SOLAR_SCALING_MODE if cfg.SOLAR_SCALING_MODE!='iterate' else 'original')
        # Zahlen
        cfg.START_ANNUAL_KWH_LAST=float(getattr(self,'var_last_start',tk.DoubleVar(value=cfg.START_ANNUAL_KWH_LAST)).get())
        cfg.FINE_STEP_PCT_LAST  =float(getattr(self,'var_last_step',tk.DoubleVar(value=cfg.FINE_STEP_PCT_LAST)).get())
        cfg.MAX_ITERS_LAST      =int(  getattr(self,'var_last_max', tk.IntVar(value=cfg.MAX_ITERS_LAST)).get())
        cfg.START_ANNUAL_KWH_SOLAR=float(getattr(self,'var_sol_start',tk.DoubleVar(value=cfg.START_ANNUAL_KWH_SOLAR)).get())
        cfg.FINE_STEP_PCT_SOLAR  =float(getattr(self,'var_sol_step',tk.DoubleVar(value=cfg.FINE_STEP_PCT_SOLAR)).get())
        cfg.MAX_ITERS_SOLAR      =int(  getattr(self,'var_sol_max', tk.IntVar(value=cfg.MAX_ITERS_SOLAR)).get())
        # Grenzen
        cfg.USE_LAST_MIN_TFAVG =bool(getattr(self,'v_l_tf',tk.BooleanVar(value=cfg.USE_LAST_MIN_TFAVG)).get())
        cfg.USE_LAST_MIN_EWT   =bool(getattr(self,'v_l_e', tk.BooleanVar(value=cfg.USE_LAST_MIN_EWT)).get())
        cfg.USE_LAST_MIN_TB    =bool(getattr(self,'v_l_tb',tk.BooleanVar(value=cfg.USE_LAST_MIN_TB)).get())
        cfg.USE_LAST_MONTH_MEAN_TB =bool(getattr(self,'v_l_mm',tk.BooleanVar(value=cfg.USE_LAST_MONTH_MEAN_TB)).get())
        cfg.USE_LAST_MONTH_MEAN_EWT=bool(getattr(self,'v_l_mm_e',tk.BooleanVar(value=cfg.USE_LAST_MONTH_MEAN_EWT)).get())
        cfg.USE_LAST_MONTH_MEAN_TFAVG=bool(getattr(self,'v_l_mm_tf',tk.BooleanVar(value=cfg.USE_LAST_MONTH_MEAN_TFAVG)).get())
        cfg.LAST_MIN_TFAVG_C   =float(getattr(self,'v_l_tf_c',tk.DoubleVar(value=cfg.LAST_MIN_TFAVG_C)).get())
        cfg.LAST_MIN_EWT_C     =float(getattr(self,'v_l_e_c', tk.DoubleVar(value=cfg.LAST_MIN_EWT_C)).get())
        cfg.LAST_MIN_TB_C      =float(getattr(self,'v_l_tb_c',tk.DoubleVar(value=cfg.LAST_MIN_TB_C)).get())
        cfg.LAST_MONTH_MEAN_TB_C=float(getattr(self,'v_l_mm_c',tk.DoubleVar(value=cfg.LAST_MONTH_MEAN_TB_C)).get())
        cfg.LAST_MONTH_MEAN_EWT_C=float(getattr(self,'v_l_mm_e_c',tk.DoubleVar(value=cfg.LAST_MONTH_MEAN_EWT_C)).get())
        cfg.LAST_MONTH_MEAN_TFAVG_C=float(getattr(self,'v_l_mm_tf_c',tk.DoubleVar(value=cfg.LAST_MONTH_MEAN_TFAVG_C)).get())
        cfg.USE_SOLAR_MAX_TFAVG =bool(getattr(self,'v_s_tf',tk.BooleanVar(value=cfg.USE_SOLAR_MAX_TFAVG)).get())
        cfg.USE_SOLAR_MAX_EWT   =bool(getattr(self,'v_s_e', tk.BooleanVar(value=cfg.USE_SOLAR_MAX_EWT)).get())
        cfg.USE_SOLAR_MAX_TB    =bool(getattr(self,'v_s_tb',tk.BooleanVar(value=cfg.USE_SOLAR_MAX_TB)).get())
        cfg.USE_SOLAR_MONTH_MEAN_TB =bool(getattr(self,'v_s_mm',tk.BooleanVar(value=cfg.USE_SOLAR_MONTH_MEAN_TB)).get())
        cfg.USE_SOLAR_MONTH_MEAN_EWT=bool(getattr(self,'v_s_mm_e',tk.BooleanVar(value=cfg.USE_SOLAR_MONTH_MEAN_EWT)).get())
        cfg.USE_SOLAR_MONTH_MEAN_TFAVG=bool(getattr(self,'v_s_mm_tf',tk.BooleanVar(value=cfg.USE_SOLAR_MONTH_MEAN_TFAVG)).get())
        cfg.SOLAR_MAX_TFAVG_C   =float(getattr(self,'v_s_tf_c',tk.DoubleVar(value=cfg.SOLAR_MAX_TFAVG_C)).get())
        cfg.SOLAR_MAX_EWT_C     =float(getattr(self,'v_s_e_c', tk.DoubleVar(value=cfg.SOLAR_MAX_EWT_C)).get())
        cfg.SOLAR_MAX_TB_C      =float(getattr(self,'v_s_tb_c',tk.DoubleVar(value=cfg.SOLAR_MAX_TB_C)).get())
        cfg.SOLAR_MONTH_MEAN_TB_C=float(getattr(self,'v_s_mm_c',tk.DoubleVar(value=cfg.SOLAR_MONTH_MEAN_TB_C)).get())
        cfg.SOLAR_MONTH_MEAN_EWT_C=float(getattr(self,'v_s_mm_e_c',tk.DoubleVar(value=cfg.SOLAR_MONTH_MEAN_EWT_C)).get())
        cfg.SOLAR_MONTH_MEAN_TFAVG_C=float(getattr(self,'v_s_mm_tf_c',tk.DoubleVar(value=cfg.SOLAR_MONTH_MEAN_TFAVG_C)).get())
        # Drift
        cfg.USE_DRIFT_MIN_MONTH_MEAN_TB = bool(getattr(self,'v_drift', tk.BooleanVar(value=cfg.USE_DRIFT_MIN_MONTH_MEAN_TB)).get())
        cfg.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR = float(getattr(self,'v_drift_c', tk.DoubleVar(value=cfg.DRIFT_MIN_MONTH_MEAN_TB_MAX_K_PER_YEAR)).get())
        cfg.USE_DRIFT_ANNUAL_MEAN_TB = bool(getattr(self,'v_drift_ann', tk.BooleanVar(value=cfg.USE_DRIFT_ANNUAL_MEAN_TB)).get())
        cfg.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR = float(getattr(self,'v_drift_ann_c', tk.DoubleVar(value=cfg.DRIFT_ANNUAL_MEAN_TB_MAX_K_PER_YEAR)).get())
        # Plots & Export
        cfg.SHOW_PLOT_TEMP_SERIES=bool(self.p_temp.get()); cfg.SHOW_PLOT_MONTHLY_MEANS=bool(self.p_mm.get())
        cfg.SHOW_PLOT_TARGET_PROFILE_PREV=bool(self.p_prev.get()); cfg.SHOW_PLOT_TARGET_PROFILE_VIOL=bool(self.p_viol.get())
        cfg.SHOW_PLOT_GROUND_PROFILE=bool(self.p_ground.get()); cfg.SHOW_PLOT_VIOLATION_DETAIL=bool(self.p_viol_detail.get())
        cfg.SHOW_PLOT_DRIFT_MIN_TB=bool(self.p_drift.get())
        cfg.EXPORT_EXCEL_LAST=bool(self.exp_last.get()); cfg.EXPORT_EXCEL_SOLAR=bool(self.exp_solar.get())
        # Pfade
        cfg.CSV_PATH=self.p_last.get(); cfg.CSV_PATH_RECHARGE=self.p_solar.get()
        # Vorab-Validierung der CSVs, um Hänger zu vermeiden
        try:
            import os
            if cfg.PROFILE_SOURCE.lower()=='csv':
                if not os.path.exists(cfg.CSV_PATH):
                    messagebox.showerror('Fehlende Datei', f"CSV Gebäudelast nicht gefunden:\n{cfg.CSV_PATH}")
                    return
                if cfg.USE_EXTRA_SOLAR_PROFILE and not os.path.exists(cfg.CSV_PATH_RECHARGE):
                    messagebox.showwarning('Hinweis', f"CSV Solar-only nicht gefunden:\n{cfg.CSV_PATH_RECHARGE}\nSolar wird ignoriert.")
                    cfg.USE_EXTRA_SOLAR_PROFILE=False
                # Zusätzliche Logik: Warnen, wenn Gebäudelast negative Werte enthält UND Solar-CSV aktiv ist
                try:
                    qtest = _read_qprime_csv(cfg.CSV_PATH)
                    if cfg.USE_EXTRA_SOLAR_PROFILE and np.any(qtest < 0.0):
                        proceed = messagebox.askokcancel(
                            'Warnung – doppelte Rückspeisung möglich',
                            'Das Gebäudelast‑Profil enthält negative Werte (interne Rückspeisung),\n'
                            'gleichzeitig ist die optionale Solar‑CSV aktiv.\n\n'
                            'Damit würde die Rückspeisung doppelt berücksichtigt werden.\n\n'
                            'OK = trotzdem fortfahren\nAbbrechen = zurück zur Eingabe')
                        if not proceed:
                            return
                except Exception:
                    pass
        except Exception:
            pass

        # Vorab: JSON-Metadaten prüfen und Zusammenfassung anzeigen
        try:
            # aktuelles Feld (gemäß Auswahl in cfg.FIELD_NAME)
            _, field = choose_field_by_name(cfg.FIELD_NAME)
            try:
                n_bh_cur = len(field)
            except TypeError:
                n_bh_cur = len(list(field))
            H_cur = float(getattr(bd, 'H'))
            L_cur = max(1.0, n_bh_cur * H_cur)

            meta_last = _read_profile_meta_for_csv(cfg.CSV_PATH) if cfg.PROFILE_SOURCE.lower()=='csv' else None
            meta_solar = _read_profile_meta_for_csv(cfg.CSV_PATH_RECHARGE) if (cfg.PROFILE_SOURCE.lower()=='csv' and cfg.USE_EXTRA_SOLAR_PROFILE) else None

            def _L_ref(meta):
                if not meta or not isinstance(meta.get('field'), dict):
                    return None
                n_ref = float(meta['field'].get('n_bh', n_bh_cur))
                H_ref = float(meta['field'].get('H_m', H_cur))
                return max(1.0, n_ref * H_ref)

            L_ref_last = _L_ref(meta_last)
            L_ref_solar = _L_ref(meta_solar)

            # Warnen, wenn JSON fehlt
            missing = []
            if cfg.PROFILE_SOURCE.lower()=='csv' and meta_last is None:
                missing.append('Gebäudelast-JSON (qprime_profile_from_excel.json)')
            if cfg.PROFILE_SOURCE.lower()=='csv' and cfg.USE_EXTRA_SOLAR_PROFILE and meta_solar is None:
                missing.append('Solar-JSON (qprime_recharge_only.json)')
            if missing:
                proceed = messagebox.askokcancel(
                    'Warnung – fehlende Metadaten',
                    'Es fehlen Metadaten (JSON) für:\n  - ' + '\n  - '.join(missing) +
                    '\n\nOhne JSON skaliert sich die Energie mit den aktuellen Bohrmetern.\n\nOK = trotzdem fortfahren\nAbbrechen = zurück zur Eingabe')
                if not proceed:
                    return

            # Start-Zusammenfassung
            def _fmt(x):
                try:
                    return f"{float(x):,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                except Exception:
                    return str(x)
            r_solar = (L_ref_solar / L_cur) if L_ref_solar else 1.0
            info_lines = [
                f"Aktuelles Feld (für pyg): n_bh={n_bh_cur}, H={H_cur:.2f} m, L_total={_fmt(L_cur)} m",
                f"Profil-Feld (Gebäudelast): L_ref={_fmt(L_ref_last) if L_ref_last else 'n/a'} m",
                f"Profil-Feld (Solar):       L_ref={_fmt(L_ref_solar) if L_ref_solar else 'n/a'} m",
                f"Umrechnung: Q_build = q′_build · L_ref;  q′_solar_scaled = q′_solar · (L_ref / L_cur) = × {_fmt(r_solar)}"
            ]
            try:
                messagebox.showinfo('Startparameter', "\n".join(info_lines))
            except Exception:
                pass
        except Exception:
            pass

        # Lauf starten
        self._cancel_evt=threading.Event()
        def _prog(i,total,label):
            try:
                pct=int((i+1)/max(1,total)*100)
                self.pvar.set(pct); self.status.set(f"{label} – {pct}%")
                self.update_idletasks()
            except Exception:
                pass
        def set_iter_energies(Eb_mwh: float, Es_mwh: float, Enet_mwh: float, Wel_mwh: float, it: int):
            try:
                self.e_build.set(f"{Eb_mwh:,.3f}".replace(',', 'X').replace('.', ',').replace('X','.'))
                self.e_solar.set(f"{Es_mwh:,.3f}".replace(',', 'X').replace('.', ',').replace('X','.'))
                self.e_net.set(  f"{Enet_mwh:,.3f}".replace(',', 'X').replace('.', ',').replace('X','.'))
                self.e_wel.set(  f"{Wel_mwh:,.3f}".replace(',', 'X').replace('.', ',').replace('X','.'))
            except Exception:
                pass

        def worker():
            try:
                # Kurzer Start‑Report in der Statuszeile
                self.status.set(f"Starte: target={cfg.ITERATION_TARGET}, last={cfg.LAST_SCALING_MODE}, solar={cfg.SOLAR_SCALING_MODE}")
                pack=run_iteration(cfg, _prog, set_iter_energies, self._cancel_evt)
                if self._cancel_evt.is_set():
                    self.status.set('Abgebrochen.')
                else:
                    # Finale UI‑Aktionen inkl. Plots auf dem Haupt‑Thread durchführen
                    def _finish_ui():
                        # Panel-Labels ggf. mit COP ergänzen
                        try:
                            if cfg.USE_COP:
                                fac = (1.0 - 1.0/max(1e-9,cfg.COP_VALUE))
                                self.lbl_wel.config(text=f'W_el (COP={cfg.COP_VALUE:.2f}; Faktor={fac:.3f}):')
                            else:
                                self.lbl_wel.config(text='W_el (COP):')
                        except Exception:
                            pass
                        viol=pack.get('violation')
                        if viol:
                            side='Gebäudelast' if viol.get('side')=='last' else 'Solar'
                            kind=viol.get('kind','')
                            val=viol.get('value',float('nan'))
                            lim=viol.get('limit',float('nan'))
                            itn=viol.get('iter',0)
                            extra=viol.get('extra','')
                            msg=(f"Iteration gestoppt (It. {itn}).\n"
                                 f"Aktiver Grenzwert ({side}): {kind}.\n"
                                 f"Messwert: {val:.3f} °C, Grenze: {lim:.3f} °C.\n"
                                 f"{extra}")
                            try:
                                messagebox.showinfo('Abbruch durch Grenzwert', msg)
                            except Exception:
                                pass
                        self.status.set('Fertig. Erzeuge Plots …')
                        try:
                            plots_and_exports(cfg, pack)
                        finally:
                            self.status.set('Fertig.')
                    try:
                        self.after(0, _finish_ui)
                    except Exception:
                        # Fallback: direkt aufrufen (z. B. falls kein Tk‑Loop)
                        _finish_ui()
            except Exception as e:
                self.status.set(f'Fehler: {e}')
                messagebox.showerror('Fehler', str(e))
        threading.Thread(target=worker, daemon=True).start()
        # Kurzer Start‑Dialog mit Zusammenfassung
        try:
            messagebox.showinfo('Start', (
                f"Simulation gestartet.\n"
                f"Iteriere: {'Gebäudelast' if cfg.ITERATION_TARGET=='last' else 'Solar'}\n"
                f"Modi – Last: {cfg.LAST_SCALING_MODE}, Solar: {cfg.SOLAR_SCALING_MODE}\n"
                f"Plots erscheinen in Matplotlib; Fortschritt unten."))
        except Exception:
            pass

    def _cancel(self):
        if self._cancel_evt:
            self._cancel_evt.set()
            # Hinweis: Abbruch greift nach dem aktuellen g-function Schritt
            self.status.set('Breche ab … (wirksam nach aktuellem Schritt)')
            # Start-Button deaktivieren, um Doppelstarts zu vermeiden
            try:
                for child in self.winfo_children():
                    for g in getattr(child, 'winfo_children', lambda: [])():
                        if getattr(g, 'cget', None) and g.cget('text') == 'Start':
                            g.configure(state='disabled')
            except Exception:
                pass

    def _on_close(self):
        try:
            self._cancel()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
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
