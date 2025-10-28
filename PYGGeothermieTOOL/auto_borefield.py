#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_borefield.py
-----------------
Automatischer Bohrfeld-Generator für dein Projekt (Version 5).

Neu (Mode 'fitmax'):
- Bestimmt zuerst die maximale Anzahl Bohrlöcher N_in, die mit gegebenem Abstand
  in das (beliebig geformte) Polygon passen (inkl. optionalem Rotations- & Phasen-Scan).
- Erzeugt anschließend ein Rechteck- oder U-Layout mit GLEICHER N und gleichem Abstand
  (als standardisierte Parametrierung für die Simulation).
- Plot-Overlay: schwarze Punkte = tatsächliche Punkte im Polygon; blaue Punkte = Rechteck/U-Layout.

Kompatibel zu Version 4 (Mode 'legacy'): erzeugt weiterhin ein Rechteck/U-Layout,
das innerhalb des Polygons liegt (wie bisher).

Arbeitet mit den gleichen Ordnern/Backups:
- Artefakte & Backups in: <Projektordner>/autoborehole
- Liest Shapefile aus:   <Projektordner>/shp

Änderbare Projektwerte (bei --overwrite):
- Data_Base.py & borefields.py: rows_sonden, Columns_sonden, Bx, By, H
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

ARTIFACT_SUBDIR = "autoborehole"

def _artifact_dir(project_dir: Path) -> Path:
    d = project_dir / ARTIFACT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d

def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Drittbibliotheken ---
try:
    import shapefile  # pyshp
except Exception:
    print("Fehler: pyshp (shapefile) ist nicht installiert. 'pip install pyshp' oder conda-forge.", file=sys.stderr); raise

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.ops import unary_union
    from shapely import affinity
except Exception:
    print("Fehler: shapely ist nicht installiert. 'pip install shapely' oder conda-forge.", file=sys.stderr); raise

try:
    from pyproj import CRS, Transformer
except Exception:
    print("Fehler: pyproj ist nicht installiert. 'pip install pyproj' oder conda-forge.", file=sys.stderr); raise

# Matplotlib ist optional (für Preview)
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# Tkinter-GUI (optional)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False


# -----------------------------
# Hilfs-Datentypen und Utils
# -----------------------------

@dataclass
class LayoutParams:
    mode: str             # "fitmax" | "legacy"
    layout_type: str      # "rectangle" | "ushaped"
    spacing_m: float      # Bx=By
    depth_m: float        # H
    edge_offset_m: float  # Sicherheitsabstand zum Polygonrand
    u_gap_m: float        # nur für U-Shape: Breite des Innenhofs (quer)
    overwrite: bool       # Projektdateien überschreiben?
    shp_name: Optional[str] = None         # Auswahl einer bestimmten SHP-Datei (ohne Endung)
    rotate_step_deg: float = 0.0           # 0 -> keine Rotation, sonst Schrittweite [°] (z.B. 5)
    phase_steps: int = 1                   # 1 -> keine Phasenverschiebung, sonst z.B. 3/5


@dataclass
class FieldResult:
    rows: int
    cols: int
    count: int
    bbox_w_m: float
    bbox_h_m: float
    points_polygon: List[Tuple[float, float]]   # Punkte, die wirklich im Polygon liegen (fitmax)
    points_layout:  List[Tuple[float, float]]   # Layout-Punkte (rechteckig / U)
    crs_epsg: Optional[int]
    shp_used: str
    meta_extra: dict


def find_shp_in_folder(shp_dir: Path) -> List[Path]:
    return sorted(shp_dir.glob("*.shp"))


def read_polygon_from_shp(shp_path: Path) -> Tuple[Polygon, CRS]:
    prj_path = shp_path.with_suffix(".prj")
    shx_path = shp_path.with_suffix(".shx")
    dbf_path = shp_path.with_suffix(".dbf")
    for p in (prj_path, shx_path, dbf_path):
        if not p.exists():
            raise FileNotFoundError(f"Fehlende Shapefile-Begleitdatei: {p.name}")

    r = shapefile.Reader(str(shp_path))
    shapes = r.shapes()
    if not shapes:
        raise RuntimeError("Shapefile enthält keine Geometrien.")

    polys = []
    for sh in shapes:
        if sh.shapeTypeName.upper() not in ("POLYGON", "POLYGONZ", "MULTIPATCH"):
            continue
        pts = sh.points
        parts = list(sh.parts) + [len(pts)]
        rings = []
        for i in range(len(parts)-1):
            ring = pts[parts[i]:parts[i+1]]
            if len(ring) >= 3:
                rings.append(ring)
        if not rings:
            continue
        poly = Polygon(rings[0], holes=rings[1:] if len(rings) > 1 else None)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)

    if not polys:
        raise RuntimeError("Keine Polygongeometrie gefunden.")

    geom = unary_union(polys)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)

    # CRS aus .prj
    wkt = prj_path.read_text(encoding="utf-8", errors="ignore")
    try:
        crs = CRS.from_wkt(wkt)
    except Exception:
        crs = CRS.from_string(wkt)

    return geom, crs


def to_metric_polygon(poly: Polygon, crs: CRS) -> Tuple[Polygon, CRS]:
    """Reprojiziere Polygon in metrische Projektion. Falls CRS schon metrisch, identisch."""
    if crs.is_projected and crs.axis_info and crs.axis_info[0].unit_name.lower().startswith("metre"):
        return poly, crs

    # Transform nach UTM (WGS84) anhand der Polygonmitte
    centroid = poly.centroid
    lon = centroid.x
    lat = centroid.y
    if not crs.is_geographic:
        to_geo = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = to_geo.transform(centroid.x, centroid.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    target = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs, target, always_xy=True)

    # Außenring transformieren
    x, y = poly.exterior.coords.xy
    xs, ys = transformer.transform(x, y)
    outer = list(zip(xs, ys))
    # Löcher transformieren
    holes = []
    for ring in poly.interiors:
        xh, yh = ring.coords.xy
        xsh, ysh = transformer.transform(xh, yh)
        holes.append(list(zip(xsh, ysh)))
    poly_m = Polygon(outer, holes=holes if holes else None)
    if not poly_m.is_valid:
        poly_m = poly_m.buffer(0)
    return poly_m, target


def compute_axis_bbox(poly: Polygon) -> Tuple[float, float, Tuple[float,float,float,float]]:
    minx, miny, maxx, maxy = poly.bounds
    return (maxx - minx), (maxy - miny), (minx, miny, maxx, maxy)


def generate_rect_grid_in_bbox(poly: Polygon, spacing: float, edge_offset: float) -> Tuple[List[Tuple[float,float]], int, int]:
    """Erzeuge ein vollständiges Nx×Ny-Raster, das komplett im *gebufferten* Polygon liegt."""
    work_poly = poly.buffer(-edge_offset) if edge_offset > 0 else poly
    if work_poly.is_empty:
        return [], 0, 0

    minx, miny, maxx, maxy = work_poly.bounds

    # Gitter-Kandidaten
    xs = []
    x0 = math.ceil(minx / spacing) * spacing
    xi = x0
    while xi <= maxx + 1e-9:
        xs.append(xi); xi += spacing

    ys = []
    y0 = math.ceil(miny / spacing) * spacing
    yi = y0
    while yi <= maxy + 1e-9:
        ys.append(yi); yi += spacing

    def all_inside(x_list, y_list) -> bool:
        for xv in x_list:
            for yv in y_list:
                if not work_poly.contains(Point(xv, yv)):
                    return False
        return True

    if not xs or not ys:
        return [], 0, 0

    left, right = 0, len(xs) - 1
    bottom, top = 0, len(ys) - 1
    changed = True
    while changed and left <= right and bottom <= top:
        changed = False
        while left <= right and not all_inside(xs[left:right+1], ys[bottom:top+1]):
            shrink_x = (right - left) >= (top - bottom)
            if shrink_x:
                if all_inside(xs[left+1:right+1], ys[bottom:top+1]):
                    left += 1; changed = True; break
                elif all_inside(xs[left:right], ys[bottom:top+1]):
                    right -= 1; changed = True; break
                else:
                    left += 1; right -= 1; changed = True
            else:
                if all_inside(xs[left:right+1], ys[bottom+1:top+1]):
                    bottom += 1; changed = True; break
                elif all_inside(xs[left:right+1], ys[bottom:top]):
                    top -= 1; changed = True; break
                else:
                    bottom += 1; top -= 1; changed = True

    if left > right or bottom > top:
        return [], 0, 0

    xs_ok = xs[left:right+1]
    ys_ok = ys[bottom:top+1]
    pts = [(xv, yv) for yv in ys_ok for xv in xs_ok]
    rows = len(ys_ok)
    cols = len(xs_ok)
    return pts, rows, cols


# --------- NEU: Punkte-ins-Polygon (mit Scan über Rotation & Phasen) ----------
def points_in_polygon_max(poly_m: Polygon, spacing: float, edge: float,
                          rotate_step_deg: float = 0.0, phase_steps: int = 1) -> List[Tuple[float,float]]:
    """
    Liefert die Punktmenge (x,y), die mit gegebenem 'spacing' ins Polygon passt,
    mit optionalem Scan über Rotation (0..180°) und Phasenverschiebungen in x/y.
    Gibt die beste Menge (max. Anzahl) zurück.
    """
    if phase_steps < 1: phase_steps = 1
    phase_vals = [i * spacing / phase_steps for i in range(phase_steps)] if phase_steps > 1 else [0.0]
    angles = [0.0]
    if rotate_step_deg and rotate_step_deg > 0.0:
        n = int(180.0 / rotate_step_deg)
        angles = [i * rotate_step_deg for i in range(n)]

    best_pts = []
    work_outer = poly_m.buffer(-edge) if edge > 0 else poly_m
    if work_outer.is_empty:
        return best_pts

    cx, cy = (work_outer.centroid.x, work_outer.centroid.y)

    for ang in angles:
        # Polygon drehen (um Zentrum), dann achsenparalleles Grid darauf
        poly_r = affinity.rotate(work_outer, ang, origin=(cx, cy), use_radians=False)
        minx, miny, maxx, maxy = poly_r.bounds

        for phx in phase_vals:
            for phy in phase_vals:
                # Startwerte so wählen, dass wir die Phasenlage berücksichtigen
                x0 = math.floor((minx - phx) / spacing) * spacing + phx
                y0 = math.floor((miny - phy) / spacing) * spacing + phy

                xs = []
                xi = x0
                while xi <= maxx + 1e-9:
                    xs.append(xi); xi += spacing

                ys = []
                yi = y0
                while yi <= maxy + 1e-9:
                    ys.append(yi); yi += spacing

                pts_r = [(x, y) for y in ys for x in xs if poly_r.contains(Point(x, y))]
                if len(pts_r) > len(best_pts):
                    # zurückrotieren in Originalkoordinaten
                    if ang != 0.0:
                        pts_orig = [affinity.rotate(Point(x, y), -ang, origin=(cx, cy), use_radians=False).coords[0] for (x, y) in pts_r]
                    else:
                        pts_orig = pts_r
                    best_pts = pts_orig
    return best_pts


# -----------------------------
# U-Shape aus rechteckigem Raster
# -----------------------------
def apply_u_shape(points: List[Tuple[float,float]], rows: int, cols: int, spacing: float, u_gap_m: float) -> List[Tuple[float,float]]:
    """Erzeuge eine U-Form, indem in der Mitte ein Korridor freigelassen wird.
    Annahme: Punkte sind in Zeilenreihenfolge (y wächst, dann x).
    """
    if rows == 0 or cols == 0 or not points:
        return points

    short_dir_cols = cols <= rows  # True -> kurze Richtung ist x (Spalten)

    if short_dir_cols:
        gap_cols = max(1, int(round(u_gap_m / max(1e-9, spacing))))
        mid = cols // 2
        left = max(0, mid - gap_cols // 2)
        right = min(cols - 1, left + gap_cols - 1)
        kept = []
        for r in range(rows):
            for c in range(cols):
                if left <= c <= right:
                    continue
                kept.append(points[r*cols + c])
        return kept
    else:
        gap_rows = max(1, int(round(u_gap_m / max(1e-9, spacing))))
        mid = rows // 2
        low = max(0, mid - gap_rows // 2)
        high = min(rows - 1, low + gap_rows - 1)
        kept = []
        for r in range(rows):
            if low <= r <= high:
                continue
            for c in range(cols):
                kept.append(points[r*cols + c])
        return kept


# -----------------------------
# rows×cols wählen (für Layout mit N)
# -----------------------------
def choose_rows_cols(N: int, aspect_bbox: float) -> Tuple[int,int]:
    """Finde (rows, cols) mit rows*cols >= N und Verhältnis cols/rows nahe aspect_bbox."""
    if N <= 0: return 0,0
    best = None
    for r in range(1, N+1):
        c_min = math.ceil(N / r)
        ratio = (c_min / r)
        score = abs(math.log((ratio+1e-9)/(aspect_bbox+1e-9))) + 0.01*(r*c_min - N)
        tup = (score, r, c_min)
        if (best is None) or (tup < best): best = tup
        if r*c_min == N and abs(math.log((ratio+1e-9)/(aspect_bbox+1e-9))) < 0.02:
            break
    _, rows, cols = best
    return rows, cols


# -----------------------------
# Rechteck/U-Layout-Punkte erzeugen (frei platzierbar, zentriert)
# -----------------------------
def layout_points_centered(N: int, spacing: float, center_xy: Tuple[float,float],
                           aspect_hint: float, layout_type: str, u_gap_m: float) -> Tuple[List[Tuple[float,float]], int, int]:
    rows, cols = choose_rows_cols(N, aspect_hint)
    cx, cy = center_xy

    # zentriertes Rechteckgitter
    width  = (cols - 1) * spacing
    height = (rows - 1) * spacing
    x0 = cx - width/2.0
    y0 = cy - height/2.0

    pts_rect = []
    for r in range(rows):
        for c in range(cols):
            pts_rect.append((x0 + c*spacing, y0 + r*spacing))

    pts = pts_rect
    if layout_type.lower().startswith("u"):
        pts = apply_u_shape(pts_rect, rows, cols, spacing, u_gap_m)
        if len(pts) < N:
            missing = N - len(pts)
            keep_set = set(pts)
            deleted = [p for p in pts_rect if p not in keep_set]
            pts += deleted[:missing]

    return pts[:N], rows, cols


# -----------------------------
# Quick-Check Helpers: Spacing Scan
# -----------------------------

def _read_hourly_csv_generic(path: Path):
    import pandas as pd
    seps = [';', ',', '\t']
    decs = [',', '.']
    last_err = None
    for sep in seps:
        for dec in decs:
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec)
                cols = {str(c).lower().strip(): c for c in df.columns}
                if 'hour' in cols and 'q_extraction_k_w' in cols:
                    cols['q_extraction_kw'] = cols.pop('q_extraction_k_w')
                if 'hour' in cols and 'q_extraction_kw' in cols and 'q_injection_kw' in cols:
                    df = df.sort_values(by=[cols['hour']]).reset_index(drop=True)
                    q_ex = pd.to_numeric(df[cols['q_extraction_kw']], errors='coerce').fillna(0.0).to_numpy(float)
                    q_in = pd.to_numeric(df[cols['q_injection_kw']], errors='coerce').fillna(0.0).to_numpy(float)
                    return q_ex, q_in
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"CSV-Format nicht erkannt: {path} | letzter Fehler: {last_err}")

def _convert_building_to_ground_arrays(q_heat_kW, q_cool_kW, scop: float, seer: float, use_injection: bool):
    import numpy as np
    q_ex = np.asarray(q_heat_kW, dtype=float)
    if scop and scop > 0:
        fac = max(0.0, 1.0 - 1.0/float(scop))
        q_ex = q_ex * fac
    q_in = np.asarray(q_cool_kW, dtype=float) if use_injection else np.zeros_like(q_ex)
    if use_injection and seer and seer > 0:
        fac = 1.0 + 1.0/float(seer)
        q_in = q_in * fac
    return q_ex, q_in

def _size_factor_for_Hfix(nx:int, ny:int, S:float, H_fix:float, D:float, r_b:float,
                          q_ex_base, q_in_base, years:int, k_s:float, T_g:float,
                          Tmin:float, Tmax:float, Rb_const:float):
    from GHEtool import Borefield, GroundConstantTemperature
    from GHEtool.VariableClasses.LoadData.GeothermalLoad.HourlyGeothermalLoad import HourlyGeothermalLoad
    import numpy as np
    ground = GroundConstantTemperature(float(k_s), float(T_g))
    # Wichtig: Borefield.__init__ akzeptiert keinen Parameter 'simulation_period'.
    # Die Simulationsdauer wird im Load-Objekt (HourlyGeothermalLoad) gesetzt.
    bf = Borefield(ground_data=ground)
    try:
        bf.create_rectangular_borefield(int(nx), int(ny), float(S), float(S), float(H_fix), float(D), float(r_b))
    except Exception:
        bf.create_rectangular_borefield(int(nx), int(ny), float(S), float(S), float(H_fix))
    if hasattr(bf, 'set_min_avg_fluid_temperature'):
        bf.set_min_avg_fluid_temperature(float(Tmin)); bf.set_max_avg_fluid_temperature(float(Tmax))
    else:
        bf.set_min_ground_temperature(float(Tmin)); bf.set_max_ground_temperature(float(Tmax))
    try:
        bf.calculation_setup(use_constant_Rb=True)
    except Exception:
        pass
    try:
        bf.Rb = float(Rb_const)
    except Exception:
        pass

    def Hreq_for_factor(f: float) -> float:
        load = HourlyGeothermalLoad(extraction_load=(q_ex_base*f), injection_load=(q_in_base*f), simulation_period=int(years))
        try:
            bf.set_load(load)
        except Exception:
            bf.load = load
        return float(bf.size(100.0, L4_sizing=True))

    f_lo = 0.0
    f_hi = 1.0
    tries = 0
    while tries < 12:
        Hreq = Hreq_for_factor(f_hi)
        if Hreq > H_fix * (1.0 + 1e-6):
            break
        f_hi *= 2.0
        tries += 1
    if tries >= 12:
        return f_hi, Hreq_for_factor(f_hi)
    for _ in range(22):
        f_mid = 0.5*(f_lo+f_hi)
        Hreq = Hreq_for_factor(f_mid)
        if Hreq > H_fix:
            f_hi = f_mid
        else:
            f_lo = f_mid
        if abs(f_hi - f_lo)/max(1e-12,f_mid) < 1e-3:
            break
    return f_lo, Hreq_for_factor(f_lo)

def _scan_spacing_on_polygon(shp_path: Path, S_values, H_fix: float, edge_offset: float,
                              q_ex_base, q_in_base, years:int, k_s:float, T_g:float,
                              Tmin:float, Tmax:float, Rb_const:float, layout_type:str='rectangle'):
    poly, crs = read_polygon_from_shp(shp_path)
    poly_m, _ = to_metric_polygon(poly, crs)
    w, h, _ = compute_axis_bbox(poly_m)
    aspect = (w/h) if h>0 else 1.0
    results = []
    for S in S_values:
        pts = points_in_polygon_max(poly_m, float(S), float(edge_offset), rotate_step_deg=5.0, phase_steps=2)
        N = len(pts)
        if N <= 0:
            results.append({'S':float(S),'N':0,'E_MWh':0.0,'eta_kWh_per_m':0.0,'L_total_m':0.0,'nx':0,'ny':0}); continue
        nx, ny = choose_rows_cols(N, aspect)
        # D und r aus Data_Base versuchen
        D_default, r_default = 1.5, 0.075
        try:
            import Data_Base as bd
            D_default = float(getattr(bd,'D', D_default)); r_default=float(getattr(bd,'r', r_default))
        except Exception:
            pass
        f, Hreq = _size_factor_for_Hfix(nx, ny, float(S), float(H_fix), float(D_default), float(r_default),
                                        q_ex_base, q_in_base, int(years), float(k_s), float(T_g), float(Tmin), float(Tmax), float(Rb_const))
        import numpy as np
        E_kWh = float(np.sum(q_ex_base)*f)
        E_MWh = E_kWh/1000.0
        L_total = N*float(H_fix)
        eta = (E_kWh / max(1.0, L_total))
        results.append({'S':float(S),'N':int(N),'nx':int(nx),'ny':int(ny),'E_MWh':E_MWh,'eta_kWh_per_m':eta,'L_total_m':L_total,'factor':f})
    return results

def _spacing_scan_plot(results, S_current: float|None, knee_method: str = 'eta95'):
    import matplotlib.pyplot as plt
    import numpy as np
    S = np.array([r['S'] for r in results], dtype=float)
    E = np.array([r['E_MWh'] for r in results], dtype=float)
    eta = np.array([r['eta_kWh_per_m'] for r in results], dtype=float)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    ax1.plot(S, E, '-o', color='tab:blue', lw=1.6, label='E_max [MWh/a]')
    ax1.set_xlabel('Sondenabstand S [m]'); ax1.set_ylabel('E_max [MWh/a]')
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(S, eta, '-s', color='tab:gray', lw=1.0, alpha=0.8, label='η [kWh/(m·a)]')
    ax2.set_ylabel('η [kWh/(m·a)]')
    knee_x = None
    if knee_method=='eta95' and eta.size>0:
        th = 0.95*np.max(eta)
        for s_val, e_val in zip(S, eta):
            if e_val >= th:
                knee_x = float(s_val); break
    if S_current and float(S_current)>0:
        ax1.axvline(float(S_current), color='orange', linestyle='--', alpha=0.8, label=f'S aktuell = {float(S_current):.2f} m')
    if knee_x:
        ax1.axvline(knee_x, color='green', linestyle=':', alpha=0.8, label=f'Knie ≈ {knee_x:.2f} m (η≥95%)')
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='best')
    fig.tight_layout(); plt.show()

def _spacing_scan_dialog(app_self):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import numpy as np
    top = tk.Toplevel(app_self)
    top.title('Optimalen Abstand finden – Quick‑Check')
    top.geometry('780x520')
    vars = {
        'shp': tk.StringVar(value=str(Path(app_self.shp_dir.get()).resolve())),
        'csv': tk.StringVar(value=''),
        'is_build': tk.BooleanVar(value=False),
        'scop': tk.DoubleVar(value=0.0),
        'seer': tk.DoubleVar(value=0.0),
        'use_inj': tk.BooleanVar(value=True),
        'k_s': tk.DoubleVar(value=2.0),
        'Tg': tk.DoubleVar(value=10.0),
        'Rb': tk.DoubleVar(value=0.20),
        'Hfix': tk.DoubleVar(value=float(app_self.depth.get())),
        'Tmin': tk.DoubleVar(value=0.0),
        'Tmax': tk.DoubleVar(value=16.0),
        'years': tk.IntVar(value=50),
        'Smin': tk.DoubleVar(value=5.0),
        'Smax': tk.DoubleVar(value=20.0),
        'npts': tk.IntVar(value=10),
        'edge': tk.DoubleVar(value=float(app_self.edge.get())),
        'export': tk.BooleanVar(value=False),
    }
    try:
        import Data_Base as bd
        vars['k_s'].set(float(getattr(bd,'k_s', getattr(bd,'conductivity',2.0))))
        vars['Tg'].set(float(getattr(bd,'T_g', 10.0)))
    except Exception:
        pass
    frm = ttk.Frame(top); frm.pack(fill='both', expand=True, padx=10, pady=10)
    r=0
    ttk.Label(frm, text='Shapefile oder Ordner:').grid(row=r, column=0, sticky='e'); ent_shp=ttk.Entry(frm,textvariable=vars['shp'], width=54); ent_shp.grid(row=r, column=1, sticky='we'); ttk.Button(frm,text='…', command=lambda: vars['shp'].set(filedialog.askopenfilename(title='SHP wählen', filetypes=[('Shapefile','*.shp')], parent=top) or vars['shp'].get())).grid(row=r, column=2); r+=1
    ttk.Label(frm, text='CSV (hourly):').grid(row=r, column=0, sticky='e'); ttk.Entry(frm,textvariable=vars['csv'], width=54).grid(row=r, column=1, sticky='we'); ttk.Button(frm,text='…', command=lambda: vars['csv'].set(filedialog.askopenfilename(title='CSV wählen', filetypes=[('CSV','*.csv'),('Alle','*.*')], parent=top) or vars['csv'].get())).grid(row=r, column=2); r+=1
    ttk.Checkbutton(frm, text='CSV ist Gebäudelast', variable=vars['is_build']).grid(row=r, column=1, sticky='w'); r+=1
    ttk.Label(frm, text='SCOP:').grid(row=r, column=0, sticky='e'); ttk.Entry(frm,textvariable=vars['scop'], width=8).grid(row=r, column=1, sticky='w'); ttk.Label(frm, text='SEER:').grid(row=r, column=1, sticky='e', padx=(120,0)); ttk.Entry(frm,textvariable=vars['seer'], width=8).grid(row=r, column=1, sticky='e', padx=(180,0)); ttk.Checkbutton(frm, text='Rückspeisung verwenden', variable=vars['use_inj']).grid(row=r, column=2, sticky='w'); r+=1
    ttk.Label(frm, text='λ [W/mK]:').grid(row=r, column=0, sticky='e'); ttk.Entry(frm,textvariable=vars['k_s'], width=8).grid(row=r, column=1, sticky='w'); ttk.Label(frm, text='Tg [°C]:').grid(row=r, column=1, sticky='e', padx=(120,0)); ttk.Entry(frm,textvariable=vars['Tg'], width=8).grid(row=r, column=1, sticky='e', padx=(180,0)); r+=1
    ttk.Label(frm, text='Rb* [mK/W]:').grid(row=r, column=0, sticky='e'); ttk.Entry(frm,textvariable=vars['Rb'], width=8).grid(row=r, column=1, sticky='w'); ttk.Label(frm, text='H_fix [m]:').grid(row=r, column=1, sticky='e', padx=(120,0)); ttk.Entry(frm,textvariable=vars['Hfix'], width=8).grid(row=r, column=1, sticky='e', padx=(180,0)); r+=1
    ttk.Label(frm, text='Grenzen Tmin/Tmax [°C]:').grid(row=r, column=0, sticky='e');
    lim = ttk.Frame(frm); lim.grid(row=r, column=1, sticky='w')
    ttk.Entry(lim, textvariable=vars['Tmin'], width=8).pack(side='left'); ttk.Label(lim, text='/').pack(side='left'); ttk.Entry(lim, textvariable=vars['Tmax'], width=8).pack(side='left'); r+=1
    ttk.Label(frm, text='S_min / S_max / Punkte:').grid(row=r, column=0, sticky='e'); rng = ttk.Frame(frm); rng.grid(row=r, column=1, sticky='w')
    ttk.Entry(rng, textvariable=vars['Smin'], width=8).pack(side='left'); ttk.Label(rng,text=' / ').pack(side='left'); ttk.Entry(rng, textvariable=vars['Smax'], width=8).pack(side='left'); ttk.Label(rng,text=' / ').pack(side='left'); ttk.Entry(rng, textvariable=vars['npts'], width=6).pack(side='left'); r+=1
    ttk.Checkbutton(frm, text='CSV/JSON Export', variable=vars['export']).grid(row=r, column=1, sticky='w'); r+=1
    prog = ttk.Progressbar(frm, orient='horizontal', mode='determinate', maximum=100); prog.grid(row=r, column=0, columnspan=3, sticky='we', pady=(6,6)); r+=1
    out = tk.Text(frm, height=8); out.grid(row=r, column=0, columnspan=3, sticky='nsew'); frm.rowconfigure(r, weight=1); r+=1
    btn = ttk.Frame(frm); btn.grid(row=r, column=0, columnspan=3, pady=(8,0))
    def log(msg:str): out.insert('end', msg+'\n'); out.see('end')
    def run_scan():
        try:
            shp_sel = vars['shp'].get().strip()
            p = Path(shp_sel)
            if p.exists() and p.is_dir():
                cand = next((q for q in sorted(p.glob('*.shp')) if q.is_file()), None)
                if not cand: raise FileNotFoundError(f'Keine SHP in {p}')
                p = cand
            shp_path = p.resolve()
            csv_path = Path(vars['csv'].get().strip()).resolve()
            if not csv_path.exists(): raise FileNotFoundError('CSV nicht gefunden.')
            q_ex0, q_in0 = _read_hourly_csv_generic(csv_path)
            if bool(vars['is_build'].get()):
                q_ex0, q_in0 = _convert_building_to_ground_arrays(q_ex0, q_in0, float(vars['scop'].get()), float(vars['seer'].get()), bool(vars['use_inj'].get()))
            else:
                if not bool(vars['use_inj'].get()):
                    # 'np' ist im äußeren Scope (_spacing_scan_dialog) importiert – nicht erneut importieren,
                    # um Python's "unbound local"-Fall zu vermeiden.
                    q_in0 = np.zeros_like(q_ex0)
            Smin, Smax, npts = float(vars['Smin'].get()), float(vars['Smax'].get()), int(vars['npts'].get())
            S_values = list(np.linspace(Smin, Smax, npts))
            log(f'Scan {len(S_values)} Abstände von {Smin:.2f} bis {Smax:.2f} m …')
            prog['value']=0; top.update_idletasks()
            res = _scan_spacing_on_polygon(shp_path, S_values, float(vars['Hfix'].get()), float(vars['edge'].get()),
                                           q_ex0, q_in0, int(vars['years'].get()), float(vars['k_s'].get()), float(vars['Tg'].get()),
                                           float(vars['Tmin'].get()), float(vars['Tmax'].get()), float(vars['Rb'].get()))
            prog['value']=100; top.update_idletasks()
            for rrow in res:
                log(f"S={rrow['S']:.2f} m | N={rrow['N']} | E={rrow['E_MWh']:.2f} MWh/a | η={rrow['eta_kWh_per_m']:.1f} kWh/(m·a)")
            try:
                _spacing_scan_plot(res, float(app_self.spacing.get()))
            except Exception:
                pass
            if bool(vars['export'].get()):
                out_dir = _artifact_dir(Path(app_self.project_dir.get()).resolve())
                csv_out = out_dir / 'spacing_scan_results.csv'
                with open(csv_out,'w',encoding='utf-8') as f:
                    f.write('S_m;N;nx;ny;E_MWh;eta_kWh_per_m;L_total_m;factor\n')
                    for rrow in res:
                        f.write(f"{rrow['S']:.3f};{rrow['N']};{rrow['nx']};{rrow['ny']};{rrow['E_MWh']:.3f};{rrow['eta_kWh_per_m']:.3f};{rrow['L_total_m']:.1f};{rrow['factor']:.5f}\n")
                import json
                json_out = out_dir / 'spacing_scan_results.json'
                meta = dict(hfix=float(vars['Hfix'].get()), Tmin=float(vars['Tmin'].get()), Tmax=float(vars['Tmax'].get()), years=int(vars['years'].get()), k_s=float(vars['k_s'].get()), Tg=float(vars['Tg'].get()), Rb=float(vars['Rb'].get()), csv=str(csv_path), shp=str(shp_path))
                with open(json_out,'w',encoding='utf-8') as f:
                    json.dump({'results':res,'meta':meta}, f, indent=2, ensure_ascii=False)
                log(f'Export: {csv_out} + {json_out}')
        except Exception as e:
            messagebox.showerror('Fehler', str(e))
    ttk.Button(btn, text='Scan starten', command=run_scan).pack(side='left', padx=6)
    ttk.Button(btn, text='Schließen', command=top.destroy).pack(side='left', padx=6)


def write_csv_and_meta(out_dir: Path, points_polygon: List[Tuple[float,float]], points_layout: List[Tuple[float,float]],
                       res: FieldResult, params: LayoutParams) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zwei CSVs: polygon_points & layout_points
    csv_poly = out_dir / "borefield_polygon_points.csv"
    with open(csv_poly, "w", encoding="utf-8") as f:
        f.write("bh_id,x_m,y_m\n")
        for i, (x, y) in enumerate(points_polygon, start=1):
            f.write(f"{i},{x:.3f},{y:.3f}\n")

    csv_layout = out_dir / "borefield_layout_points.csv"
    with open(csv_layout, "w", encoding="utf-8") as f:
        f.write("bh_id,x_m,y_m\n")
        for i, (x, y) in enumerate(points_layout, start=1):
            f.write(f"{i},{x:.3f},{y:.3f}\n")

    meta = {
        "timestamp": _now_stamp(),
        "mode": params.mode,
        "layout_type": params.layout_type,
        "spacing_m": params.spacing_m, "depth_m": params.depth_m,
        "edge_offset_m": params.edge_offset_m, "u_gap_m": params.u_gap_m,
        "rotate_step_deg": params.rotate_step_deg, "phase_steps": params.phase_steps,
        "rows": res.rows, "cols": res.cols, "N": res.count,
        "bbox_w_m": res.bbox_w_m, "bbox_h_m": res.bbox_h_m,
        "crs_epsg": res.crs_epsg, "shp_used": res.shp_used,
        "extra": res.meta_extra,
        "notes": "polygon_points = maximale Punkte im Polygon; layout_points = standardisiertes Rechteck/U mit N=polygon_points.count"
    }
    meta_path = out_dir / "borefield_layout.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return csv_poly, csv_layout, meta_path


def preview_png(out_dir: Path, poly, points_polygon: List[Tuple[float,float]], points_layout: List[Tuple[float,float]]):
    if not _HAS_PLT:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,4), dpi=140)
    try:
        x, y = poly.exterior.xy
        ax.plot(x, y, '-', lw=1.2, alpha=0.9, label="Polygon")
        for ring in poly.interiors:
            xi, yi = ring.xy
            ax.plot(xi, yi, '-', lw=1.0, alpha=0.6)
    except Exception:
        pass
    if points_polygon:
        xs = [p[0] for p in points_polygon]
        ys = [p[1] for p in points_polygon]
        ax.scatter(xs, ys, s=10, alpha=0.9, label="Boreholes (im Polygon)", marker='o')
    if points_layout:
        xs2 = [p[0] for p in points_layout]
        ys2 = [p[1] for p in points_layout]
        ax.scatter(xs2, ys2, s=16, alpha=0.9, label="Rechteck/U (N gleich)", marker='o')
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / "borefield_preview.png")
    plt.close(fig)


# -----------------------------
# Projekt-Dateien patchen
# -----------------------------

_ASSIGN_PATTERNS = {
    "H": r"^\s*H\s*=\s*([0-9eE\.\,\+\-]+)",
    "Bx": r"^\s*Bx\s*=\s*([0-9eE\.\,\+\-]+)",
    "By": r"^\s*By\s*=\s*([0-9eE\.\,\+\-]+)",
    "rows_sonden": r"^\s*rows_sonden\s*=\s*([0-9]+)",
    "Columns_sonden": r"^\s*Columns_sonden\s*=\s*([0-9]+)",
}

def _replace_assignments(text: str, updates: dict) -> Tuple[str, List[str]]:
    """Ersetze einfache Zuweisungen in Python-Quelltext. Gibt (neuer_text, liste_geaenderter_keys) zurück."""
    changed = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for key, pat in _ASSIGN_PATTERNS.items():
            if key not in updates:
                continue
            if re.match(pat, line):
                val = updates[key]
                if isinstance(val, float):
                    rep = f"{key} = {val:.6g}"
                else:
                    rep = f"{key} = {val}"
                lines[i] = rep
                if key not in changed:
                    changed.append(key)
    new_text = "\n".join(lines)
    return new_text, changed


def patch_project_files(project_dir: Path, rows: int, cols: int, spacing: float, depth: float, overwrite: bool = False) -> List[Path]:
    """Sichert Backups und aktualisiert Data_Base.py und (falls vorhanden) borefields.py."""
    modified_files = []
    ts = _now_stamp()
    backup_dir = _artifact_dir(project_dir)

    targets = ["Data_Base.py", "borefields.py"]
    updates = {
        "H": float(depth),
        "Bx": float(spacing),
        "By": float(spacing),
        "rows_sonden": int(rows),
        "Columns_sonden": int(cols),
    }

    for fname in targets:
        fpath = project_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        new_text, changed_keys = _replace_assignments(text, updates)

        if changed_keys:
            bk = backup_dir / f"{fname}.backup.{ts}"
            fpath.replace(bk)
            modified_files.append(bk)
            (project_dir / fname).write_text(new_text, encoding="utf-8")
            modified_files.append(fpath)
        else:
            if overwrite:
                bk = backup_dir / f"{fname}.backup.{ts}"
                fpath.replace(bk)
                modified_files.append(bk)
                appendix = "\n\n# --- auto_borefield.py: AUTOGENERIERTE PARAMETER ---\n"
                appendix += f"H = {updates['H']:.6g}\nBx = {updates['Bx']:.6g}\nBy = {updates['By']:.6g}\n"
                appendix += f"rows_sonden = {updates['rows_sonden']}\nColumns_sonden = {updates['Columns_sonden']}\n"
                (project_dir / fname).write_text(text + appendix, encoding="utf-8")
                modified_files.append(fpath)

    return modified_files


def restore_latest_backup(project_dir: Path) -> List[Path]:
    """Stellt den zuletzt erzeugten Backup-Stand wieder her (Data_Base.py/borefields.py)."""
    restored = []
    for base in ["Data_Base.py", "borefields.py"]:
        backup_dir = _artifact_dir(project_dir)
        backups = sorted(backup_dir.glob(f"{base}.backup.*"))
        if not backups:
            continue
        latest = backups[-1]
        orig = project_dir / base
        if orig.exists():
            ts = _now_stamp()
            orig.replace(project_dir / f"{base}.replaced.{ts}")
        latest.replace(orig)
        restored.append(orig)
    return restored


# -----------------------------
# Hauptlogik
# -----------------------------

def compute_borefield(shp_dir: Path, params: LayoutParams, project_dir: Path) -> FieldResult:
    # SHP finden
    candidates = find_shp_in_folder(shp_dir)
    if not candidates:
        raise FileNotFoundError(f"Kein .shp in {shp_dir} gefunden.")
    shp_path = None
    if params.shp_name:
        for c in candidates:
            if c.stem == params.shp_name:
                shp_path = c; break
        if shp_path is None:
            raise FileNotFoundError(f"{params.shp_name}.shp nicht in {shp_dir} gefunden.")
    else:
        shp_path = candidates[0]

    # Polygon laden + metrisch
    poly, crs = read_polygon_from_shp(shp_path)
    poly_m, crs_m = to_metric_polygon(poly, crs)
    bbox_w, bbox_h, _ = compute_axis_bbox(poly_m)
    aspect = (bbox_w / bbox_h) if bbox_h > 0 else 1.0
    centroid = (poly_m.centroid.x, poly_m.centroid.y)

    points_poly = []
    rows = cols = 0
    points_layout = []

    if params.mode == "legacy":
        # altes Verhalten: ein volles Nx×Ny-Raster innerhalb der Fläche
        points_rect, rows, cols = generate_rect_grid_in_bbox(poly_m, params.spacing_m, params.edge_offset_m)
        if rows == 0 or cols == 0:
            raise RuntimeError("Kein passendes Raster innerhalb der Fläche gefunden. Spacing/Offset prüfen.")
        # ggf. U-Shape anwenden
        points = points_rect
        if params.layout_type.lower().startswith("u"):
            points = apply_u_shape(points_rect, rows, cols, params.spacing_m, params.u_gap_m)
        points_poly = points
        N = len(points)
        # Standardisiertes Layout = identisch (legacy)
        points_layout = points[:]
    else:
        # NEU: fitmax – maximale Punktezahl im Polygon ermitteln
        points_poly = points_in_polygon_max(poly_m, params.spacing_m, params.edge_offset_m,
                                            rotate_step_deg=params.rotate_step_deg, phase_steps=params.phase_steps)
        N = len(points_poly)

        if N == 0:
            raise RuntimeError("Es passen keine Bohrpunkte in das Polygon (prüfe Spacing/Randabstand).")

        if params.layout_type == 'freeform':
            # Kein standardisiertes Layout – Freiformpunkte werden für Longterm genutzt
            points_layout = []
            rows = cols = 0
        else:
            # Rechteck-/U-Layout mit gleicher N und gleichem Spacing, zentriert
            points_layout, rows, cols = layout_points_centered(
                N=N, spacing=params.spacing_m, center_xy=centroid,
                aspect_hint=aspect, layout_type=params.layout_type, u_gap_m=params.u_gap_m
            )

    count = N
    res = FieldResult(
        rows=rows, cols=cols, count=count,
        bbox_w_m=bbox_w, bbox_h_m=bbox_h,
        points_polygon=points_poly, points_layout=points_layout,
        crs_epsg=(crs_m.to_epsg() if crs_m else None),
        shp_used=shp_path.name,
        meta_extra={"N_in_polygon": N, "mode": params.mode}
    )

    out_dir = _artifact_dir(project_dir)
    write_csv_and_meta(out_dir, points_poly, points_layout, res, params)
    preview_png(out_dir, poly_m, points_poly, points_layout)

    if params.overwrite:
        patch_project_files(project_dir, rows, cols, params.spacing_m, params.depth_m, overwrite=True)

    return res


# -----------------------------
# CLI
# -----------------------------

def cli_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Automatischer Bohrfeld-Generator (Shapefile → Projektwerte).")
    p.add_argument("--mode", choices=["fitmax","legacy"], default="fitmax", help="fitmax: N aus Fläche ermitteln und als Rechteck/U übernehmen; legacy: altes Verhalten")
    p.add_argument("--layout", choices=["rectangle","ushaped"], default="rectangle", help="Layout-Typ für das standardisierte Feld")
    p.add_argument("--spacing", type=float, default=6.0, help="Mindestabstand (m) – Bx=By")
    p.add_argument("--depth", type=float, default=200.0, help="Bohrtiefe H (m)")
    p.add_argument("--edge", type=float, default=1.0, help="Sicherheitsabstand zum Polygonrand (m)")
    p.add_argument("--u-gap", type=float, default=12.0, help="U-Form: Breite des Innenhofs (m)")
    p.add_argument("--overwrite", action="store_true", help="Projektdateien sofort überschreiben (Backup wird angelegt)")
    p.add_argument("--shp-name", type=str, default=None, help="SHP-Dateiname ohne Endung (falls mehrere vorhanden)")
    p.add_argument("--restore", action="store_true", help="Letzten Backup-Stand wiederherstellen und beenden")
    p.add_argument("--project-dir", type=str, default=".", help="Projektordner (mit Data_Base.py/borefields.py)")
    p.add_argument("--shp-dir", type=str, default="./shp", help="Ordner, der das Shapefile enthält")
    # Scan-Optionen (nur fitmax)
    p.add_argument("--rotate-step", type=float, default=5.0, help="Rotations-Scan [°] (0=aus)")
    p.add_argument("--phase-steps", type=int, default=3, help="Phasen-Scan (Anzahl Versätze pro Achse; 1=aus)")

    args = p.parse_args(argv)

    project_dir = Path(args.project_dir).resolve()
    shp_dir = Path(args.shp_dir).resolve()

    if args.restore:
        restored = restore_latest_backup(project_dir)
        if restored:
            print("Wiederhergestellt:", ", ".join(str(p) for p in restored))
            return 0
        else:
            print("Keine Backups gefunden.")
            return 1

    params = LayoutParams(
        mode=args.mode,
        layout_type=args.layout,
        spacing_m=args.spacing,
        depth_m=args.depth,
        edge_offset_m=args.edge,
        u_gap_m=args.u_gap,
        overwrite=args.overwrite,
        shp_name=args.shp_name,
        rotate_step_deg=args.rotate_step,
        phase_steps=args.phase_steps,
    )

    res = compute_borefield(shp_dir, params, project_dir)

    print("\nErgebnis")
    print("========")
    print(f"Mode:        {params.mode}")
    print(f"Layout:      {params.layout_type}")
    print(f"Spacing:     {params.spacing_m:.2f} m   |   Tiefe H: {params.depth_m:.2f} m")
    print(f"Rect/U-Layout: rows × cols = {res.rows} × {res.cols}   →   N = {res.count}")
    print(f"Polygon-BBox: {res.bbox_w_m:.1f} m × {res.bbox_h_m:.1f} m  |  CRS: EPSG:{res.crs_epsg}")
    print(f"SHP-Datei:   {res.shp_used}")
    print(f"N (im Polygon möglich): {res.meta_extra.get('N_in_polygon')}")
    if params.overwrite:
        print("Projektdateien wurden überschrieben (Backups *.backup.<ts>).")
    else:
        print("Projektdateien wurden NICHT überschrieben (CSV/Meta/Preview wurden abgelegt).")
    print("Dateien im Unterordner 'autoborehole':")
    print("  - borefield_polygon_points.csv  (tatsächliche Punkte im Polygon)")
    print("  - borefield_layout_points.csv   (Rechteck/U mit gleicher N)")
    print("  - borefield_layout.meta.json")
    print("  - borefield_preview.png (Overlay-Plot)")
    return 0


# -----------------------------
# Tkinter-GUI
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Borefield – Generator (fitmax)")
        self.geometry("760x600")
        self.resizable(False, False)

        self.project_dir = tk.StringVar(value=str(Path(".").resolve()))
        self.shp_dir = tk.StringVar(value=str((Path(".") / "shp").resolve()))
        self.mode = tk.StringVar(value="fitmax")
        self.layout = tk.StringVar(value="rectangle")
        self.spacing = tk.DoubleVar(value=6.0)
        self.depth = tk.DoubleVar(value=200.0)
        self.edge = tk.DoubleVar(value=1.0)
        self.u_gap = tk.DoubleVar(value=12.0)
        self.rotate_step = tk.DoubleVar(value=5.0)
        self.phase_steps = tk.IntVar(value=3)
        self.overwrite = tk.BooleanVar(value=False)

        self._build()

    def _build(self):
        pad = {'padx':10, 'pady':6}
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        r=0
        ttk.Label(frm, text="Projektordner:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.project_dir, width=48).grid(row=r, column=1, **pad); r+=1

        ttk.Label(frm, text="SHP-Ordner:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.shp_dir, width=48).grid(row=r, column=1, **pad); r+=1

        ttk.Label(frm, text="Modus:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm, text="fitmax (empfohlen)", variable=self.mode, value="fitmax").grid(row=r, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm, text="legacy", variable=self.mode, value="legacy").grid(row=r, column=1, sticky="e", **pad); r+=1

        ttk.Label(frm, text="Layout:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm, text="Rectangle", variable=self.layout, value="rectangle").grid(row=r, column=1, sticky="w", **pad)
        ttk.Radiobutton(frm, text="U-shaped",  variable=self.layout, value="ushaped").grid(row=r, column=1, sticky="e", **pad); r+=1
        ttk.Radiobutton(frm, text="Freiform (nur Longterm)",  variable=self.layout, value="freeform").grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="Abstand S (m):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.spacing, width=12).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="Tiefe H (m):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.depth, width=12).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="Randabstand (m):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.edge, width=12).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="U-Gap (m):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.u_gap, width=12).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="Rotations-Scan [°]:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.rotate_step, width=12).grid(row=r, column=1, sticky="w", **pad); r+=1

        ttk.Label(frm, text="Phasen-Scan (Schritte):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.phase_steps, width=12).grid(row=r, column=1, **pad); r+=1

        ttk.Checkbutton(frm, text="Projektwerte überschreiben (Backup wird angelegt)", variable=self.overwrite).grid(row=r, column=0, columnspan=2, sticky="w", **pad); r+=1

        btnfrm = ttk.Frame(frm); btnfrm.grid(row=r, column=0, columnspan=2, pady=(18,0))
        ttk.Button(btnfrm, text="Borefield generieren", command=self.run_generate).grid(row=0, column=0, padx=10)
        ttk.Button(btnfrm, text="Optimalen Abstand finden", command=self.run_spacing_scan_dialog).grid(row=0, column=1, padx=10)
        ttk.Button(btnfrm, text="Letzten Backup wiederherstellen", command=self.run_restore).grid(row=0, column=2, padx=10)
        ttk.Button(btnfrm, text="Schließen", command=self.destroy).grid(row=0, column=3, padx=10)

        info = (
            "Hinweise:\n"
            "• Mode 'fitmax': bestimmt N aus der Polygonfläche (mit Rotation/Phase) und erzeugt ein Rechteck/U mit gleicher N.\n"
            "• Legacy: altes Verhalten (Rechteck/U innerhalb der Fläche).\n"
            "• CSV/Meta/Preview & Backups im Unterordner 'autoborehole'.\n"
            "• Geändert werden nur: rows_sonden, Columns_sonden, Bx, By, H\n"
        )
        ttk.Label(frm, text=info, foreground="#333").grid(row=r+1, column=0, columnspan=2, sticky="w", padx=5, pady=10)

    def run_generate(self):
        try:
            params = LayoutParams(
                mode=self.mode.get(),
                layout_type=self.layout.get(),
                spacing_m=float(self.spacing.get()),
                depth_m=float(self.depth.get()),
                edge_offset_m=float(self.edge.get()),
                u_gap_m=float(self.u_gap.get()),
                overwrite=bool(self.overwrite.get()),
                rotate_step_deg=float(self.rotate_step.get()),
                phase_steps=int(self.phase_steps.get()),
            )
            project_dir = Path(self.project_dir.get()).resolve()
            shp_dir = Path(self.shp_dir.get()).resolve()

            res = compute_borefield(shp_dir, params, project_dir)

            msg = (
                f"Mode: {params.mode}\n"
                f"Layout: {params.layout_type}\n"
                f"Spacing: {params.spacing_m:.2f} m | Tiefe H: {params.depth_m:.2f} m\n"
                f"Rect/U-Layout: rows×cols = {res.rows} × {res.cols}  → N = {res.count}\n"
                f"Polygon-BBox: {res.bbox_w_m:.1f} m × {res.bbox_h_m:.1f} m | CRS EPSG:{res.crs_epsg}\n\n"
                f"{'Projektdateien wurden überschrieben.' if params.overwrite else 'Projektdateien NICHT überschrieben.'}\n"
                f"CSV/Meta/Preview im Unterordner 'autoborehole'."
            )
            messagebox.showinfo("Erfolg", msg)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def run_restore(self):
        try:
            project_dir = Path(self.project_dir.get()).resolve()
            restored = restore_latest_backup(project_dir)
            if restored:
                messagebox.showinfo("Restore", "Wiederhergestellt:\n" + "\n".join(str(p) for p in restored))
            else:
                messagebox.showinfo("Restore", "Keine Backups gefunden.")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def run_spacing_scan_dialog(self):
        try:
            # Freiform ist für den Quick‑Check nicht geeignet (Rechteckreferenz)
            if self.layout.get() == 'freeform':
                messagebox.showinfo('Hinweis', 'Spacing‑Quick‑Check nur mit Rechteck/U‑Layout möglich. Bitte Layout ≠ Freiform wählen.')
                return
            _spacing_scan_dialog(self)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli_main())
    else:
        if not _HAS_TK:
            print("Tkinter nicht verfügbar – starte CLI. Für GUI bitte Tkinter installieren.", file=sys.stderr)
            sys.exit(cli_main([]))
        app = App()
        app.mainloop()
