#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANWENDUNG_START.py – Projekt‑Launcher

Zentrale Start‑GUI, welche die vorhandenen Tools in sinnvoller Reihenfolge
orchestriert:

1) Feld aus Shapefile bestimmen (auto_borefield.py – eigene GUI)
2) Massenstrom‑Korridor bestimmen (massflow_limits_scanner.py – im Hintergrund)
   und optional den nominalen Gesamt‑Massenstrom nach Data_Base.py schreiben.
3) Lastprofile erzeugen
   3A) Excel → GHEtool‑Last (excel_to_GHE3.py – eigene GUI)
   3B) Excel → q′(t) [W/m] für Longterm (excel_to_qprime.py – eigene GUI)
4) GHEtool‑Dimensionierer (GHEtool_Dimensionierer_v4.py – eigene GUI)
5) Longterm‑Checker (longterm_profile_scaler_v13.py – eigene GUI)

Hinweis: Externe GUIs werden als Subprozesse gestartet, damit Tkinter‑Mainloops
nicht kollidieren. Hintergrundaktionen (Schritt 2) laufen im selben Prozess.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox


# -----------------------------
# Utils
# -----------------------------

ROOT = Path(__file__).resolve().parent


def _log_exc_to_status(widget: tk.Text, exc: Exception) -> None:
    widget.insert("end", f"Fehler: {exc}\n")
    widget.see("end")


def _spawn_script(script_name: str) -> None:
    """Startet ein Python‑Skript in einem separaten Prozess (ohne zu warten)."""
    script_path = (ROOT / script_name).resolve()
    if not script_path.exists():
        messagebox.showerror("Fehlt", f"Skript nicht gefunden: {script_path}")
        return
    try:
        # Spyder/IPython setzen häufig MPLBACKEND=module://matplotlib_inline.backend_inline.
        # Das führt in Subprozessen dazu, dass keine Plotfenster erscheinen.
        # Lösung: Umgebung säubern und Backend auf TkAgg setzen.
        env = os.environ.copy()
        mb = env.get('MPLBACKEND','')
        if 'inline' in mb.lower() or mb.startswith('module://'):
            env.pop('MPLBACKEND', None)
            env['MPLBACKEND'] = 'TkAgg'
        # Optional: entkopple Qt-Settings, falls gesetzt
        env.pop('QT_API', None)
        subprocess.Popen([sys.executable, str(script_path)], cwd=str(ROOT), env=env)
    except Exception as e:
        messagebox.showerror("Start fehlgeschlagen", str(e))


def _safe_import(name: str):
    try:
        if name in sys.modules:
            return importlib.reload(sys.modules[name])
        return importlib.import_module(name)
    except Exception as e:
        return e


def _get_field_status():
    """Liest Projektstatus aus Data_Base/borefields. Gibt dict mit Kernwerten."""
    status = {
        "errors": [],
        "rows": None,
        "cols": None,
        "H": None,
        "Bx": None,
        "By": None,
        "n_bh": None,
        "L_total": None,
        "flow": None,
        "field_src": None,
    }
    bd = _safe_import("Data_Base")
    if isinstance(bd, Exception):
        status["errors"].append(f"Data_Base: {bd}")
        return status
    bf = _safe_import("borefields")
    if isinstance(bf, Exception):
        status["errors"].append(f"borefields: {bf}")
        return status

    try:
        status["rows"] = int(getattr(bd, "rows_sonden"))
        status["cols"] = int(getattr(bd, "Columns_sonden"))
        status["H"] = float(getattr(bd, "H"))
        status["Bx"] = float(getattr(bd, "Bx"))
        status["By"] = float(getattr(bd, "By"))
        status["flow"] = float(getattr(bd, "flow"))
    except Exception as e:
        status["errors"].append(f"Data_Base Felder: {e}")

    try:
        # Sammle Verfügbarkeit aller Felder für die Statusanzeige
        status["fields"] = []  # Liste von (name, n_bh, L_total)

        # Rechteck/U: N direkt aus rows*cols ableiten
        try:
            n_grid = int(getattr(bd, "rows_sonden")) * int(getattr(bd, "Columns_sonden"))
        except Exception:
            n_grid = 0
        if n_grid > 0:
            status["fields"].append(("rectangle", n_grid, float(status["H"]) * n_grid if status["H"] else None))
            status["fields"].append(("ushaped", n_grid, float(status["H"]) * n_grid if status["H"] else None))

        # Freeform: Länge der Liste aus borefields
        try:
            ff = getattr(bf, "freeform_field", None)
            if ff is not None:
                try:
                    n_ff = len(ff)
                except TypeError:
                    n_ff = len(list(ff))
                if int(n_ff) > 0:
                    status["fields"].append(("freeform", int(n_ff), float(status["H"]) * int(n_ff) if status["H"] else None))
        except Exception:
            pass

        # Setze Default-Feldauswahl (für n_bh/L_total Anzeige): prefer freeform > rectangle > ushaped
        for src in ("freeform", "rectangle", "ushaped"):
            for (nm, nbh, Ltot) in status.get("fields", []):
                if nm == src and int(nbh) > 0:
                    status["n_bh"] = int(nbh)
                    status["L_total"] = Ltot
                    status["field_src"] = nm
                    raise StopIteration
    except StopIteration:
        pass
    except Exception as e:
        status["errors"].append(f"Feldstatus: {e}")
    return status


def _find_latest_autoborehole_meta() -> Optional[Path]:
    d = ROOT / "autoborehole"
    if not d.exists():
        return None
    candidates = sorted(d.glob("borefield_layout.meta.json"))
    if candidates:
        return candidates[-1]
    # Fallback: neueste *.meta.json
    c2 = sorted(d.glob("*.meta.json"), key=lambda p: p.stat().st_mtime)
    return c2[-1] if c2 else None


def _patch_flow_in_Data_Base(new_flow: float) -> bool:
    """Ersetzt die Zuweisung 'flow = ...' in Data_Base.py. Rückgabe: True bei Erfolg."""
    path = ROOT / "Data_Base.py"
    if not path.exists():
        messagebox.showerror("Fehlt", f"Data_Base.py nicht gefunden unter {path}")
        return False
    txt = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"^\s*flow\s*=\s*([0-9eE\.\,\+\-]+)", re.MULTILINE)
    if not pattern.search(txt):
        # kein Treffer – hänge unten an
        new_txt = txt + f"\nflow = {float(new_flow):.6g}\n"
    else:
        new_txt = pattern.sub(f"flow = {float(new_flow):.6g}", txt)
    try:
        path.write_text(new_txt, encoding="utf-8")
        return True
    except Exception as e:
        messagebox.showerror("Schreiben fehlgeschlagen", str(e))
        return False


# -----------------------------
# GUI
# -----------------------------


@dataclass
class MassflowSuggestion:
    m_bh_min: float
    m_bh_nom: float
    m_bh_max: float
    m_total_min: float
    m_total_nom: float
    m_total_max: float
    limiting: str
    n_bh: int
    H: float


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Projekt‑Launcher")
        self.geometry("980x760")

        self._massflow: Optional[MassflowSuggestion] = None
        self._field_choices: list[str] = []
        self.var_field_choice = tk.StringVar(value="rectangle_field")

        # Kopf: Projektstatus
        frm_state = ttk.LabelFrame(self, text="Projektstatus")
        frm_state.pack(fill="x", padx=10, pady=(10, 6))
        self.lbl_state = ttk.Label(frm_state, text="–", wraplength=900, justify="left")
        self.lbl_state.pack(fill="x", padx=8, pady=6)
        ttk.Button(frm_state, text="Status aktualisieren", command=self.refresh_state).pack(anchor="e", padx=8, pady=(0,6))

        # Schritte
        self._build_steps()

        # Log unten
        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, padx=10, pady=(6, 10))
        self.txt = tk.Text(frm_log, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True)

        self.refresh_state()

    # ---------- UI Aufbau ----------
    def _build_steps(self) -> None:
        pad = dict(padx=10, pady=6)

        # 1) Feld bestimmen
        f1 = ttk.LabelFrame(self, text="1) Feld aus Shapefile bestimmen (Auto‑Borefield)")
        f1.pack(fill="x", padx=10, pady=6)
        ttk.Label(f1, text=(
            "Liest ./shp/*.shp, bestimmt N/rows×cols/Bx/By/H und patcht Projektdateien. "
            "Artefakte & Backups im Unterordner 'autoborehole'."), wraplength=900, justify="left").grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(f1, text="Auto‑Borefield starten (GUI)", command=lambda: _spawn_script("auto_borefield.py")).grid(row=1, column=0, sticky="w", **pad)
        ttk.Button(f1, text="Letztes Meta anzeigen", command=self._show_latest_borefield_meta).grid(row=1, column=1, sticky="w", **pad)
        ttk.Button(f1, text="Status aktualisieren", command=self.refresh_state).grid(row=1, column=2, sticky="e", **pad)

        # 2) Massenstrom
        f2 = ttk.LabelFrame(self, text="2) Massenstrom‑Korridor bestimmen und übernehmen")
        f2.pack(fill="x", padx=10, pady=6)
        ttk.Label(f2, text=(
            "Hydraulisch sinnvoller Massenstrom (ohne Last): min/nom/max pro Sonde & gesamt. "
            "Nominalwert kann direkt nach Data_Base.py geschrieben werden."), wraplength=900, justify="left").grid(row=0, column=0, columnspan=5, sticky="w", **pad)
        ttk.Button(f2, text="Korridor berechnen", command=self._calc_massflow).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(f2, text="Feld:").grid(row=1, column=1, sticky="e", **pad)
        self.cmb_field = ttk.Combobox(f2, textvariable=self.var_field_choice, values=[], width=22, state="readonly")
        self.cmb_field.grid(row=1, column=2, sticky="w", **pad)
        ttk.Label(f2, text="Übernehmen:").grid(row=1, column=3, sticky="e", **pad)
        self.var_mset = tk.StringVar(value="nominal")
        ttk.Radiobutton(f2, text="min", variable=self.var_mset, value="min").grid(row=1, column=4, sticky="w", **pad)
        ttk.Radiobutton(f2, text="nominal", variable=self.var_mset, value="nominal").grid(row=1, column=5, sticky="w", **pad)
        ttk.Radiobutton(f2, text="max", variable=self.var_mset, value="max").grid(row=1, column=6, sticky="w", **pad)
        ttk.Button(f2, text="In Data_Base.py schreiben", command=self._apply_massflow).grid(row=1, column=7, sticky="w", **pad)
        # Breiter, um alle Korridor‑Infos lesbar anzuzeigen (mit Umbruch)
        self.lbl_massflow = ttk.Label(f2, text="–", wraplength=900, justify="left")
        self.lbl_massflow.grid(row=2, column=0, columnspan=8, sticky="w", **pad)

        # 3) Lastprofile
        f3 = ttk.LabelFrame(self, text="3) Lastprofile erzeugen")
        f3.pack(fill="x", padx=10, pady=6)
        ttk.Label(f3, text="GHEtool‑Last (kW) und q′(t) [W/m] für Longterm erzeugen (Excel‑basiert).", wraplength=900, justify="left").grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(f3, text="Excel → GHEtool‑Last (GUI)", command=lambda: _spawn_script("excel_to_GHE3.py")).grid(row=1, column=0, sticky="w", **pad)
        ttk.Button(f3, text="Excel → q′(t) [W/m] (GUI)", command=lambda: _spawn_script("excel_to_qprime.py")).grid(row=1, column=1, sticky="w", **pad)
        ttk.Button(f3, text="Status aktualisieren", command=self.refresh_state).grid(row=1, column=2, sticky="e", **pad)

        # 4) GHEtool
        f4 = ttk.LabelFrame(self, text="4) GHEtool‑Dimensionierer (L4)")
        f4.pack(fill="x", padx=10, pady=6)
        ttk.Label(f4, text="Prüft erforderliche Tiefe bzw. Layout bei gegebener Last (CSV).", wraplength=900, justify="left").grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(f4, text="GHEtool‑GUI starten", command=lambda: _spawn_script("GHEtool_Dimensionierer_v4.py")).grid(row=0, column=1, sticky="w", **pad)

        # 5) Longterm
        f5 = ttk.LabelFrame(self, text="5) Longterm‑Checker v13")
        f5.pack(fill="x", padx=10, pady=6)
        ttk.Label(f5, text="Iteriert Gebäudelast ODER Solar bis Grenzwertverletzung; nutzt q′‑CSV(s).", wraplength=900, justify="left").grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(f5, text="Longterm‑GUI starten", command=lambda: _spawn_script("longterm_profile_scaler_v13.py")).grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(f5, text="Vorabschätzung nach VDI", command=lambda: _spawn_script("vdi_grobcheck.py")).grid(row=0, column=2, sticky="w", **pad)

        # 6) COP‑Schätzer
        f6 = ttk.LabelFrame(self, text="6) COP‑Schätzer (Heizbetrieb)")
        f6.pack(fill="x", padx=10, pady=6)
        ttk.Label(f6, text="COP‑Abschätzung aus q′(t) [W/m] oder synthetischem Profil; rechnet T_b/T_in/T_out und COP(t).", wraplength=900, justify="left").grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(f6, text="COP‑GUI starten", command=lambda: _spawn_script("hp_cop_analyzer.py")).grid(row=0, column=1, sticky="w", **pad)

    # ---------- Aktionen ----------
    def refresh_state(self) -> None:
        st = _get_field_status()
        if st["errors"]:
            txt = " | ".join(st["errors"]) or "–"
        else:
            src = (st.get('field_src') or 'auto')
            rows = st.get('rows')
            cols = st.get('cols')
            H = st.get('H')
            Bx = st.get('Bx')
            By = st.get('By')
            L_total = st.get('L_total')
            flow = st.get('flow')

            parts = [
                f"Feld: rows×cols = {rows}×{cols}",
                f"n_bh={st.get('n_bh')}"
            ]
            parts.append(f"H={H:.2f} m" if H is not None else "H=–")
            if Bx is not None and By is not None:
                parts.append(f"Bx/By={Bx:.2f}/{By:.2f} m")
            else:
                parts.append("Bx/By=–/– m")
            parts.append(f"L_total={L_total:.0f} m" if L_total is not None else "L_total=– m")
            parts.append(f"flow={flow:.3f} kg/s" if flow is not None else "flow=– kg/s")
            parts.append(f"src={src}")
            base = "  |  ".join(parts)

            fields_list = []
            for (nm, nbh, Ltot) in (st.get('fields') or []):
                label = f"{nm}: n={nbh}, L={int(Ltot or 0)} m"
                if nm == 'freeform':
                    label += " (für GHEtool nicht nutzbar)"
                fields_list.append(label)
            extra = (" | Felder: " + "; ".join(fields_list)) if fields_list else ""
            txt = base + extra
        self.lbl_state.config(text=txt)

        # q′‑CSV Status kurz ins Log
        qp = ROOT / "qprime_profile_from_excel.csv"
        rs = ROOT / "qprime_recharge_only.csv"
        g1 = ROOT / "Wärmelastprofil2_GHEload_ohne_Rueckspeisung.csv"
        g2 = ROOT / "Wärmelastprofil2_GHEload_mit_Rueckspeisung.csv"
        hints = []
        hints.append(f"q′ CSV: {'OK' if qp.exists() else 'fehlt'}")
        hints.append(f"Solar CSV: {'OK' if rs.exists() else 'fehlt'}")
        if g1.exists() or g2.exists():
            hints.append("GHEtool‑CSV vorhanden")
        self.txt.insert("end", "Status: " + ", ".join(hints) + "\n")
        self.txt.see("end")
        # Feldliste für Massflow-Scan aktualisieren
        try:
            mls = _safe_import("massflow_limits_scanner")
            bf = _safe_import("borefields")
            if not isinstance(mls, Exception) and not isinstance(bf, Exception):
                fields = mls.detect_fields(bf) or []
                # Ergänze Rechteck/U explizit via .to_boreholes(), falls vorhanden
                try:
                    rect = getattr(bf, 'rectangle_field', None)
                    if rect is not None and hasattr(rect, 'to_boreholes'):
                        holes = rect.to_boreholes()
                        if holes:
                            fields.append(('rectangle_field', holes))
                except Exception:
                    pass
                try:
                    ush = getattr(bf, 'U_shaped_field', None)
                    if ush is not None and hasattr(ush, 'to_boreholes'):
                        holes = ush.to_boreholes()
                        if holes:
                            fields.append(('U_shaped_field', holes))
                except Exception:
                    pass
                # De-Dupe by name (last wins so our conversions override if needed)
                names = []
                unique = []
                for nm, obj in fields:
                    if nm not in names:
                        names.append(nm); unique.append((nm, obj))
                fields = unique
                if names:
                    self._field_choices = names
                    try:
                        self.cmb_field.configure(values=names)
                        # Standardauswahl: zuvor gewählte behalten; sonst freeform, dann rectangle, sonst erstes
                        pref = (self.var_field_choice.get() or '').strip()
                        if pref not in names:
                            if 'freeform_field' in names:
                                self.var_field_choice.set('freeform_field')
                            elif 'rectangle_field' in names:
                                self.var_field_choice.set('rectangle_field')
                            else:
                                self.var_field_choice.set(names[0])
                    except Exception:
                        pass
        except Exception:
            pass

    def _show_latest_borefield_meta(self) -> None:
        p = _find_latest_autoborehole_meta()
        if not p:
            messagebox.showinfo("Meta", "Keine Meta‑Datei in ./autoborehole gefunden.")
            return
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Lesefehler", str(e)); return
        lines = [
            f"Datei: {p}",
            f"N={meta.get('N')}  rows×cols={meta.get('rows')}×{meta.get('cols')}  spacing={meta.get('spacing_m')} m",
            f"H={meta.get('depth_m')} m  BBox={meta.get('bbox_w_m')} × {meta.get('bbox_h_m')} m",
            f"Layout={meta.get('layout_type')}  Mode={meta.get('mode')}  EPSG={meta.get('crs_epsg')}",
        ]
        messagebox.showinfo("Auto‑Borefield Meta", "\n".join(lines))

    def _calc_massflow(self) -> None:
        self.lbl_massflow.config(text="Berechne …")
        try:
            mls = _safe_import("massflow_limits_scanner")
            if isinstance(mls, Exception):
                raise mls
            bf = _safe_import("borefields")
            if isinstance(bf, Exception):
                raise bf

            # Feldwahl: aus Combobox, sonst rectangle bevorzugen
            fields = mls.detect_fields(bf)
            if not fields:
                fields = []
            # Ergänze Rechteck/U explizit über .to_boreholes(), falls vorhanden
            try:
                rect = getattr(bf, 'rectangle_field', None)
                if rect is not None and hasattr(rect, 'to_boreholes'):
                    holes = rect.to_boreholes()
                    if holes:
                        fields.append(('rectangle_field', holes))
            except Exception:
                pass
            try:
                ush = getattr(bf, 'U_shaped_field', None)
                if ush is not None and hasattr(ush, 'to_boreholes'):
                    holes = ush.to_boreholes()
                    if holes:
                        fields.append(('U_shaped_field', holes))
            except Exception:
                pass
            if not fields:
                raise RuntimeError("Keine Borefields in borefields.py erkannt.")
            # Map der letzten Vorkommen (unsere Konvertierungen überschreiben ggf.)
            names = {}
            for nm, obj in fields:
                names[nm] = obj
            chosen = self.var_field_choice.get().strip() or 'rectangle_field'
            if chosen in names:
                field_name, field = chosen, names[chosen]
            elif 'freeform_field' in names:
                field_name, field = 'freeform_field', names['freeform_field']
            elif 'rectangle_field' in names:
                field_name, field = 'rectangle_field', names['rectangle_field']
            else:
                field_name, field = next(iter(names.items()))

            info = mls.compute_massflow_limits_for_field(field_name, field)
            self._massflow = MassflowSuggestion(
                m_bh_min=info['m_bh_min'], m_bh_nom=info['m_bh_nom'], m_bh_max=info['m_bh_max'],
                m_total_min=info['m_total_min'], m_total_nom=info['m_total_nom'], m_total_max=info['m_total_max'],
                limiting=str(info['limiting']), n_bh=int(info['n_bh']), H=float(info['H'])
            )
            msg = (
                "Massenstrom – min/nom/max (ohne Last)\n"
                f"Feld: {field_name} | n_bh={info['n_bh']} | H={info['H']:.1f} m\n"
                f"pro Sonde [kg/s]: min={info['m_bh_min']:.4f}  nom={info['m_bh_nom']:.4f}  max={info['m_bh_max']:.4f}\n"
                f"gesamt [kg/s]:   min={info['m_total_min']:.3f}  nom={info['m_total_nom']:.3f}  max={info['m_total_max']:.3f}\n"
                "Definitionen: min=Baseline‑Deckel (aktuelles Setup: v≤1.0 m/s, Δp_bh≤0.2 bar);\n"
                "nom=oberes Mittel (marktüblich): thermische Sättigung (ΔRb/Rb≤1%) UND Δp_bh≤0.3 bar, v≤1.1 m/s;\n"
                "max=Markt‑Deckel: thermische Sättigung UND Δp_bh≤0.5 bar, v≤1.5 m/s (z. B. Double‑U/40er/Coax).\n"
                f"Limiter(Baseline): {info['limiting']}  |  Header‑Hinweis ≤ {info['dp_header_hint_bar']:.2f} bar"
            )
            self.lbl_massflow.config(text=msg)
            self.txt.insert("end", msg + "\n")
            self.txt.see("end")
        except Exception as e:
            self._massflow = None
            self.lbl_massflow.config(text=f"Fehler: {e}")
            _log_exc_to_status(self.txt, e)

    def _apply_massflow(self) -> None:
        if not self._massflow:
            messagebox.showwarning("Hinweis", "Bitte zuerst den Korridor berechnen.")
            return
        choice = self.var_mset.get()
        if choice == "min":
            m_total = self._massflow.m_total_min
        elif choice == "max":
            m_total = self._massflow.m_total_max
        else:
            m_total = self._massflow.m_total_nom

        if _patch_flow_in_Data_Base(m_total):
            messagebox.showinfo("Gespeichert", f"flow = {m_total:.6g} kg/s in Data_Base.py gesetzt.")
            # Neu laden, Status aktualisieren
            if "Data_Base" in sys.modules:
                try:
                    importlib.reload(sys.modules["Data_Base"])  # type: ignore
                except Exception:
                    pass
            self.refresh_state()
        else:
            messagebox.showerror("Fehler", "Konnte flow nicht in Data_Base.py schreiben.")


def main() -> None:
    app = Launcher()
    app.mainloop()


if __name__ == "__main__":
    main()
