"""VDI-Grobcheck für Erdwärmesondenfelder.

Dieser Vorab-Checker ermittelt aus den VDI-Tabellen (VDI 4640 Blatt 2,
Anhang B, Tabellen B2–B7 sowie Tabelle B1 für laminares Strömungsregime)
eine konservative spezifische Entzugsleistung q′ [W/m] und leitet daraus
die Spitzenleistung sowie die Jahresentzugsenergie für das aktuelle Feld
ab. Das Tool ersetzt keine Detailsimulation – es liefert die VDI-konforme
Größenordnung, bevor Longterm/pygfunction ins Detail gehen.

Voraussetzungen laut VDI-Tabellen:
- Linienanordnung mit 6 m Sondenabstand
- Doppel-U 32×3,0; Rb,eff ≈ 0,12 K/(W/m); 50 Jahre Betrachtungszeitraum
- Ungestörte Untergrundtemperatur 11 °C
- Turbulente Strömung (Laminarfaktor aus Tabelle B1 auswählbar)
- Tabellenwerte für 1…5 Sonden je Linie; wir clampen konservativ auf 5

Die Berechnung interpoliert linear in λ (1…4 W/(m·K)) und in den
Jahresvollaststunden (1200…2400 h/a bzw. 1500…2400 h/a mit Trinkwasser).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import Data_Base as bd
import borefields as bf


# ---------------------------------------------------------------------------
# VDI Tabellenwerte (B2–B7) und Laminar-Faktoren (B1)
# ---------------------------------------------------------------------------

LAMBDA_POINTS: List[float] = [1.0, 2.0, 3.0, 4.0]

LAMINAR_FACTORS: Dict[float, float] = {
    1.0: 0.85,
    2.0: 0.82,
    3.0: 0.80,
    4.0: 0.79,
}


def _table(rows: List[List[List[float]]], flh_values: List[int]) -> Dict[int, List[List[float]]]:
    """Utility to build dictionary {FLH: rows} from inline literals."""
    return {flh: [r[:] for r in table_rows] for flh, table_rows in zip(flh_values, rows)}


TABLE_B2 = _table(
    [
        [
            [37.5, 52.0, 61.5, 68.3],
            [34.3, 48.6, 58.3, 65.3],
            [32.1, 46.3, 56.1, 63.2],
            [30.6, 44.4, 54.3, 61.5],
            [29.7, 43.4, 53.4, 60.8],
        ],
        [
            [32.4, 47.0, 56.9, 64.0],
            [29.3, 43.4, 53.4, 60.7],
            [27.2, 41.0, 50.9, 58.4],
            [25.7, 39.1, 49.0, 56.5],
            [24.9, 38.0, 48.0, 55.6],
        ],
        [
            [28.6, 43.0, 53.0, 60.4],
            [25.6, 39.3, 49.3, 56.8],
            [23.8, 36.9, 46.7, 54.3],
            [22.4, 35.0, 44.8, 52.3],
            [21.6, 33.9, 43.6, 51.3],
        ],
        [
            [25.8, 39.8, 49.8, 57.4],
            [23.0, 36.1, 45.9, 53.5],
            [21.2, 33.7, 43.3, 50.9],
            [19.9, 31.9, 41.3, 48.8],
            [19.2, 30.8, 40.1, 47.7],
        ],
        [
            [23.7, 37.4, 47.3, 55.0],
            [21.0, 33.6, 43.3, 50.9],
            [19.3, 31.2, 40.6, 48.1],
            [18.0, 29.5, 38.5, 46.0],
            [17.3, 28.3, 37.3, 44.8],
        ],
    ],
    [1200, 1500, 1800, 2100, 2400],
)

TABLE_B3 = _table(
    [
        [
            [32.2, 44.7, 52.8, 58.6],
            [29.4, 41.6, 49.9, 55.9],
            [27.4, 39.4, 47.8, 53.9],
            [26.0, 37.7, 46.1, 52.2],
            [25.2, 36.8, 45.3, 51.6],
        ],
        [
            [27.8, 40.3, 48.8, 55.0],
            [25.1, 37.1, 45.6, 51.9],
            [23.3, 34.9, 43.4, 49.7],
            [22.0, 33.3, 41.6, 48.0],
            [21.3, 32.4, 40.7, 47.1],
        ],
        [
            [24.5, 36.9, 45.4, 51.8],
            [22.0, 33.6, 42.1, 48.5],
            [20.3, 31.5, 39.8, 46.2],
            [19.1, 29.9, 38.0, 44.4],
            [18.4, 28.9, 37.0, 43.4],
        ],
        [
            [22.1, 34.1, 42.7, 49.2],
            [19.7, 30.9, 39.2, 45.7],
            [18.1, 28.8, 36.9, 43.4],
            [17.0, 27.1, 35.0, 41.4],
            [16.4, 26.2, 34.0, 40.3],
        ],
        [
            [20.4, 32.1, 40.6, 47.1],
            [18.0, 28.8, 37.0, 43.4],
            [16.5, 26.6, 34.5, 40.9],
            [15.4, 25.0, 32.7, 39.0],
            [15.0, 24.3, 31.9, 38.2],
        ],
    ],
    [1200, 1500, 1800, 2100, 2400],
)

TABLE_B4 = _table(
    [
        [
            [24.4, 33.7, 39.8, 44.3],
            [22.1, 31.2, 37.4, 41.9],
            [20.6, 29.5, 35.6, 40.2],
            [19.4, 28.1, 34.2, 38.9],
            [18.8, 27.4, 31.8, 38.2],
        ],
        [
            [21.0, 30.4, 36.8, 41.5],
            [18.9, 27.9, 34.1, 38.9],
            [17.4, 26.1, 32.3, 37.0],
            [16.4, 24.7, 30.8, 35.6],
            [15.9, 24.0, 30.1, 34.9],
        ],
        [
            [18.6, 27.8, 34.2, 39.1],
            [16.6, 25.3, 31.5, 36.3],
            [15.2, 23.5, 29.6, 34.4],
            [14.2, 22.2, 28.1, 32.9],
            [13.8, 21.5, 27.2, 32.1],
        ],
        [
            [16.7, 25.7, 32.2, 37.0],
            [14.8, 23.2, 29.3, 34.2],
            [13.6, 21.5, 27.4, 32.2],
            [12.6, 20.1, 25.8, 30.6],
            [12.5, 19.4, 25.0, 29.8],
        ],
        [
            [15.4, 24.2, 30.5, 35.5],
            [13.6, 21.6, 27.6, 32.5],
            [12.0, 19.5, 25.3, 30.2],
            [11.1, 18.5, 24.1, 28.7],
            [10.7, 17.8, 23.3, 27.8],
        ],
    ],
    [1200, 1500, 1800, 2100, 2400],
)

TABLE_B5 = _table(
    [
        [
            [33.4, 48.0, 57.9, 65.0],
            [30.1, 44.3, 54.3, 61.6],
            [28.0, 41.8, 51.8, 59.2],
            [26.4, 39.9, 49.9, 57.4],
            [25.5, 38.8, 48.8, 56.5],
        ],
        [
            [29.4, 43.9, 53.9, 61.3],
            [26.3, 40.1, 50.2, 57.7],
            [24.4, 37.6, 47.5, 55.1],
            [22.9, 35.7, 45.5, 53.1],
            [22.1, 34.6, 44.4, 52.1],
        ],
        [
            [26.6, 40.7, 50.7, 58.3],
            [23.6, 36.9, 46.8, 54.4],
            [21.7, 34.4, 44.1, 51.7],
            [20.4, 32.5, 42.0, 49.6],
            [19.6, 31.4, 40.8, 48.5],
        ],
        [
            [24.4, 38.2, 48.3, 55.9],
            [21.6, 34.4, 44.1, 51.2],
            [19.8, 31.9, 41.3, 48.9],
            [18.4, 30.0, 39.2, 46.8],
            [17.7, 28.9, 38.0, 45.5],
        ],
    ],
    [1500, 1800, 2100, 2400],
)

TABLE_B6 = _table(
    [
        [
            [28.6, 41.2, 49.7, 55.8],
            [25.8, 37.9, 46.4, 52.7],
            [23.9, 35.6, 44.1, 50.4],
            [22.6, 33.9, 42.3, 48.7],
            [21.8, 33.0, 41.4, 47.8],
        ],
        [
            [25.3, 37.7, 46.3, 52.6],
            [22.6, 34.3, 42.8, 49.3],
            [21.2, 32.1, 40.5, 46.9],
            [19.6, 30.4, 38.6, 45.1],
            [18.8, 29.5, 37.6, 44.1],
        ],
        [
            [22.8, 34.9, 43.5, 50.0],
            [20.2, 31.6, 39.9, 46.4],
            [18.5, 29.3, 37.5, 44.0],
            [17.3, 27.7, 35.6, 42.0],
            [16.7, 26.7, 34.6, 41.0],
        ],
        [
            [21.0, 32.8, 41.3, 47.9],
            [18.5, 29.4, 37.7, 44.2],
            [16.9, 27.2, 35.2, 41.6],
            [15.8, 25.5, 33.3, 39.6],
            [15.1, 24.5, 32.1, 38.5],
        ],
    ],
    [1500, 1800, 2100, 2400],
)

TABLE_B7 = _table(
    [
        [
            [21.7, 31.1, 37.4, 42.1],
            [19.4, 28.5, 34.7, 39.4],
            [17.9, 26.6, 32.8, 37.6],
            [16.8, 25.2, 31.3, 36.0],
            [16.2, 24.5, 30.6, 35.4],
        ],
        [
            [18.6, 27.8, 34.2, 39.1],
            [17.0, 25.8, 32.1, 36.9],
            [15.6, 24.0, 30.1, 34.9],
            [14.6, 22.6, 28.5, 33.3],
            [14.1, 21.8, 27.7, 32.6],
        ],
        [
            [17.2, 26.3, 32.8, 37.6],
            [15.2, 23.7, 29.9, 34.7],
            [13.9, 21.9, 27.8, 32.7],
            [12.9, 20.5, 26.3, 31.1],
            [12.1, 19.8, 25.4, 30.2],
        ],
        [
            [15.9, 24.7, 31.1, 36.0],
            [13.9, 22.1, 28.1, 33.0],
            [12.3, 19.9, 25.8, 30.6],
            [11.4, 18.9, 24.5, 29.2],
            [10.9, 18.2, 23.7, 28.3],
        ],
    ],
    [1500, 1800, 2100, 2400],
)

INTERFERENCE_OPTIONS: List[Tuple[str, float]] = [
    ("Keine Zusatzkorrektur (VDI)", 0.0),
    ("−5 % (ca. 6–10 je Linie)", 0.05),
    ("−10 % (ca. 11–20 je Linie)", 0.10),
    ("−20 % (ca. 21–30 je Linie)", 0.20),
    ("−30 % (>30 je Linie – konservativ)", 0.30),
    ("Benutzerdefiniert …", -1.0),
]

VDI_TABLES = {
    ("heating", -5): TABLE_B2,
    ("heating", -3): TABLE_B3,
    ("heating", 0): TABLE_B4,
    ("heating_dhw", -5): TABLE_B5,
    ("heating_dhw", -3): TABLE_B6,
    ("heating_dhw", 0): TABLE_B7,
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def fmt_de(value: float, decimals: int = 0) -> str:
    """Format mit deutschem Dezimal- und Tausenderseparator."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


@dataclass
class FeldInfo:
    name: str
    n_bh: int
    H: float
    L_total: float
    rows: int
    cols: int
    Bx: float
    By: float
    spacing_mean: float
    lambda_default: float
    n_in_line_real: int
    n_in_line_used: int
    clamped: bool


def _detect_fields() -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    try:
        ff = getattr(bf, "freeform_field", None)
        if ff is not None:
            try:
                n = len(ff)
            except TypeError:
                n = len(list(ff))
            if n and int(n) > 0:
                out.append(("freeform_field", ff))
    except Exception:
        pass

    for nm in ("rectangle_field", "U_shaped_field"):
        try:
            obj = getattr(bf, nm, None)
            if obj is None:
                continue
            try:
                holes = obj.to_boreholes() if hasattr(obj, "to_boreholes") else obj
                n = len(holes)
            except Exception:
                try:
                    n = len(list(obj))
                except Exception:
                    n = 0
            if n and int(n) > 0:
                out.append((nm, obj))
        except Exception:
            pass
    return out


def _ermittle_feldinfo(feld_name: str, feld_obj) -> FeldInfo:
    try:
        n_bh = len(feld_obj)
    except TypeError:
        n_bh = len(list(feld_obj))

    H = float(getattr(bd, "H", 0.0))
    Bx = float(getattr(bd, "Bx", 0.0))
    By = float(getattr(bd, "By", 0.0))
    L_total = n_bh * H

    rows = int(getattr(bd, "rows_sonden", 0))
    cols = int(getattr(bd, "Columns_sonden", 0))
    if rows <= 0 and cols <= 0:
        # fallback: grob Wurzel des Feldes (für Freeform)
        import math

        approx = int(round(math.sqrt(max(1, n_bh))))
        rows = cols = approx

    n_in_line_real = max(rows, cols, 1)
    n_in_line_used = min(5, n_in_line_real)
    clamped = n_in_line_used != n_in_line_real

    spacing_mean = 0.5 * (Bx + By) if Bx and By else 0.0
    lambda_ground = float(getattr(bd, "k_s", getattr(bd, "conductivity", 2.0)))

    return FeldInfo(
        name=feld_name,
        n_bh=n_bh,
        H=H,
        L_total=L_total,
        rows=rows,
        cols=cols,
        Bx=Bx,
        By=By,
        spacing_mean=spacing_mean,
        lambda_default=lambda_ground,
        n_in_line_real=n_in_line_real,
        n_in_line_used=n_in_line_used,
        clamped=clamped,
    )


def _interp_linear(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    alpha = (x - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)


def _interpolate_rows(table: Dict[int, List[List[float]]], flh: float) -> List[List[float]]:
    flh_points = sorted(table.keys())
    if flh <= flh_points[0]:
        return [row[:] for row in table[flh_points[0]]]
    if flh >= flh_points[-1]:
        return [row[:] for row in table[flh_points[-1]]]
    for i in range(len(flh_points) - 1):
        f0, f1 = flh_points[i], flh_points[i + 1]
        if f0 <= flh <= f1:
            rows0 = table[f0]
            rows1 = table[f1]
            return [
                [_interp_linear(flh, f0, f1, v0, v1) for v0, v1 in zip(row0, row1)]
                for row0, row1 in zip(rows0, rows1)
            ]
    return [row[:] for row in table[flh_points[-1]]]


def _interp_lambda(values: List[float], lam: float) -> float:
    lam = max(min(lam, LAMBDA_POINTS[-1]), LAMBDA_POINTS[0])
    for i in range(len(LAMBDA_POINTS) - 1):
        l0, l1 = LAMBDA_POINTS[i], LAMBDA_POINTS[i + 1]
        if l0 <= lam <= l1:
            return _interp_linear(lam, l0, l1, values[i], values[i + 1])
    return values[-1]


def _laminar_factor(lam: float) -> float:
    lam = max(min(lam, max(LAMINAR_FACTORS)), min(LAMINAR_FACTORS))
    points = sorted(LAMINAR_FACTORS.keys())
    if lam <= points[0]:
        return LAMINAR_FACTORS[points[0]]
    if lam >= points[-1]:
        return LAMINAR_FACTORS[points[-1]]
    for i in range(len(points) - 1):
        l0, l1 = points[i], points[i + 1]
        if l0 <= lam <= l1:
            return _interp_linear(lam, l0, l1, LAMINAR_FACTORS[l0], LAMINAR_FACTORS[l1])
    return 1.0


def compute_qprime(
    fi: FeldInfo,
    lam: float,
    flh: float,
    mode: str,
    twp: float,
    laminar: bool,
    interference: float,
) -> Tuple[float, Dict[str, float]]:
    key = (mode, twp)
    if key not in VDI_TABLES:
        raise ValueError("Keine VDI-Tabelle für diese Kombination verfügbar.")

    table = VDI_TABLES[key]
    rows_interp = _interpolate_rows(table, flh)
    n_idx = max(1, min(fi.n_in_line_used, len(rows_interp))) - 1
    lambda_values = rows_interp[n_idx]
    q_prime = _interp_lambda(lambda_values, lam)

    lam_factor = 1.0
    if laminar:
        lam_factor = _laminar_factor(lam)
        q_prime *= lam_factor

    q_prime_raw = q_prime

    if interference > 0.0:
        interference = min(max(interference, 0.0), 0.8)
        q_prime *= (1.0 - interference)

    q_peak_w = q_prime * fi.L_total
    annual_kwh = q_peak_w * flh / 1000.0
    specific_kwh_per_m = q_prime * flh / 1000.0

    extras = {
        "q_prime_raw": q_prime_raw,
        "lam_factor": lam_factor,
        "q_peak_w": q_peak_w,
        "annual_kwh": annual_kwh,
        "specific_kwh_per_m": specific_kwh_per_m,
        "interference": interference,
    }
    return q_prime, extras


# ---------------------------------------------------------------------------
# Report-Aufbereitung
# ---------------------------------------------------------------------------


def build_report(
    fi: FeldInfo,
    q_prime: float,
    extras: Dict[str, float],
    lam: float,
    flh: float,
    mode: str,
    twp: float,
    laminar: bool,
) -> str:
    q_peak_kw = extras["q_peak_w"] / 1000.0
    q_peak_mw = q_peak_kw / 1000.0
    annual_kwh = extras["annual_kwh"]
    annual_mwh = annual_kwh / 1000.0
    annual_gwh = annual_kwh / 1_000_000.0
    spec_kwh_per_m = extras["specific_kwh_per_m"]
    interference = extras.get("interference", 0.0)
    q_prime_raw = extras.get("q_prime_raw", q_prime)

    lines = []
    lines.append("VDI-Grobcheck (Anhang B) – Vorababschätzung")
    lines.append("————————————————————————————————————————————————————————————")
    lines.append(f"Feld: {fi.name}")
    lines.append(f"  Sonden gesamt:          {fmt_de(fi.n_bh, 0)}")
    lines.append(f"  Bohrlochlänge je Sonde: {fmt_de(fi.H, 2)} m")
    lines.append(f"  Gesamtbohrmeter:        {fmt_de(fi.L_total, 0)} m")
    lines.append(f"  Raster Rows×Cols:       {fmt_de(fi.rows,0)} × {fmt_de(fi.cols,0)}")
    if fi.Bx and fi.By:
        lines.append(
            f"  Sondenabstände Bx/By:   {fmt_de(fi.Bx,2)} m / {fmt_de(fi.By,2)} m"
        )
        if abs(fi.spacing_mean - 6.0) > 0.5:
            lines.append("  Hinweis: VDI-Tabellen basieren auf 6 m Linienabstand (Plausibilitätscheck).")
    if fi.clamped:
        lines.append(
            f"  Hinweis: Feld hat {fmt_de(fi.n_in_line_real,0)} Sonden pro Linie – VDI-Tabelle gilt bis 5, konservativ mit 5 angesetzt."
        )

    lines.append("")
    lines.append("Eingaben:")
    lines.append(f"  λ           = {lam:.3f} W/(m·K)")
    lines.append(f"  FLH         = {flh:.0f} h/a")
    lines.append(f"  Modus       = {'nur Heizen' if mode=='heating' else 'Heizen + Trinkwasser'}")
    lines.append(f"  TWP-Austritt ≥ {twp:+.0f} °C")
    lines.append(
        f"  Strömung    = {'laminar' if laminar else 'turbulent'}"
        + (f" (Faktor {extras['lam_factor']:.3f} aus Tabelle B1)" if laminar else "")
    )

    lines.append("")
    lines.append("VDI-Ergebnis:")
    if abs(q_prime - q_prime_raw) > 1e-6:
        lines.append(f"  q′ (VDI)     = {q_prime_raw:.1f} W/m")
        lines.append(f"  q′ (mit Korr)= {q_prime:.1f} W/m  (Abschlag {interference*100:.0f} %)" )
    else:
        lines.append(f"  q′           = {q_prime:.1f} W/m")
    lines.append(f"  Q_peak       = {q_peak_kw:.0f} kW ({q_peak_mw:.3f} MW)")
    lines.append(
        f"  Jahresenergie = {fmt_de(annual_kwh,0)} kWh  (≈ {fmt_de(annual_mwh,1)} MWh / {fmt_de(annual_gwh,3)} GWh)"
    )
    lines.append(f"  Spezifisch    = {spec_kwh_per_m:.1f} kWh/(m·a)")

    lines.append("————————————————————————————————————————————————————————————")
    source = (
        "Quelle: VDI 4640 Blatt 2 (2019), Anhang B Tabellen B2–B7 (Interpolation über λ und FLH); "
        "Laminarfaktor gemäß Tabelle B1. Gilt für 6 m Linienabstand, Doppel-U 32×3,0, 50 Jahre."
    )
    if interference > 0.0:
        source += " Großfeld-Abschlag ist heuristisch (nicht VDI)."
    lines.append(source)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GUI / CLI
# ---------------------------------------------------------------------------


def _compute_and_show(
    field_name: str,
    lam: float,
    flh: float,
    mode: str,
    twp: float,
    laminar: bool,
    interference: float,
):
    feld_obj = getattr(bf, field_name, None)
    if feld_obj is None:
        raise ValueError("Feld nicht gefunden: " + field_name)
    fi = _ermittle_feldinfo(field_name, feld_obj)
    q_prime, extras = compute_qprime(fi, lam, flh, mode, twp, laminar, interference)
    return build_report(fi, q_prime, extras, lam, flh, mode, twp, laminar)


def _cli_main():
    fields = _detect_fields()
    if not fields:
        print("Kein nutzbares Feld gefunden. Bitte Projektparameter prüfen.")
        return
    print("Feld wählen:")
    for idx, (nm, _) in enumerate(fields, start=1):
        print(f"  [{idx}] {nm}")
    try:
        choice = int(input("Auswahl -> ").strip()) - 1
    except Exception:
        choice = 0
    choice = max(0, min(choice, len(fields) - 1))
    field_name, _ = fields[choice]

    lam = float(input(f"λ [W/(m·K)] (Default {bd.k_s}): ") or bd.k_s)
    flh = float(input("Volllaststunden h/a (Default 1800): ") or 1800.0)
    mode_in = input("Modus [h=Heizen, t=Heizen+TWW] (Default h): ").strip().lower() or "h"
    mode = "heating_dhw" if mode_in == "t" else "heating"
    twp_in = input("TWP-Austritt-Grenze [-5/-3/0] (Default -3): ").strip() or "-3"
    twp = float(twp_in)
    laminar = (input("Laminar? [y/N]: ").strip().lower() == "y")
    corr_in = input("Großfeld-Abschlag [%] (Default 0): ").strip()
    corr_factor = float(corr_in)/100.0 if corr_in else 0.0

    report = _compute_and_show(field_name, lam, flh, mode, twp, laminar, corr_factor)
    print(report)


def _run_gui():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception:
        _cli_main()
        return

    root = tk.Tk()
    root.title("VDI-Grobcheck (Anhang B)")
    root.geometry("720x540")
    root.minsize(640, 520)

    pad = {"padx": 10, "pady": 6}

    fields = [nm for (nm, _) in _detect_fields()]
    if not fields:
        fields = ["rectangle_field", "U_shaped_field", "freeform_field"]

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Feld:").grid(row=0, column=0, sticky="e", **pad)
    var_field = tk.StringVar(value=fields[0])
    cmb_field = ttk.Combobox(frame, textvariable=var_field, values=fields, width=28, state="readonly")
    cmb_field.grid(row=0, column=1, sticky="w", **pad)

    ttk.Label(frame, text="λ [W/(m·K)]:").grid(row=1, column=0, sticky="e", **pad)
    var_lambda = tk.DoubleVar(value=float(getattr(bd, "k_s", 2.5)))
    ttk.Entry(frame, textvariable=var_lambda, width=12).grid(row=1, column=1, sticky="w", **pad)

    ttk.Label(frame, text="Volllaststunden h/a:").grid(row=2, column=0, sticky="e", **pad)
    var_flh = tk.DoubleVar(value=1800.0)
    ttk.Entry(frame, textvariable=var_flh, width=12).grid(row=2, column=1, sticky="w", **pad)

    ttk.Label(frame, text="Betriebsart:").grid(row=3, column=0, sticky="e", **pad)
    var_mode = tk.StringVar(value="heating")
    ttk.Radiobutton(frame, text="nur Heizen", variable=var_mode, value="heating").grid(row=3, column=1, sticky="w", **pad)
    ttk.Radiobutton(frame, text="Heizen + TWW", variable=var_mode, value="heating_dhw").grid(row=3, column=2, sticky="w", **pad)

    ttk.Label(frame, text="TWP-Austritt ≥:").grid(row=4, column=0, sticky="e", **pad)
    var_twp = tk.StringVar(value="-3")
    cmb_twp = ttk.Combobox(frame, textvariable=var_twp, values=["-5", "-3", "0"], width=6, state="readonly")
    cmb_twp.grid(row=4, column=1, sticky="w", **pad)
    ttk.Label(frame, text="°C").grid(row=4, column=2, sticky="w")

    var_laminar = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Laminar-Korrektur (Tab. B1)", variable=var_laminar).grid(row=5, column=1, sticky="w", **pad)

    ttk.Label(frame, text="Großfeld-Abschlag:").grid(row=5, column=0, sticky="e", **pad)
    var_corr = tk.StringVar(value=INTERFERENCE_OPTIONS[0][0])
    cmb_corr = ttk.Combobox(frame, textvariable=var_corr, values=[opt[0] for opt in INTERFERENCE_OPTIONS], width=28, state="readonly")
    cmb_corr.grid(row=5, column=2, sticky="w", **pad)

    var_corr_custom = tk.DoubleVar(value=0.0)
    ttk.Label(frame, text="Benutzerdefiniert [%]:").grid(row=6, column=0, sticky="e", **pad)
    entry_custom = ttk.Entry(frame, textvariable=var_corr_custom, width=10)
    entry_custom.grid(row=6, column=1, sticky="w", **pad)
    ttk.Label(frame, text="(wirkt nur bei Option 'Benutzerdefiniert …')").grid(row=6, column=2, sticky="w", **pad)

    text_out = tk.Text(frame, height=12, wrap="word", font=("Courier New", 10))
    text_out.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=10, pady=(10, 6))
    text_out.configure(state="disabled")

    def _current_corr_factor() -> float:
        label = var_corr.get()
        for text_label, value in INTERFERENCE_OPTIONS:
            if text_label == label:
                if value >= 0:
                    return value
                break
        return max(0.0, float(var_corr_custom.get()) / 100.0)

    info_text = (
        "Berechnungsweg: q′ aus VDI 4640 Blatt 2 (2019), Anhang B Tabellen B2–B7; λ und Volllaststunden werden linear interpoliert; "
        "max. 5 Sonden je Linie (VDI-Bereich). Optionaler Laminar-Faktor aus Tabelle B1 sowie optionaler Großfeld-Abschlag (heuristisch, nicht VDI)."
    )
    ttk.Label(frame, text=info_text, wraplength=660, foreground="#444").grid(
        row=8, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 10)
    )

    frame.grid_rowconfigure(7, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    def show_report(report: str) -> None:
        text_out.configure(state="normal")
        text_out.delete("1.0", "end")
        text_out.insert("1.0", report)
        text_out.configure(state="disabled")

    def on_compute() -> None:
        try:
            field = var_field.get().strip()
            lam_val = float(var_lambda.get())
            flh_val = float(var_flh.get())
            if lam_val <= 0 or flh_val <= 0:
                raise ValueError("λ und FLH müssen > 0 sein.")
            mode = var_mode.get()
            twp = float(var_twp.get())
            corr_factor = _current_corr_factor()
            report = _compute_and_show(field, lam_val, flh_val, mode, twp, var_laminar.get(), corr_factor)
            show_report(report)
        except Exception as exc:
            messagebox.showerror("Fehler", str(exc))

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=9, column=0, columnspan=3, pady=(0, 10))
    ttk.Button(btn_frame, text="Berechnen", command=on_compute).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Schließen", command=root.destroy).pack(side="left", padx=6)

    # Initiale Ausgabe
    try:
        show_report(
            _compute_and_show(
                var_field.get(),
                var_lambda.get(),
                var_flh.get(),
                var_mode.get(),
                float(var_twp.get()),
                var_laminar.get(),
                _current_corr_factor(),
            )
        )
    except Exception as exc:
        show_report("Fehler bei der Initialberechnung: " + str(exc))

    root.mainloop()


def main():
    try:
        _run_gui()
    except Exception:
        _cli_main()


if __name__ == "__main__":
    main()
