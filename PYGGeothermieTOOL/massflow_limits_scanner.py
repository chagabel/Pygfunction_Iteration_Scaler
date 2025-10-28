"""
massflow_limits_scanner.py – Massenstrom-Scanner (3‑Stufen)
-----------------------------------------------------------
Kurz erklärt (für Potenzialtests):
- Liefert pro Feld drei Massenstromstufen (kg/s, je Sonde und gesamt):
  1) min  = Baseline‑Deckel (heutiges „max“): hydraulisch oberes Limit des aktuellen Setups
            (v≤V_MAX_BASE, Δp_bh≤DP_BASE). Das ist die derzeit praxistaugliche Obergrenze.
  2) nom  = „oberes Mittel“: thermisch nicht mehr limitierend (Sättigung Rb_eff) UND moderater
            hydraulischer Deckel (v≤V_NOM, Δp_bh≤DP_NOM). Empfehlung für Potenzial‑Runs.
  3) max  = Markt‑Deckel: gleiche Geometrie, aber liberaler Deckel (v≤V_MAX_MARKET, Δp_bh≤DP_MARKET)
            als obere Machbarkeit (ersetzt „besseres Setup“ ohne Feldänderung).

Rechenweg in Kürze:
- Hydraulik: Darcy–Weisbach (Haaland‑f), Pfad ≈ 2·H·(1+L_eq_factor); v/Re aus rp_in.
- Thermik: Sättigung über Rb_eff(ṁ_bh) via pygfunction.pipes; erfüllt bei
  (Rb_eff(2·ṁ)−Rb_eff(ṁ))/Rb_eff(ṁ) ≤ SAT_EPS (1–2 %). Oberhalb davon bringt mehr ṁ kaum EWT‑Gewinn.
"""

from math import pi, log10
import inspect
import numpy as np
import matplotlib.pyplot as plt
import pygfunction as gt

import Data_Base as bd
import borefields as bf

# ==========================
# CONFIG – hier anpassen
# ==========================
RE_MIN_TARGET   = 2500     # Unteres Reynolds-Ziel (pro Rohrweg)
V_MIN_TARGET    = 0.40     # Untere Geschwindigkeit im Rohr [m/s]

# Baseline-Deckel (heutiges "max" ⇒ neues 'min') – praxistauglich "as built"
V_MAX_BASE      = 1.00     # [m/s]
DP_BH_MAX_BAR_BASE = 0.20  # [bar] pro Bohrlochpfad

# Moderater Deckel (für 'nom') – oberes Mittel
V_NOM_TARGET    = 1.10     # [m/s]
DP_BH_MAX_BAR_NOM = 0.30   # [bar]

# Markt-Deckel (für 'max') – obere Machbarkeit
V_MAX_MARKET    = 1.50     # [m/s]
DP_BH_MAX_BAR_MARK = 0.50  # [bar]

L_EQ_FACTOR     = 0.15     # Zuschlag äquivalente Länge (Bögen, U-Bogen, Ein-/Auslässe) [% von 2H]
LEGS_PER_BH     = 2        # Single-U = 2; Double-U = 4 (nur für Geschwindigkeits-Infos)

# Sättigungskriterium (thermisch „ṁ nicht mehr limitierend“)
SAT_EPS         = 0.01     # 1 % Rb_eff-Änderung bei Verdopplung

# Optional: Druckverlustgrenze für Verteiler (nur Hinweis)
DP_HEADER_HINT_BAR = 0.10  # bar (wird NICHT berechnet, nur als Reminder im Report gezeigt)

# ==========================
# Hilfsfunktionen
# ==========================

def fmt_de(x, nd=2):
    s = f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return s


def friction_factor(Re, eps, D):
    """Darcy-Reibungsbeiwert f. Laminar (64/Re), sonst Haaland-Approximation."""
    if Re <= 0:
        return np.inf
    if Re < 2300:
        return 64.0 / Re
    # Haaland (1983): 1/sqrt(f) = -1.8 log10[ (eps/D/3.7)^1.11 + 6.9/Re ]
    A = (eps / (D * 3.7))**1.11
    B = 6.9 / Re
    inv_sqrt_f = -1.8 * log10(A + B)
    f = 1.0 / (inv_sqrt_f**2)
    return f


def detect_fields(module_bf):
    """Suche in borefields.* alle Variablen, die iterierbar sind und Boreholes enthalten."""
    fields = []
    for name, obj in inspect.getmembers(module_bf):
        if name.startswith("__"):
            continue
        # Iterables mit Boreholes?
        try:
            it = iter(obj)
        except TypeError:
            continue
        # Prüfe ersten Eintrag
        try:
            first = next(iter(obj))
        except StopIteration:
            continue
        if isinstance(first, gt.boreholes.Borehole):
            fields.append((name, obj))
    return fields


def compute_massflow_limits_for_field(name, field):
    # Geometrie
    try:
        n_bh = len(field)
    except TypeError:
        n_bh = len(list(field))
    H = float(bd.H)
    L_path = 2.0 * H * (1.0 + L_EQ_FACTOR)   # effektive Pfadlänge je Sonde [m]

    # Fluid/rohr Daten
    rho = float(bd.fluid_density)            # kg/m3
    mu  = float(bd.fluid_viscosity)          # Pa·s
    cp  = float(bd.fluid_isobaric_heatcapacity)  # J/(kg·K), nur Info
    rp_in = float(bd.rp_in)                  # m
    eps   = float(getattr(bd, 'epsilon', 1.0e-6))
    D  = 2.0 * rp_in
    A  = pi * rp_in**2

    # Untere Turbulenzgrenze (Info)
    # pro Rohr (Schenkel): m_leg = rho*v*A; Re = rho*v*D/mu => m_leg = Re*mu*A/D
    m_bh_from_vmin  = LEGS_PER_BH * rho * V_MIN_TARGET * A
    m_bh_from_ReMin = LEGS_PER_BH * (RE_MIN_TARGET * mu * A / D)
    m_bh_turbulent_floor = max(m_bh_from_vmin, m_bh_from_ReMin)

    # Hydraulik-Helfer: Δp bei gegebener ṁ_bh
    def dp_borehole_for_mbh(m_bh):
        m_leg = m_bh / LEGS_PER_BH
        v = m_leg / (rho * A)
        Re = rho * v * D / mu
        f = friction_factor(Re, eps, D)
        dp = f * (L_path / D) * (rho * v * v / 2.0)  # Pa
        return dp, v, Re, f

    def hyd_limit(v_max_target: float, dp_budget_bar: float):
        m_bh_velmax = LEGS_PER_BH * rho * v_max_target * A
        dp_budget = float(dp_budget_bar) * 1e5
        dp_at_vmax, _, _, _ = dp_borehole_for_mbh(m_bh_velmax)
        if dp_at_vmax <= dp_budget:
            return m_bh_velmax, 'v_max'
        lo = 1e-9
        hi = m_bh_velmax
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            dp_mid, _, _, _ = dp_borehole_for_mbh(mid)
            if dp_mid > dp_budget:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) / max(1e-12, mid) < 1e-4:
                break
        return lo, 'Δp_budget'

    # Baseline-Deckel (heutiges "max" → neues 'min') – Single-U 32
    m_bh_baseline, limiting = hyd_limit(V_MAX_BASE, DP_BH_MAX_BAR_BASE)

    # Thermische Sättigung via Rb_eff‑Kriterium (Verdopplungstest)
    def rb_eff_for_mbh(m_bh: float) -> float:
        try:
            R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(bd.rp_in, bd.rp_out, bd.k_p)
            m_leg = max(1e-12, m_bh / LEGS_PER_BH)
            h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                m_leg, bd.rp_in, bd.fluid_viscosity, bd.fluid_density,
                bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon)
            R_f = 1.0/(h_f*2.0*np.pi*bd.rp_in)
            tmp = gt.pipes.SingleUTube(bd.pos_single, bd.rp_in, bd.rp_out, next(iter(field)), bd.k_s, bd.k_g, R_f+R_p)
            return float(tmp.effective_borehole_thermal_resistance(m_leg, bd.fluid_isobaric_heatcapacity))
        except Exception:
            return float('nan')

    m_test = max(m_bh_turbulent_floor, 1e-6)
    m_bh_sat = m_test
    for _ in range(12):
        R1 = rb_eff_for_mbh(m_test)
        R2 = rb_eff_for_mbh(2.0*m_test)
        if not np.isfinite(R1) or not np.isfinite(R2):
            break
        rel = abs(R2 - R1) / max(1e-9, R1)
        if rel <= SAT_EPS:
            m_bh_sat = m_test
            break
        m_test *= 2.0
    else:
        m_bh_sat = m_test

    # Varianten für nom/max (marktüblich): Single-U 40, Double-U 2x32, Double-U 2x40
    rp_in_32 = float(bd.rp_in); rp_out_32 = float(bd.rp_out)
    A32 = pi*rp_in_32**2; D32 = 2.0*rp_in_32
    rp_in_40 = 0.0163; rp_out_40 = 0.0200
    A40 = pi*rp_in_40**2; D40 = 2.0*rp_in_40

    def hyd_limit_variant(Avar, Dvar, legs, vlim, dplim_bar):
        def dp_for(m_bh):
            m_leg = m_bh/legs
            v = m_leg/(rho*Avar)
            Re = rho*v*Dvar/mu
            f = friction_factor(Re, eps, Dvar)
            return f*(L_path/Dvar)*(rho*v*v/2.0)
        m_vel = legs*rho*vlim*Avar
        dp_budget = dplim_bar*1e5
        if dp_for(m_vel) <= dp_budget:
            return m_vel
        lo = 1e-9; hi = m_vel
        for _ in range(50):
            mid=(lo+hi)/2
            if dp_for(mid) > dp_budget: hi=mid
            else: lo=mid
            if abs(hi-lo)/max(1e-12,mid) < 1e-4: break
        return lo

    # nom: best of Single-U 40 (legs=2) and Double-U 2x32 (legs=4) @ Δp=0.30 bar
    m_bh_nom_40 = hyd_limit_variant(A40, D40, 2, V_NOM_TARGET, DP_BH_MAX_BAR_NOM)
    m_bh_nom_du = hyd_limit_variant(A32, D32, 4, V_NOM_TARGET, DP_BH_MAX_BAR_NOM)
    m_bh_nom = max(m_bh_nom_40, m_bh_nom_du)

    # max: best of Double-U 2x40 and Single-U 40 @ Δp=0.50 bar
    m_bh_max_du40 = hyd_limit_variant(A40, D40, 4, V_MAX_MARKET, DP_BH_MAX_BAR_MARK)
    m_bh_max_40   = hyd_limit_variant(A40, D40, 2, V_MAX_MARKET, DP_BH_MAX_BAR_MARK)
    m_bh_max = max(m_bh_max_du40, m_bh_max_40)

    # Ausgabe-Mindestwert: konservativ am Turbulenz-Floor aufsetzen (wie bisherige Praxis)
    m_bh_min_out = max(m_bh_baseline, m_bh_turbulent_floor)

    # Gesamte Feldmassenströme (neue Semantik: min = Baseline/Floor, nom/max wie oben)
    m_total_min = n_bh * m_bh_min_out
    m_total_nom = n_bh * m_bh_nom
    m_total_max = n_bh * m_bh_max

    # Info zusammenstellen
    info = {
        'field_name': name,
        'n_bh': n_bh,
        'H': H,
        'legs_per_bh': LEGS_PER_BH,
        'rho': rho,
        'mu': mu,
        'cp': cp,
        'rp_in': rp_in,
        'D': D,
        'A': A,
        'Re_min_target': RE_MIN_TARGET,
        'v_min_target': V_MIN_TARGET,
        'v_max_target': V_MAX_BASE,
        'dp_bh_max_bar': DP_BH_MAX_BAR_BASE,
        'L_path': L_path,
        'eps': eps,
        'm_bh_min': m_bh_min_out,
        'm_bh_nom': m_bh_nom,
        'm_bh_max': m_bh_max,
        'm_total_min': m_total_min,
        'm_total_max': m_total_max,
        'm_total_nom': m_total_nom,
        'limiting': limiting,
        'm_bh_sat': m_bh_sat,
        'm_bh_hyd_nom': m_bh_nom,
        'm_bh_hyd_market': m_bh_max,
        'sat_epsilon': SAT_EPS,
        'dp_header_hint_bar': DP_HEADER_HINT_BAR,
    }
    return info


def print_and_box_report(infos):
    # Konsolenreport pro Feld
    print("\nMassenstrom – drei Stufen (lastunabhängig)")
    print("="*80)
    for inf in infos:
        print(f"Feld: {inf['field_name']} | n_bh={inf['n_bh']} | H={inf['H']:.2f} m | LEGS={inf['legs_per_bh']}")
        print("-"*80)
        print("Definitionen: min=Baseline‑Deckel (heutiges max), nom=Sättigung∧moderater Deckel, max=Markt‑Deckel.")
        print(f"min(Baseline):  m_dot_bh = {inf['m_bh_min']:.4f} kg/s  →  m_total = {inf['m_total_min']:.3f} kg/s  | limiting={inf['limiting']}")
        print(f"nom(Empfehlung): m_dot_bh = {inf['m_bh_nom']:.4f} kg/s  →  m_total = {inf['m_total_nom']:.3f} kg/s  | sat_eps={inf.get('sat_epsilon',0):.3f}")
        print(f"max(Markt):     m_dot_bh = {inf['m_bh_max']:.4f} kg/s  →  m_total = {inf['m_total_max']:.3f} kg/s")
        print(f"Δp-Header (nicht berechnet): Empfehlung ≤ {inf['dp_header_hint_bar']:.2f} bar (Prüfung außerhalb)")
        print()

    # Info-Box (ein Plot mit allen Feldern)
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=120)
    ax.axis('off')
    ax.text(0.02, 0.97, "Massenstrom – min/nom/max (ohne Last)", fontsize=13, fontweight='bold', va='top')

    y = 0.90
    for inf in infos:
        block = (
            f"Feld: {inf['field_name']}  |  n_bh={inf['n_bh']}  |  H={inf['H']:.2f} m  |  LEGS={inf['legs_per_bh']}\n"
            f"min(Baseline): {fmt_de(inf['m_bh_min'],4)} kg/s  …  nom: {fmt_de(inf['m_bh_nom'],4)}  …  max: {fmt_de(inf['m_bh_max'],4)}\n"
            f"gesamt: min={fmt_de(inf['m_total_min'],3)}  …  nom={fmt_de(inf['m_total_nom'],3)}  …  max={fmt_de(inf['m_total_max'],3)} kg/s\n"
            f"Limiter(Baseline): {inf['limiting']}   |   Δp(Header) ≤ {fmt_de(inf['dp_header_hint_bar'],2)} bar\n"
        )
        bbox = dict(boxstyle='round,pad=0.5', fc='#eef5ff', ec='#4a87d3', lw=1.2)
        ax.text(0.02, y, block, va='top', ha='left', fontsize=10, bbox=bbox)
        y -= 0.28
        if y < 0.05:
            break

    plt.tight_layout()
    plt.show()


def main():
    # Felder automatisch erkennen
    fields = detect_fields(bf)
    if not fields:
        print("Keine Borefields in borefields.py gefunden.")
        return

    infos = []
    for name, field in fields:
        info = compute_massflow_limits_for_field(name, field)
        infos.append(info)

    print_and_box_report(infos)


if __name__ == "__main__":
    main()
