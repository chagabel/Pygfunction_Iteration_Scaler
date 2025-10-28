# borehole.py
# -----------------------------------------------------------------------------
# WICHTIGE ÄNDERUNGEN (Fixes)
# -----------------------------------------------------------------------------
# 1) Effektiver Bohrlochwiderstand:
#    Statt R_b ≈ (R_f + R_p) wird jetzt für Single-U-Tube der
#    ***effektive Bohrlochwiderstand*** R_b,eff aus der Geometrie + Cross-Talk
#    verwendet:
#       R_b,eff = SingleUTube.effective_borehole_thermal_resistance(m_bh, cp)
#    -> realistischere Fluidtemperaturen.
#
# 2) Massenstrom pro Sonde:
#    m_bh = flow / n_bh (robuste n_bh-Ermittlung).
#
# 3) Diagnostik:
#    Ausgabe von R_b,eff und Vergleich zu (R_f + R_p).
#
# 4) Coaxial:
#    Belässt R_ff/R_fp wie in pygfunction vorgesehen, meldet aber R_b,eff
#    zur Diagnose (Coaxial nutzt separate Widerstandsarme; R_b,eff nur als Info).
# -----------------------------------------------------------------------------

import numpy as np
from scipy.constants import pi
import pygfunction as gt
import Data_Base as bd
import matplotlib.pyplot as plt
import borefields as bf


# Ein einzelnes Borehole-Objekt (für Geometrie/Plots)
borehole = gt.boreholes.Borehole(bd.H, bd.D, bd.r, x=0., y=0.)


def _count_boreholes() -> int:
    """Robuste Ermittlung der Sondenanzahl."""
    # 1) Falls explizit in Data_Base gesetzt
    if hasattr(bd, "Total_sondas"):
        try:
            n = int(bd.Total_sondas)
            if n > 0:
                return n
        except Exception:
            pass
    # 2) Versuche, aus den Feldern zu lesen
    for fld in (getattr(bf, "rectangle_field", None),
                getattr(bf, "U_shaped_field", None)):
        if fld is None:
            continue
        try:
            return len(fld)
        except TypeError:
            try:
                return len(list(fld))
            except Exception:
                continue
    # 3) Fallback
    return 1


def _flow_per_borehole():
    n_bh = max(1, _count_boreholes())
    return float(bd.flow) / n_bh, n_bh


def U_tube():
    """
    Demo/Diagnose für Single-U-Tube:
    - Bestimmt R_b,eff für den aktuellen m_bh und cp
    - Visualisiert die Rohrgeometrie
    - Gibt Plausibilitäts- und Widerstandsinfos aus
    """
    # --- Massenstrom je Sonde ---
    m_bh, n_bh = _flow_per_borehole()
    cp = float(bd.fluid_isobaric_heatcapacity)

    # --- Rohr-/Filmwiderstände (für Diagnose) ---
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(bd.rp_in, bd.rp_out, bd.k_p)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_bh, bd.rp_in, bd.fluid_viscosity, bd.fluid_density,
        bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon
    )
    R_f = 1.0 / (h_f * 2 * pi * bd.rp_in)
    R_sum = R_f + R_p

    # --- Temporäre Instanz, um R_b,eff zu berechnen ---
    tmp = gt.pipes.SingleUTube(bd.pos_single, bd.rp_in, bd.rp_out, borehole,
                               bd.k_s, bd.k_g, R_sum)
    check_single = tmp._check_geometry()
    R_b_eff = tmp.effective_borehole_thermal_resistance(m_bh, cp)

    # --- Finale Instanz mit R_b,eff (für korrekte T_in/T_out-Rechnungen anderswo) ---
    SingleUTube_eff = gt.pipes.SingleUTube(bd.pos_single, bd.rp_in, bd.rp_out, borehole,
                                           bd.k_s, bd.k_g, R_b_eff)

    # --- Diagnose-Ausgaben ---
    print(f'Borehole-Geometrie plausibel: {check_single!s}.')
    print(f'Single U-tube – Diagnose:')
    print(f'  n_bh={n_bh:d} | m_bh={m_bh:.3f} kg/s | cp={cp:.0f} J/(kg·K)')
    print(f'  R_f ≈ {R_f:.4f} m·K/W | R_p ≈ {R_p:.4f} m·K/W | (R_f+R_p) ≈ {R_sum:.4f} m·K/W')
    print(f'  R_b,eff ≈ {R_b_eff:.4f} m·K/W   (ersetzt (R_f+R_p) in der finalen Instanz)')

    # --- Visualisierung ---
    SingleUTube_eff.visualize_pipes()
    plt.show()


def coaxial():
    """
    Demo/Diagnose für Coaxial:
    - Nutzt die pygfunction-Coaxial-Parameterisierung (R_ff / R_fp)
    - Berechnet und meldet R_b,eff diagnostisch
    - Visualisiert die Rohrgeometrie
    """
    # --- Massenstrom je Sonde ---
    m_bh, n_bh = _flow_per_borehole()
    cp = float(bd.fluid_isobaric_heatcapacity)

    # --- Coaxial-Geometrie & Materialwiderstände ---
    pos = (0., 0.)
    R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(bd.r_in_in, bd.r_in_out, bd.k_p)
    R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(bd.r_out_in, bd.r_out_out, bd.k_p)

    # --- Filmwiderstände (innen & Ringspalt) ---
    h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_bh, bd.r_in_in, bd.fluid_viscosity, bd.fluid_density,
        bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon
    )
    R_f_in = 1.0 / (h_f_in * 2 * pi * bd.r_in_in)

    h_f_a_in, h_f_a_out = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
        m_bh, bd.r_in_out, bd.r_out_in, bd.fluid_viscosity, bd.fluid_density,
        bd.fluid_termal_conduct, bd.fluid_isobaric_heatcapacity, bd.epsilon
    )
    R_f_out_in  = 1.0 / (h_f_a_in  * 2 * pi * bd.r_in_out)
    R_f_out_out = 1.0 / (h_f_a_out * 2 * pi * bd.r_out_in)

    # --- Aggregation gemäß pygfunction-Coaxial ---
    R_ff = R_f_in + R_p_in + R_f_out_in
    R_fp = R_p_out + R_f_out_out

    Coaxial = gt.pipes.Coaxial(pos, bd.r_inner, bd.r_outer, borehole,
                               bd.k_s, bd.k_g, R_ff, R_fp, J=2)

    # --- Effektiven Bohrlochwiderstand diagnostisch melden ---
    R_b_eff = Coaxial.effective_borehole_thermal_resistance(m_bh, cp)

    print('Coaxial – Diagnose:')
    print(f'  n_bh={n_bh:d} | m_bh={m_bh:.3f} kg/s | cp={cp:.0f} J/(kg·K)')
    print(f'  R_ff ≈ {R_ff:.4f} m·K/W | R_fp ≈ {R_fp:.4f} m·K/W')
    print(f'  R_b,eff ≈ {R_b_eff:.4f} m·K/W  (diagnostisch; Coaxial nutzt R_ff/R_fp)')

    # --- Visualisierung ---
    Coaxial.visualize_pipes()
    plt.show()
