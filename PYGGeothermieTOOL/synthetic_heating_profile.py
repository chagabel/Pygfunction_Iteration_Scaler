# synthetic_heating_profile.py
# Heiz-only Profil (Rückspeisung = 0), auf Jahresenergie "Bedarf" skaliert.
# Konvention: Heizen = POSITIV (Winter +, Sommer − wird auf 0 gekappt).

import numpy as np
import matplotlib.pyplot as plt

def _bernier_raw(hours: np.ndarray) -> np.ndarray:
    A = 2000.0; B = 2190.0; C = 80.0; D = 2.0; E = 0.01; F = 0.0; G = 0.95
    hours = np.asarray(hours, dtype=float)
    func = (168.0 - C)/168.0
    for i in [1, 2, 3]:
        func += 1.0/(i*np.pi) * (np.cos(C*np.pi*i/84.0) - 1.0) * np.sin(np.pi*i/84.0 * (hours - B))
    func = func * A * np.sin(np.pi/12.0*(hours - B)) * np.sin(np.pi/4380.0*(hours - B))
    y = (func
         + (-1.0)**np.floor(D/8760.0*(hours - B)) * np.abs(func)
         + E * (-1.0)**np.floor(D/8760.0*(hours - B)) / np.sign(np.cos(D*np.pi/4380.0*(hours - F)) + G))
    return y  # Rohprofil (±)

def build_heating_only_profile(time_seconds, dt_seconds, annual_kWh):
    """
    Heizprofil Q[W] OHNE Rückspeisung (alles <0 → 0), skaliert auf 'annual_kWh'.
    Heizen = POSITIV (Winter +). Sommer (−) wird gekappt.
    Returns: Q (np.ndarray), info (dict)
    """
    t = np.asarray(time_seconds, dtype=float)
    dt = float(dt_seconds)
    hours = t / 3600.0

    # WICHTIG: Vorzeichen drehen, damit Winter + / Sommer − wie im ursprünglichen Projekt
    q_raw  = -_bernier_raw(hours)               # <- Fix
    q_heat = np.where(q_raw > 0.0, q_raw, 0.0)  # Sommer-Rückspeisung kappen

    E_raw_kWh = (np.sum(q_heat) * dt) / 3.6e6
    scale = 0.0 if E_raw_kWh == 0 else (annual_kWh / E_raw_kWh)
    Q = q_heat * scale

    info = {
        "E_raw_kWh": float(E_raw_kWh),
        "scale_factor": float(scale),
        "E_scaled_kWh": float((np.sum(Q) * dt) / 3.6e6),
        "Q_peak_W": float(Q.max() if Q.size else 0.0),
        "hours": hours.tolist(),
    }
    return Q, info

def _plot_profiles(hours, q_raw, q_heat, Q, dt_seconds, annual_kWh):
    plt.figure(figsize=(15,6))
    plt.plot(hours, q_raw,  alpha=0.5, label="Bernier Rohprofil (Winter + / Sommer −)")
    plt.plot(hours, q_heat, alpha=0.8, label="Heizen-only unskaliert (Sommer=0)")
    plt.plot(hours, Q,      linewidth=1.2, label="Heizen-only skaliert auf Bedarf")
    plt.xlabel("Zeit [h]"); plt.ylabel("Leistung Q [W]")
    plt.title("Heizlastprofil – Rückspeisung gekappt und auf Jahresbedarf skaliert")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()

    E_kWh = np.cumsum(Q) * dt_seconds / 3.6e6
    plt.figure(figsize=(12,5))
    plt.plot(hours, E_kWh, label="kumulative Energie [kWh]")
    plt.axhline(annual_kWh, linestyle="--", label=f"Bedarf = {annual_kWh:,.0f} kWh".replace(",", "X").replace(".", ",").replace("X", "."))
    plt.xlabel("Zeit [h]"); plt.ylabel("Energie [kWh]")
    plt.title("Kumulative Jahresenergie – Soll/Ist")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import Data_Base as bd
    Q, info = build_heating_only_profile(bd.time, bd.dt, bd.Bedarf)
    print(f"[heating-profile] E_scaled={info['E_scaled_kWh']:.1f} kWh (target={bd.Bedarf:.1f} kWh, "
          f"Δ={(abs(info['E_scaled_kWh']-bd.Bedarf)/max(1e-9,bd.Bedarf))*100:.2f}%) | "
          f"Q_peak={info['Q_peak_W']:.0f} W")
    hrs = np.asarray(info["hours"])
    q_raw  = -_bernier_raw(hrs)                 # konsistent mit Fix
    q_heat = np.where(q_raw > 0.0, q_raw, 0.0)
    _plot_profiles(hrs, q_raw, q_heat, Q, bd.dt, bd.Bedarf)
