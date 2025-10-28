import numpy as np
import pygfunction as gt
from synthetic_heating_profile import build_heating_only_profile

# ===============================
#   Projekt-Eckdaten / Defaults
# ===============================
Bedarf = 20000        # kWh/a (Ziel-Jahresenergie für Heizen)
soil_capacity = 50           # (nur Info) Daumenwert; NICHT zur Feldableitung verwenden!
                             # Anmerkung: Als Heuristik müsste dies kWh/(m·a) sein, nicht W/m.

# -------------------------------
#   Fixes Bohrfeld (WICHTIG!)
# -------------------------------
# Feld NICHT vom Bedarf ableiten. Hier fest vorgeben:
rows_sonden = 0
Columns_sonden = 0
Total_sondas    = rows_sonden * Columns_sonden

# Geometrie
H = 100
D = 1.5                     # m, Tiefe vom Geländeoberfläche bis Rohranfang
r = 0.075                    # m, Bohrlochradius
Bx = 6
By = 6

# Boden / Material
difusivity = 3.735e-7       # m²/s, thermische Diffusivität
conductivity = 2.03         # W/(m·K), falls k_s nicht gesetzt
alpha = difusivity          # Alias
k_s = 2.01                  # W/(m·K), Bodenleitfähigkeit
T_g = 11.5                  # °C, ungestörte Bodentemperatur
k_g = 1.6                   # W/(m·K), Verpressmaterial

# Rohr (Single-U Standard)
rp_out = 0.0211             # m, Außendurchmesser Rohr
rp_in  = 0.0147             # m, Innendurchmesser Rohr
D_s    = 0.052              # m, Schenkeldistanz
epsilon = 1.0e-6            # m, Rauheit
k_p = 0.4                   # W/(m·K), Rohrleitfähigkeit

# Coaxial (für deine anderen Demos)
r_in_in   = 0.0221
r_in_out  = 0.025
r_out_in  = 0.0487
r_out_out = 0.055
r_inner = np.array([r_in_in, r_out_in])
r_outer = np.array([r_in_out, r_out_out])

# Fluid (MEG 25 %)
percent_flow = 25
flow = 301.528                     # kg/s (Demo; bei Feldsimulation eher pro Sonde verteilen)
fluid = gt.media.Fluid('MEG', percent=25)
fluid_isobaric_heatcapacity = fluid.cp
fluid_density = fluid.rho
fluid_viscosity = fluid.mu
fluid_termal_conduct = fluid.k

# -----------------------------
#   Zeitdiskretisierung
# -----------------------------
time_sim = 1                   # Jahre
dt = 3600                      # s
tmax = time_sim * 8760 * dt
number_steps = int(np.ceil(tmax/dt))
time = dt * np.arange(1, number_steps + 1)

# Load Aggregation (für alle Sim-Skripte)
LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)
time_req = LoadAgg.get_times_for_simulation()

# G-Function Zeitraster (für viele deiner Demos)
dt2   = 100*3600.             # s
tmax2 = 3000.*8760.*3600.     # s
Nt    = 25
ts2   = H**2/(9.*alpha)
time2 = gt.utilities.time_geometric(dt2, tmax2, Nt)
lntts = np.log(time2/ts2)

# -----------------------------
#   Positions-Arrays
# -----------------------------
pos_single  = [(-D_s, 0.), (D_s, 0.)]
pos_double  = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]
pos_coaxial = (0., 0.)

# -----------------------------
#   Plot-/Demo-Arrays
# -----------------------------
T_b = np.zeros(number_steps)
T_f_in_single      = np.zeros(number_steps)
T_f_in_double_par  = np.zeros(number_steps)
T_f_out_single     = np.zeros(number_steps)
T_f_out_double_par = np.zeros(number_steps)
nz = 20
it = 8724
z = np.linspace(0., H, num=nz)

# ------------------------------------------
#   (Kompatibilität) Bernier-Rohfunktion
#   -> Für Synthetic_load.py-Plots etc.
# ------------------------------------------
def synthetic_load(x):
    """
    Synthetic load profile of Bernier et al. (2004).
    Returns load y (in watts) at time x (in hours).
    Achtung: Kann +/- sein; NUR für Visualisierung/Alt-Code.
    """
    A = 2000.0; B = 2190.0; C = 80.0; D = 2.0; E = 0.01; F = 0.0; G = 0.95
    func = (168.0 - C)/168.0
    for i in [1, 2, 3]:
        func += 1.0/(i*np.pi)*(np.cos(C*np.pi*i/84.0)-1.0) * (np.sin(np.pi*i/84.0*(x-B)))
    func = func*A*np.sin(np.pi/12.0*(x-B)) * np.sin(np.pi/4380.0*(x-B))
    y = func + (-1.0)**np.floor(D/8760.0*(x-B))*abs(func) \
      + E*(-1.0)**np.floor(D/8760.0*(x-B))/np.sign(np.cos(D*np.pi/4380.0*(x-F))+G)
    return y  # (+/-); unser Heizprofil nutzt diese Funktion NICHT direkt.

# ------------------------------------------
#   HEIZPROFIL (neu & korrekt):
#   - Winter positiv, Sommer (Rückspeisung) = 0
#   - auf Bedarf (kWh/a) skaliert
#   - Q ist Gesamtlast des Gebäudes (feldweit)
# ------------------------------------------
Q, _qinfo = build_heating_only_profile(time, dt, Bedarf)

# Optionaler Konsolencheck der Skalierung
try:
    E_kWh = float(np.sum(Q) * dt / 3.6e6)
    rel = 0.0 if Bedarf == 0 else abs(E_kWh - Bedarf) / Bedarf * 100.0
    print(f"[heating-profile] E_scaled={E_kWh:.1f} kWh "
          f"(target={Bedarf:.1f} kWh, Δ={rel:.2f}%) | "
          f"Q_peak={_qinfo.get('Q_peak_W',0.0):.0f} W")
except Exception:
    pass

# ---------------------------------------------------------
#   Optionen/Methode – für deine anderen Demo-Skripte
# ---------------------------------------------------------
options = {
    'nSegments': 12,
    'segment_ratios': None,
    'disp': True,
    'profiles': True
}
options2 = [
    {'nSegments': 1,  'segment_ratios': None, 'disp': True, 'profiles': True},
    {'nSegments': 12, 'segment_ratios': None, 'disp': True, 'profiles': True},
    {'nSegments': 1,  'segment_ratios': None, 'disp': True, 'profiles': True},
]
method = 'similarities'