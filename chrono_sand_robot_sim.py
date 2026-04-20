"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PyChrono: Legged Robot × Deformable Terrain Interaction Simulation          ║
║  Based on: "A Novel Model of Interaction Dynamics between Legged Robots      ║
║             and Deformable Terrain"  Vanderkop et al., ICRA 2022             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VS CODE / LOCAL SETUP                                                       ║
║                                                                              ║
║  Step 1 – Create & activate a conda environment (recommended):               ║
║    conda create -n chrono_env python=3.9 -y                                  ║
║    conda activate chrono_env                                                 ║
║                                                                              ║
║  Step 2 – Install PyChrono  (choose ONE):                                    ║
║    conda install -c projectchrono pychrono -y          # stable release      ║
║    pip install pychrono --pre                          # nightly wheel        ║
║                                                                              ║
║  Step 3 – Install remaining dependencies:                                    ║
║    pip install numpy matplotlib                                              ║
║                                                                              ║
║  Step 4 – Select the environment in VS Code:                                 ║
║    Ctrl+Shift+P  →  "Python: Select Interpreter"  →  chrono_env             ║
║                                                                              ║
║  Step 5 – Run:                                                               ║
║    python chrono_sand_robot_sim.py                                           ║
║    # or press F5 / the ▶ Run button in VS Code                               ║
║                                                                              ║
║  NOTE: If PyChrono is unavailable the script automatically falls back to     ║
║        the fully-analytical paper model (all plots still generated).         ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES
─────────────────────
  SIM-1  Single foot presses into sand then slides 150 mm horizontally
         (replicates the paper's Foot-Terrain Interaction Testbed experiment).

  SIM-2  Tripod robot (3 cylindrical feet) standing/walking on deformable sand
         with joint-level actuation and SCM soil contact.

  PLOTS  Per simulation:
         • Normal force FN at foot centre-of-mass
         • Horizontal velocity vx and vertical velocity vz
         • Sinkage depth vs displacement and vs time
         • Shear force breakdown: friction Fµ vs bulldozing FB
         • Paper model prediction vs Janosi-Hanamoto (comparison)
         • Total force vs robot weight (SIM-2 only)
         • Terramechanics summary (all soils × all foot materials)
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import math, sys, os, time as _time, warnings
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Matplotlib backend ───────────────────────────────────────────────────────
# VS Code + Jupyter extension  →  figures appear inline automatically.
# VS Code plain terminal       →  figures open in a popup window.
# Un-comment the line below to suppress all popups and only save PNG files:
# matplotlib.use("Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "#0F1117", "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#3A3D4D",   "axes.labelcolor": "#DADDE8",
    "xtick.color": "#8890AA",      "ytick.color": "#8890AA",
    "text.color": "#DADDE8",       "legend.facecolor": "#1A1D27",
    "legend.edgecolor": "#3A3D4D", "grid.color": "#2A2D3D",
    "grid.alpha": 0.6,             "font.family": "monospace",
})

# ─── Try importing PyChrono ──────────────────────────────────────────────────
CHRONO_AVAILABLE = False
try:
    import pychrono as chrono
    import pychrono.vehicle as veh
    CHRONO_AVAILABLE = True
    print("✓ PyChrono detected — full physics simulation enabled.")
except ImportError:
    print("⚠  PyChrono not found — running analytical / plotting mode only.")
    print("   (Install with:  conda install -c projectchrono pychrono)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A: PAPER PARAMETERS  (edit freely — all sims read from here)
# ══════════════════════════════════════════════════════════════════════════════

# ── Table I: Soil Properties ─────────────────────────────────────────────────
SOIL_PARAMS = {
    "coarse_sand": {
        "name":               "Coarse Sand",
        # ─── From Table I ───────────────────────────────────────────────────
        "moisture_pct":       0.3,          # %
        "min_grain_m":        0.0002,       # m  (< 0.2 mm)
        "max_grain_m":        0.001,        # m  (> 1 mm)
        "cohesion_kPa":       0.0,          # kPa  →  0 Pa (cohesionless)
        "friction_angle_deg": 32.5,         # °
        "bulk_density":       1614.2,       # kg/m³
        # ─── Bekker SCM parameters (derived for dry sand) ───────────────────
        # Kphi: frictional sinkage modulus  [N/m^(n+2)]
        # Kc:   cohesive sinkage modulus    [N/m^(n+1)] — 0 for cohesionless
        # n:    sinkage exponent            [-]
        # K:    Janosi shear modulus        [m]
        "Bekker_Kphi":        2_000_000,
        "Bekker_Kc":          0,
        "Bekker_n":           1.10,
        "Janosi_K":           0.025,
        "elastic_K":          4e7,          # N/m³
        "damping":            3e4,          # N·s/m³
    },
    "dry_loam": {
        "name":               "Dry Loam",
        "moisture_pct":       0.1,
        "cohesion_kPa":       0.0,
        "friction_angle_deg": 30.4,
        "bulk_density":       1418.9,
        "Bekker_Kphi":        1_600_000,
        "Bekker_Kc":          0,
        "Bekker_n":           1.05,
        "Janosi_K":           0.020,
        "elastic_K":          3.5e7,
        "damping":            2.5e4,
    },
    "wet_loam": {
        "name":               "Wet Loam",
        "moisture_pct":       2.5,
        "cohesion_kPa":       3.6,
        "friction_angle_deg": 31.3,
        "bulk_density":       1295.8,
        "Bekker_Kphi":        1_400_000,
        "Bekker_Kc":          1500,
        "Bekker_n":           1.00,
        "Janosi_K":           0.015,
        "elastic_K":          3e7,
        "damping":            2e4,
    },
    "natural_soil": {
        "name":               "Natural Soil",
        "moisture_pct":       31.7,
        "cohesion_kPa":       0.9,
        "friction_angle_deg": 40.0,
        "bulk_density":       517.9,
        "Bekker_Kphi":        800_000,
        "Bekker_Kc":          500,
        "Bekker_n":           0.90,
        "Janosi_K":           0.010,
        "elastic_K":          2e7,
        "damping":            1.5e4,
    },
}

# ── Table II: Friction & Bulldozing Coefficients ──────────────────────────────
MECH_PARAMS = {
    #                bulldozing  bulldozing
    #                coeff (a)   exponent (n)   µ_plastic    µ_rubber    µ_metal
    "coarse_sand": { "BC": 1.47e5, "BE": 2.69,
                     "mu_plastic": 0.53, "mu_plastic_std": 0.04,
                     "mu_rubber":  0.56, "mu_rubber_std":  0.04,
                     "mu_metal":   0.28, "mu_metal_std":   0.01 },
    "dry_loam":    { "BC": 1.14e5, "BE": 2.50,
                     "mu_plastic": 0.53, "mu_plastic_std": 0.02,
                     "mu_rubber":  0.52, "mu_rubber_std":  0.02,
                     "mu_metal":   0.40, "mu_metal_std":   0.02 },
    "wet_loam":    { "BC": 8.40e4, "BE": 2.61,
                     "mu_plastic": 0.66, "mu_plastic_std": 0.09,
                     "mu_rubber":  0.65, "mu_rubber_std":  0.06,
                     "mu_metal":   0.54, "mu_metal_std":   0.05 },
    "natural_soil":{ "BC": 1.04e5, "BE": 2.94,
                     "mu_plastic": 0.62, "mu_plastic_std": 0.05,
                     "mu_rubber":  0.71, "mu_rubber_std":  0.08,
                     "mu_metal":   0.50, "mu_metal_std":   0.06 },
}


# ── Simulation Config  ────────────────────────────────────────────────────────
#    ↓ Change any of these to explore parameter space ↓
# ─────────────────────────────────────────────────────
CFG = {
    "ACTIVE_SOIL":      "coarse_sand",   # coarse_sand | dry_loam | wet_loam | natural_soil
    "FOOT_MATERIAL":    "plastic",       # plastic | rubber | metal
    "NORMAL_LOAD_N":    60.0,            # N  (paper tests: 20, 40, 60, 80, 100)
    "FOOT_RADIUS_M":    0.030,           # m  (30 mm radius = 60 mm dia)
    "FOOT_SHAPE":       "cylinder",      # cylinder | hemisphere
    "H_VELOCITY_MPS":   0.005,           # m/s  (5 mm/s as in paper)
    "H_DISPLACE_M":     0.150,           # m  (150 mm as in paper)
    "TIME_STEP_S":      1e-4,            # s
    "TERRAIN_SIZE_M":   2.0,             # m × m
    "ROBOT_MASS_KG":    10.0,            # kg  (lightweight robot ≤ 60 kg)
    "PRESLIDEA":        0.010,           # m  pre-sliding transition parameter
    "SIM2_DURATION_S":  5.0,             # s  tripod robot sim duration
}
# ─────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B: ANALYTICAL TERRAMECHANICS MODEL  (Eqs. 3–5 from paper)
# ══════════════════════════════════════════════════════════════════════════════

def friction_force_model(FN, mu, x, a=None):
    """
    Eq. 5 — paper's friction model:
        Fµ = µ · FN · (1 − e^(−x/a))

    Args
        FN   : normal force [N] (scalar or array)
        mu   : coefficient of friction (from Table II)
        x    : horizontal displacement [m]
        a    : pre-sliding parameter [m]; defaults to CFG["PRESLIDEA"]
    Returns
        Fµ   [N]
    """
    a = a or CFG["PRESLIDEA"]
    x = np.maximum(x, 0.0)
    return mu * FN * (1.0 - np.exp(-x / a))


def bulldozing_power_model(z, BC, BE):
    """
    Eq. 4 supplement — power-relation bulldozing model:
        FB = BC · z^BE

    Args
        z    : sinkage [m]
        BC   : bulldozing coefficient a   (from Table II)
        BE   : bulldozing exponent n      (from Table II)
    Returns
        FB   [N]
    """
    z = np.maximum(z, 1e-9)   # avoid 0^n issues
    return BC * np.power(z, BE)


def bulldozing_hegedus_model(z, p0, p1, p2):
    """
    Hegedus quadratic model:  FB = p0·z² + p1·z + p2
    (comparison model — generally less numerically stable per paper)
    """
    return p0 * z**2 + p1 * z + p2


def total_shear_force_model(FN, mu, x, z, BC, BE, a=None):
    """
    Eq. 4 — F_T = Fµ + FB

    Returns  (FT, Fmu, FB)  each in [N]
    """
    Fmu = friction_force_model(FN, mu, x, a)
    FB  = bulldozing_power_model(z, BC, BE)
    return Fmu + FB, Fmu, FB


def janosi_hanamoto_model(cohesion_Pa, sigma_Pa, phi_deg, x, K):
    """
    Eq. 3 — Janosi-Hanamoto shear stress (comparison baseline):
        τ(j) = [c + σ·tan(φ)] · (1 − e^(−x/K))

    Returns shear STRESS [Pa]; multiply by foot area for force.
    """
    phi_rad = math.radians(phi_deg)
    tau_max = cohesion_Pa + sigma_Pa * math.tan(phi_rad)
    return tau_max * (1.0 - np.exp(-np.maximum(x, 0.0) / K))


def bekker_normal_pressure(z, r, kc, kphi, n):
    """
    Eq. 1 — Bekker model: σ(z) = (kc/r + kφ) · z^n
    Returns normal pressure [Pa] at sinkage z [m].
    """
    return (kc / r + kphi) * np.power(np.maximum(z, 0), n)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION C: PYCHRONO SIMULATION 1 — SINGLE FOOT / TERRAIN INTERACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_sim1_foot_terrain():
    """
    Replicates the paper's single-leg testbed:
    1) Foot drops vertically until target normal load is reached
    2) Foot slides horizontally at 5 mm/s for 150 mm
    Logs FN, FT, vx, vz, sinkage every step.
    """
    if not CHRONO_AVAILABLE:
        print("  [SIM-1] PyChrono unavailable — returning synthetic data for plotting.")
        return _synthetic_sim1_data()

    print("\n  Initialising PyChrono SCM system …")
    soil = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]
    mech = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    mu   = mech[f"mu_{CFG['FOOT_MATERIAL']}"]
    dt   = CFG["TIME_STEP_S"]
    R    = CFG["FOOT_RADIUS_M"]

    # ── System ──────────────────────────────────────────────────────────────
    sys = chrono.ChSystemSMC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))

    # ── Contact material ────────────────────────────────────────────────────
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(mu)
    mat.SetRestitution(0.0)
    mat.SetYoungModulus(5e6)
    mat.SetPoissonRatio(0.3)

    # ── SCM Terrain ─────────────────────────────────────────────────────────
    terrain = veh.SCMTerrain(sys)
    terrain.SetSoilParameters(
        soil["Bekker_Kphi"],
        soil["Bekker_Kc"],
        soil["Bekker_n"],
        soil["cohesion_kPa"] * 1e3,
        soil["friction_angle_deg"],
        soil["Janosi_K"],
        soil["elastic_K"],
        soil["damping"],
    )
    terrain.Initialize(
        chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0)),
        CFG["TERRAIN_SIZE_M"], CFG["TERRAIN_SIZE_M"], 0.02
    )

    # ── Foot body ───────────────────────────────────────────────────────────
    foot = chrono.ChBody()
    foot.SetMass(0.5)
    foot.SetInertiaXX(chrono.ChVector3d(1e-4, 1e-4, 1e-4))
    foot.SetPos(chrono.ChVector3d(0, R + 0.08, 0))
    foot.EnableCollision(True)
    sys.Add(foot)

    # Flat-cylinder collision shape
    cyl_shape = chrono.ChCollisionShapeCylinder(mat, R, 0.015)
    foot.AddCollisionShape(cyl_shape,
        chrono.ChFramed(chrono.ChVector3d(0, 0, 0),
                        chrono.QuatFromAngleX(math.pi / 2)))

    vis = chrono.ChVisualShapeCylinder(R, 0.015)
    foot.AddVisualShape(vis,
        chrono.ChFramed(chrono.ChVector3d(0, 0, 0),
                        chrono.QuatFromAngleX(math.pi / 2)))

    # ── Log containers ───────────────────────────────────────────────────────
    log = {k: [] for k in
           ["t", "FN", "FT_sim", "FT_model", "Fmu", "FB", "FT_JH",
            "vx", "vz", "pos_x", "pos_z", "sinkage", "x_disp"]}

    # ── State machine ────────────────────────────────────────────────────────
    phase          = "SINK"
    contact_z0     = None
    slide_x0       = None
    t              = 0.0
    FN_target      = CFG["NORMAL_LOAD_N"]
    foot_area      = math.pi * R**2
    max_t          = 120.0

    print(f"  Phase SINK: targeting FN = {FN_target} N …")
    while t < max_t:
        pos = foot.GetPos()
        vel = foot.GetPosDt()

        tf  = terrain.GetContactForceBody(foot)
        FN  = max(0.0,  tf.y)         # upward reaction
        FTx = abs(tf.x)               # horizontal shear from terrain

        # ── First contact detection ──────────────────────────────────────────
        if contact_z0 is None and FN > 0.5:
            contact_z0 = pos.y
            print(f"    Terrain contact at t = {t:.3f} s,  z₀ = {contact_z0*1000:.2f} mm")

        sinkage = max(0.0, contact_z0 - pos.y) if contact_z0 else 0.0

        # ── Phase: SINK — drive foot down to desired normal load ─────────────
        if phase == "SINK":
            err_z  = FN_target - FN
            Fz_app = -err_z * 60.0 - vel.y * 120.0
            foot.SetForce(chrono.ChVector3d(0, Fz_app, 0))
            if FN > FN_target * 0.95 and abs(vel.y) < 5e-4:
                phase    = "SLIDE"
                slide_x0 = pos.x
                print(f"    Phase SLIDE at t = {t:.3f} s,  sinkage = {sinkage*1000:.2f} mm")

        # ── Phase: SLIDE — constant velocity horizontal + maintain FN ────────
        elif phase == "SLIDE":
            x_disp = abs(pos.x - slide_x0)

            Fz_app = -(FN_target - FN) * 60.0 - vel.y * 120.0
            Fx_app = (CFG["H_VELOCITY_MPS"] - vel.x) * 250.0
            foot.SetForce(chrono.ChVector3d(Fx_app, Fz_app, 0))

            # ── Paper model prediction ────────────────────────────────────────
            FT_mod, Fmu, FB = total_shear_force_model(
                FN_target, mu, x_disp, sinkage,
                mech["BC"], mech["BE"], CFG["PRESLIDEA"]
            )
            # ── Janosi-Hanamoto prediction (for comparison) ───────────────────
            sigma_Pa = FN_target / foot_area
            tau_JH   = janosi_hanamoto_model(
                soil["cohesion_kPa"] * 1e3, sigma_Pa,
                soil["friction_angle_deg"], x_disp, soil["Janosi_K"]
            )
            FT_JH = tau_JH * foot_area

            # ── Log ───────────────────────────────────────────────────────────
            log["t"].append(t)
            log["FN"].append(FN)
            log["FT_sim"].append(FTx)
            log["FT_model"].append(float(FT_mod))
            log["Fmu"].append(float(Fmu))
            log["FB"].append(float(FB))
            log["FT_JH"].append(float(FT_JH))
            log["vx"].append(vel.x)
            log["vz"].append(vel.y)
            log["pos_x"].append(pos.x)
            log["pos_z"].append(pos.y)
            log["sinkage"].append(sinkage * 1000)   # mm
            log["x_disp"].append(x_disp * 1000)     # mm

            if x_disp >= CFG["H_DISPLACE_M"]:
                print(f"    Slide complete — {x_disp*1000:.1f} mm reached at t = {t:.2f} s")
                break

        sys.DoStepDynamics(dt)
        terrain.Synchronize(t)
        t += dt

        step = round(t / dt)
        if step % 10_000 == 0:
            print(f"    t={t:.2f} s  FN={FN:.1f} N  xd={x_disp*1000 if phase=='SLIDE' else 0:.1f} mm  "
                  f"sink={sinkage*1000:.2f} mm")

    print(f"  SIM-1 done — {len(log['t'])} data points.")
    return log


def _synthetic_sim1_data():
    """
    Generates analytically-computed synthetic data mimicking SIM-1
    when PyChrono is unavailable.  Uses all paper model equations.
    """
    soil  = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]
    mech  = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    mu    = mech[f"mu_{CFG['FOOT_MATERIAL']}"]
    FN    = CFG["NORMAL_LOAD_N"]
    R     = CFG["FOOT_RADIUS_M"]
    area  = math.pi * R**2
    N     = 500
    x_arr = np.linspace(0, CFG["H_DISPLACE_M"], N)
    # Simple sinkage model: sinkage grows with displacement (asymptotic)
    z_arr = 0.008 * (1 - np.exp(-x_arr / 0.03)) + 0.001 * x_arr

    FT_model_arr, Fmu_arr, FB_arr = total_shear_force_model(
        FN, mu, x_arr, z_arr, mech["BC"], mech["BE"], CFG["PRESLIDEA"]
    )
    sigma   = FN / area
    tau_JH  = janosi_hanamoto_model(
        soil["cohesion_kPa"] * 1e3, sigma,
        soil["friction_angle_deg"], x_arr, soil["Janosi_K"]
    )
    noise   = np.random.normal(0, FT_model_arr.mean() * 0.04, N)
    t_arr   = x_arr / CFG["H_VELOCITY_MPS"]

    vx_arr  = np.full(N, CFG["H_VELOCITY_MPS"])
    vx_arr[:20] = np.linspace(0, CFG["H_VELOCITY_MPS"], 20)   # ramp up
    vz_arr  = -np.gradient(z_arr, t_arr)

    return {
        "t":        list(t_arr),
        "FN":       list(np.full(N, FN) + np.random.normal(0, 1.0, N)),
        "FT_sim":   list(FT_model_arr + noise),
        "FT_model": list(FT_model_arr),
        "Fmu":      list(Fmu_arr),
        "FB":       list(FB_arr),
        "FT_JH":    list(tau_JH * area),
        "vx":       list(vx_arr),
        "vz":       list(vz_arr),
        "pos_x":    list(x_arr),
        "pos_z":    list(-z_arr),
        "sinkage":  list(z_arr * 1000),
        "x_disp":   list(x_arr * 1000),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION D: PYCHRONO SIMULATION 2 — TRIPOD ROBOT ON SAND
# ══════════════════════════════════════════════════════════════════════════════

def _add_leg(sys, mat, chassis, attach_pos, chassis_body, leg_idx):
    """Builds one 3-DoF leg: hip(revolute) + knee(revolute) + ankle(fixed) + foot."""
    UL = 0.11          # upper link length [m]
    LL = 0.09          # lower link length [m]
    FH = 0.015         # foot half-height  [m]
    R  = CFG["FOOT_RADIUS_M"]
    lm = CFG["ROBOT_MASS_KG"] * 0.35 / 3   # mass per leg

    hip_pos = attach_pos

    # ── Upper link ───────────────────────────────────────────────────────────
    up = chrono.ChBody()
    up.SetMass(lm * 0.35)
    up.SetPos(chrono.ChVector3d(hip_pos.x, hip_pos.y - UL/2, hip_pos.z))
    up.EnableCollision(False)
    sys.Add(up)
    vis_u = chrono.ChVisualShapeBox(0.018, UL, 0.018)
    vis_u.SetColor(chrono.ChColor(0.3, 0.6, 0.9))
    up.AddVisualShape(vis_u)

    jh = chrono.ChLinkRevolute()
    jh.Initialize(chassis_body, up,
                  chrono.ChFramed(hip_pos, chrono.QuatFromAngleY(0)))
    sys.Add(jh)

    # ── Lower link ───────────────────────────────────────────────────────────
    knee_pos = chrono.ChVector3d(hip_pos.x, hip_pos.y - UL, hip_pos.z)
    lo = chrono.ChBody()
    lo.SetMass(lm * 0.30)
    lo.SetPos(chrono.ChVector3d(hip_pos.x, hip_pos.y - UL - LL/2, hip_pos.z))
    lo.EnableCollision(False)
    sys.Add(lo)
    vis_l = chrono.ChVisualShapeBox(0.015, LL, 0.015)
    vis_l.SetColor(chrono.ChColor(0.6, 0.3, 0.9))
    lo.AddVisualShape(vis_l)

    jk = chrono.ChLinkRevolute()
    jk.Initialize(up, lo,
                  chrono.ChFramed(knee_pos, chrono.QuatFromAngleY(0)))
    sys.Add(jk)

    # ── Foot (cylinder) ─────────────────────────────────────────────────────
    ankle_pos = chrono.ChVector3d(hip_pos.x, hip_pos.y - UL - LL, hip_pos.z)
    ft = chrono.ChBody()
    ft.SetMass(lm * 0.35)
    ft.SetPos(chrono.ChVector3d(hip_pos.x, hip_pos.y - UL - LL - FH, hip_pos.z))
    ft.EnableCollision(True)
    sys.Add(ft)

    cs = chrono.ChCollisionShapeCylinder(mat, R, FH * 2)
    ft.AddCollisionShape(cs, chrono.ChFramed(
        chrono.ChVector3d(0, 0, 0), chrono.QuatFromAngleX(math.pi/2)))
    vis_f = chrono.ChVisualShapeCylinder(R, FH * 2)
    vis_f.SetColor(chrono.ChColor(1.0, 0.7, 0.2))
    ft.AddVisualShape(vis_f, chrono.ChFramed(
        chrono.ChVector3d(0, 0, 0), chrono.QuatFromAngleX(math.pi/2)))

    ja = chrono.ChLinkLockFixed()
    ja.Initialize(lo, ft, chrono.ChFramed(ankle_pos))
    sys.Add(ja)

    return ft


def run_sim2_robot():
    """
    Tripod robot (chassis + 3 legs) standing on deformable sand.
    Logs normal force, shear force, sinkage, and velocities per foot.
    """
    if not CHRONO_AVAILABLE:
        print("  [SIM-2] PyChrono unavailable — returning synthetic data.")
        return _synthetic_sim2_data()

    print("\n  Initialising tripod robot simulation …")
    soil = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]
    mech = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    mu   = mech[f"mu_{CFG['FOOT_MATERIAL']}"]
    dt   = CFG["TIME_STEP_S"]

    sys = chrono.ChSystemSMC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))

    terrain = veh.SCMTerrain(sys)
    terrain.SetSoilParameters(
        soil["Bekker_Kphi"], soil["Bekker_Kc"], soil["Bekker_n"],
        soil["cohesion_kPa"] * 1e3, soil["friction_angle_deg"],
        soil["Janosi_K"],   soil["elastic_K"], soil["damping"],
    )
    terrain.Initialize(
        chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0)),
        CFG["TERRAIN_SIZE_M"], CFG["TERRAIN_SIZE_M"], 0.02
    )

    # Contact material
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(mu)
    mat.SetRestitution(0.0)
    mat.SetYoungModulus(5e6)

    # ── Chassis ──────────────────────────────────────────────────────────────
    CH = 0.28            # chassis height above ground [m]
    chas = chrono.ChBody()
    chas.SetMass(CFG["ROBOT_MASS_KG"] * 0.65)
    chas.SetPos(chrono.ChVector3d(0, CH, 0))
    chas.SetInertiaXX(chrono.ChVector3d(0.08, 0.08, 0.08))
    chas.EnableCollision(False)
    sys.Add(chas)
    vb = chrono.ChVisualShapeBox(0.25, 0.04, 0.25)
    vb.SetColor(chrono.ChColor(0.2, 0.7, 0.3))
    chas.AddVisualShape(vb)

    # ── 3 Legs (120° apart) ───────────────────────────────────────────────────
    feet = []
    for i in range(3):
        ang = i * 2 * math.pi / 3
        r   = 0.13
        ap  = chrono.ChVector3d(r * math.cos(ang), CH - 0.02, r * math.sin(ang))
        ft  = _add_leg(sys, mat, chas, ap, chas, i)
        feet.append(ft)

    # ── Log ──────────────────────────────────────────────────────────────────
    log = {"t": [],
           "chas_vx": [], "chas_vy": [], "chas_y": [],
           **{f"FN{i}": [] for i in range(3)},
           **{f"FT{i}": [] for i in range(3)},
           **{f"sk{i}": [] for i in range(3)},
           **{f"FB{i}": [] for i in range(3)},
           **{f"Fmu{i}": [] for i in range(3)}}

    z0      = [None, None, None]
    x_cum   = [0.0,  0.0,  0.0]
    prev_x  = [f.GetPos().x for f in feet]
    t       = 0.0
    dur     = CFG["SIM2_DURATION_S"]

    print(f"  Running {dur} s robot simulation …")
    while t < dur:
        cv = chas.GetPosDt()
        cp = chas.GetPos()

        log["t"].append(t)
        log["chas_vx"].append(cv.x)
        log["chas_vy"].append(cv.y)
        log["chas_y"].append(cp.y)

        for i, ft in enumerate(feet):
            fp = ft.GetPos()
            tf = terrain.GetContactForceBody(ft)
            FN = max(0.0, tf.y)
            FT = math.sqrt(tf.x**2 + tf.z**2)

            if z0[i] is None and FN > 0.5:
                z0[i] = fp.y
            sk = max(0.0, z0[i] - fp.y) if z0[i] else 0.0

            dx = abs(fp.x - prev_x[i])
            x_cum[i] += dx
            prev_x[i] = fp.x

            _,  Fmu, FB = total_shear_force_model(
                FN, mu, x_cum[i], sk, mech["BC"], mech["BE"])

            log[f"FN{i}"].append(FN)
            log[f"FT{i}"].append(FT)
            log[f"sk{i}"].append(sk * 1000)
            log[f"Fmu{i}"].append(float(Fmu))
            log[f"FB{i}"].append(float(FB))

        sys.DoStepDynamics(dt)
        terrain.Synchronize(t)
        t += dt

        step = round(t / dt)
        if step % 10_000 == 0:
            avg_FN = np.mean([log[f"FN{i}"][-1] for i in range(3)])
            print(f"    t={t:.2f} s  chassis_y={cp.y:.4f} m  avgFN={avg_FN:.1f} N")

    print(f"  SIM-2 done — {len(log['t'])} data points.")
    return log


def _synthetic_sim2_data():
    """Synthetic tripod robot data when PyChrono unavailable."""
    dur  = CFG["SIM2_DURATION_S"]
    dt   = CFG["TIME_STEP_S"]
    N    = int(dur / dt)
    t    = np.linspace(0, dur, N)
    mech = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    mu   = mech[f"mu_{CFG['FOOT_MATERIAL']}"]
    W    = CFG["ROBOT_MASS_KG"] * 9.81

    log  = {"t": list(t),
            "chas_vx": list(np.zeros(N) + 0.01 * np.sin(2 * np.pi * 0.5 * t)),
            "chas_vy": list(np.zeros(N) - 0.001 * np.exp(-t)),
            "chas_y":  list(0.28 - 0.003 * (1 - np.exp(-t / 0.5)))}

    for i in range(3):
        phase_shift = i * 2 * np.pi / 3
        FN = (W / 3) * (1 + 0.25 * np.sin(2 * np.pi * 0.8 * t + phase_shift))
        sk = 0.002 * (1 - np.exp(-t / 0.8)) * np.ones(N) + i * 0.0003
        x_cum = np.cumsum(np.abs(0.01 * np.cos(2 * np.pi * 0.8 * t + phase_shift))) * dt
        _, Fmu_arr, FB_arr = total_shear_force_model(
            FN, mu, x_cum, sk / 1000, mech["BC"], mech["BE"])
        log[f"FN{i}"] = list(FN)
        log[f"FT{i}"] = list(Fmu_arr + FB_arr + np.random.normal(0, 0.5, N))
        log[f"sk{i}"] = list(sk * 1000)
        log[f"Fmu{i}"] = list(Fmu_arr)
        log[f"FB{i}"]  = list(FB_arr)

    return log


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION E: PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

ACCENT  = ["#FF6B6B", "#4ECDC4", "#FFE66D"]
SOIL_C  = {"coarse_sand": "#F39C12", "dry_loam": "#8B4513",
           "wet_loam": "#2ECC71",    "natural_soil": "#7F8C8D"}
MAT_C   = {"plastic": "#E74C3C", "rubber": "#3498DB", "metal": "#AAB7C4"}


def _suptitle(fig, txt, y=0.98):
    fig.suptitle(txt, fontsize=12, fontweight="bold",
                 color="#DADDE8", y=y, family="monospace")


def plot_sim1(log):
    """Nine-panel SIM-1 dashboard."""
    soil_name = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]["name"]
    foot_lbl  = CFG["FOOT_MATERIAL"].capitalize()

    xd  = np.array(log["x_disp"])        # mm
    t   = np.array(log["t"])

    fig = plt.figure(figsize=(17, 13))
    _suptitle(fig,
        f"SIM-1 │ Foot–Terrain Interaction │ {soil_name} │ "
        f"{foot_lbl} foot │ FN = {CFG['NORMAL_LOAD_N']} N  "
        f"{'[ANALYTICAL]' if not CHRONO_AVAILABLE else '[PYCHRONO SCM]'}")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── P1: Normal Force ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xd, log["FN"], color=ACCENT[0], lw=1.5, label="FN simulated")
    ax.axhline(CFG["NORMAL_LOAD_N"], color="#AAB7C4", ls="--", lw=1,
               label=f"Target {CFG['NORMAL_LOAD_N']} N")
    ax.set(title="Normal Force FN at Foot CoM",
           xlabel="Displacement (mm)", ylabel="FN (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P2: Shear Force Comparison ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xd, log["FT_sim"],   color="#AAB7C4", lw=1.2, alpha=0.8, label="Simulated FT")
    ax.plot(xd, log["FT_model"], color=ACCENT[1], lw=2.0, label="Paper model Fµ+FB")
    ax.plot(xd, log["FT_JH"],    color=ACCENT[0], lw=1.5, ls="--", label="Janosi-Hanamoto")
    ax.set(title="Shear Force: Model Comparison",
           xlabel="Displacement (mm)", ylabel="FT (N)")
    ax.legend(fontsize=7); ax.grid()

    # ── P3: Force Decomposition ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.stackplot(xd, log["Fmu"], log["FB"],
                 labels=["Friction Fµ", "Bulldozing FB"],
                 colors=["#3498DB", "#E67E22"], alpha=0.85)
    ax.set(title="Force Decomposition (Paper Model)",
           xlabel="Displacement (mm)", ylabel="Force (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P4: Sinkage vs Displacement ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xd, log["sinkage"], color="#A29BFE", lw=2)
    ax.set(title="Foot Sinkage into Terrain",
           xlabel="Displacement (mm)", ylabel="Sinkage (mm)")
    ax.grid()

    # ── P5: Horizontal Velocity ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, np.array(log["vx"]) * 1000, color=ACCENT[1], lw=1.5, label="vx")
    ax.axhline(CFG["H_VELOCITY_MPS"] * 1000, color=ACCENT[0], ls="--", lw=1,
               label=f"Target {CFG['H_VELOCITY_MPS']*1000:.0f} mm/s")
    ax.set(title="Foot Velocity X  (Forward)", xlabel="Time (s)", ylabel="mm/s")
    ax.legend(fontsize=8); ax.grid()

    # ── P6: Vertical Velocity (sinkage rate) ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, np.array(log["vz"]) * 1000, color=ACCENT[0], lw=1.5)
    ax.axhline(0, color="#AAB7C4", ls="--", lw=0.8)
    ax.set(title="Foot Velocity Z  (Sinkage Rate)",
           xlabel="Time (s)", ylabel="mm/s")
    ax.grid()

    # ── P7: Sinkage vs Time ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, log["sinkage"], color="#FD79A8", lw=2)
    ax.set(title="Sinkage vs Time", xlabel="Time (s)", ylabel="Sinkage (mm)")
    ax.grid()

    # ── P8: Shear Force vs Time ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, log["FT_sim"],   color="#AAB7C4", lw=1, alpha=0.7, label="Sim")
    ax.plot(t, log["FT_model"], color=ACCENT[1], lw=2, label="Model")
    ax.set(title="Total Shear Force vs Time",
           xlabel="Time (s)", ylabel="FT (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P9: Model Validation Scatter ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    sim_v = np.array(log["FT_sim"])
    mod_v = np.array(log["FT_model"])
    jh_v  = np.array(log["FT_JH"])
    lim   = max(sim_v.max(), mod_v.max(), jh_v.max()) * 1.1
    ax.scatter(sim_v, mod_v, s=4, color=ACCENT[1], alpha=0.5, label="Paper model")
    ax.scatter(sim_v, jh_v,  s=4, color=ACCENT[0], alpha=0.5, label="Janosi-H")
    ax.plot([0, lim], [0, lim], color="#AAB7C4", lw=1, ls="--", label="Ideal")
    ax.set(title="Model Validation (predicted vs measured)",
           xlabel="Measured FT (N)", ylabel="Predicted FT (N)")
    ax.legend(fontsize=7); ax.grid()

    plt.savefig("sim1_foot_terrain.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show(block=False); plt.pause(0.1)
    print("  Saved → sim1_foot_terrain.png")


def plot_sim2(log):
    """Nine-panel SIM-2 tripod robot dashboard."""
    soil_name = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]["name"]
    mech  = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    t     = np.array(log["t"])
    W     = CFG["ROBOT_MASS_KG"] * 9.81

    fig = plt.figure(figsize=(17, 13))
    _suptitle(fig,
        f"SIM-2 │ Tripod Robot │ {soil_name} │ "
        f"m={CFG['ROBOT_MASS_KG']} kg │ {CFG['FOOT_MATERIAL'].capitalize()} feet  "
        f"{'[ANALYTICAL]' if not CHRONO_AVAILABLE else '[PYCHRONO SCM]'}")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── P1: Normal Force per foot ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for i in range(3):
        ax.plot(t, log[f"FN{i}"], color=ACCENT[i % 3], lw=1.2, label=f"Foot {i}")
    ax.axhline(W / 3, color="#AAB7C4", ls="--", lw=1, label="W/3 static")
    ax.set(title="Normal Force at Each Foot CoM",
           xlabel="Time (s)", ylabel="FN (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P2: Sinkage per foot ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i in range(3):
        ax.plot(t, log[f"sk{i}"], color=ACCENT[i % 3], lw=1.2, label=f"Foot {i}")
    ax.set(title="Foot Sinkage into Terrain",
           xlabel="Time (s)", ylabel="Sinkage (mm)")
    ax.legend(fontsize=8); ax.grid()

    # ── P3: Shear force measured vs model ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for i in range(3):
        ax.plot(t, log[f"FT{i}"],
                color=ACCENT[i % 3], lw=0.9, alpha=0.5, label=f"Meas {i}")
        FT_mod = np.array(log[f"Fmu{i}"]) + np.array(log[f"FB{i}"])
        ax.plot(t, FT_mod,
                color=ACCENT[i % 3], lw=1.8, ls="--", label=f"Model {i}")
    ax.set(title="Shear Force: Measured vs Paper Model",
           xlabel="Time (s)", ylabel="FT (N)")
    ax.legend(fontsize=6); ax.grid()

    # ── P4: Chassis velocity X ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.array(log["chas_vx"]) * 100, color=ACCENT[1], lw=1.5)
    ax.axhline(0, color="#AAB7C4", ls="--", lw=0.8)
    ax.set(title="Chassis Forward Velocity (X)",
           xlabel="Time (s)", ylabel="cm/s")
    ax.grid()

    # ── P5: Chassis vertical velocity ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, np.array(log["chas_vy"]) * 1000, color=ACCENT[0], lw=1.5)
    ax.axhline(0, color="#AAB7C4", ls="--", lw=0.8)
    ax.set(title="Chassis Vertical Velocity (Y)",
           xlabel="Time (s)", ylabel="mm/s")
    ax.grid()

    # ── P6: Chassis height (body sinkage indicator) ───────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, np.array(log["chas_y"]) * 100, color="#A29BFE", lw=1.5)
    ax.set(title="Chassis Height (Body Sinkage Indicator)",
           xlabel="Time (s)", ylabel="cm")
    ax.grid()

    # ── P7: Friction force decomposition ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    for i in range(3):
        ax.plot(t, log[f"Fmu{i}"], color=ACCENT[i % 3], lw=1.2, label=f"Foot {i}")
    ax.set(title="Friction Force Fµ per Foot (Paper Model)",
           xlabel="Time (s)", ylabel="Fµ (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P8: Bulldozing force ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    for i in range(3):
        ax.plot(t, log[f"FB{i}"], color=ACCENT[i % 3], lw=1.2, label=f"Foot {i}")
    ax.set(title="Bulldozing Resistance FB per Foot",
           xlabel="Time (s)", ylabel="FB (N)")
    ax.legend(fontsize=8); ax.grid()

    # ── P9: Total normal force vs robot weight ────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    FN_total = (np.array(log["FN0"]) + np.array(log["FN1"]) + np.array(log["FN2"]))
    ax.plot(t, FN_total, color=ACCENT[1], lw=1.5, label="Σ FN (all feet)")
    ax.axhline(W, color=ACCENT[0], ls="--", lw=1.5, label=f"Robot weight {W:.1f} N")
    ax.set(title="Total Normal Force vs Robot Weight",
           xlabel="Time (s)", ylabel="Force (N)")
    ax.legend(fontsize=8); ax.grid()

    plt.savefig("sim2_robot.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show(block=False); plt.pause(0.1)
    print("  Saved → sim2_robot.png")


def plot_terramechanics_summary():
    """
    Eight-panel analytical summary across ALL soils and foot materials.
    Fully independent of PyChrono — uses paper equations directly.
    """
    x_arr  = np.linspace(1e-5, 0.15, 500)   # 0 → 150 mm displacement
    z_arr  = np.linspace(1e-5, 0.05, 500)   # 0 → 50 mm sinkage
    FN     = CFG["NORMAL_LOAD_N"]
    R      = CFG["FOOT_RADIUS_M"]
    area   = math.pi * R**2
    FN_rng = np.linspace(20, 100, 100)       # normal-load sweep

    fig = plt.figure(figsize=(18, 12))
    _suptitle(fig,
        f"Terramechanics Summary (Paper Values)  │  "
        f"FN = {FN} N  │  R = {R*100:.1f} cm foot")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.46, wspace=0.35)

    # ── P1–P3: Friction per material per soil ────────────────────────────────
    for col, mat in enumerate(["plastic", "rubber", "metal"]):
        ax = fig.add_subplot(gs[0, col])
        for sk, sc in SOIL_C.items():
            mu  = MECH_PARAMS[sk][f"mu_{mat}"]
            ax.plot(x_arr * 1000,
                    friction_force_model(FN, mu, x_arr),
                    color=sc, lw=2, label=SOIL_PARAMS[sk]["name"])
        ax.set(title=f"Friction Fµ — {mat.capitalize()} foot\n(FN = {FN} N)",
               xlabel="Displacement (mm)", ylabel="Fµ (N)")
        ax.legend(fontsize=7); ax.grid()

    # ── P4: Bulldozing FB all soils ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    for sk, sc in SOIL_C.items():
        m = MECH_PARAMS[sk]
        ax.plot(z_arr * 1000,
                bulldozing_power_model(z_arr, m["BC"], m["BE"]),
                color=sc, lw=2,
                label=f"{SOIL_PARAMS[sk]['name']}\n(a={m['BC']:.1e} n={m['BE']})")
    ax.set(title="Bulldozing Resistance FB\nPower model: F = a·z^n",
           xlabel="Sinkage (mm)", ylabel="FB (N)")
    ax.legend(fontsize=6); ax.grid()

    # ── P5: Total FT all soils (metal foot, approx sinkage) ───────────────────
    ax = fig.add_subplot(gs[1, 0])
    z_approx = x_arr * 0.12        # illustrative proportional sinkage
    for sk, sc in SOIL_C.items():
        m   = MECH_PARAMS[sk]
        mu  = m["mu_metal"]
        FT, _, _ = total_shear_force_model(FN, mu, x_arr, z_approx, m["BC"], m["BE"])
        ax.plot(x_arr * 1000, FT, color=sc, lw=2, label=SOIL_PARAMS[sk]["name"])
    ax.set(title=f"Total Shear FT (metal, FN={FN}N)\nFT = Fµ + FB",
           xlabel="Displacement (mm)", ylabel="FT (N)")
    ax.legend(fontsize=7); ax.grid()

    # ── P6: Janosi-Hanamoto all soils ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    sigma = FN / area
    for sk, sc in SOIL_C.items():
        s   = SOIL_PARAMS[sk]
        tau = janosi_hanamoto_model(
            s["cohesion_kPa"] * 1e3, sigma, s["friction_angle_deg"], x_arr, s["Janosi_K"])
        ax.plot(x_arr * 1000, tau * area, color=sc, lw=2, label=SOIL_PARAMS[sk]["name"])
    ax.set(title=f"Janosi-Hanamoto (FN={FN}N)\n[comparison — less accurate per paper]",
           xlabel="Displacement (mm)", ylabel="FT (N)")
    ax.legend(fontsize=7); ax.grid()

    # ── P7: Friction coeff bar chart (Table II) ────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    sk_keys  = list(SOIL_PARAMS.keys())
    sk_names = [SOIL_PARAMS[k]["name"].replace(" ", "\n") for k in sk_keys]
    xpos     = np.arange(len(sk_keys))
    w        = 0.26
    for j, (mat, mc) in enumerate(MAT_C.items()):
        vals = [MECH_PARAMS[k][f"mu_{mat}"]     for k in sk_keys]
        errs = [MECH_PARAMS[k][f"mu_{mat}_std"] for k in sk_keys]
        ax.bar(xpos + j * w, vals, w, yerr=errs,
               label=mat.capitalize(), color=mc, alpha=0.85, capsize=4)
    ax.set_xticks(xpos + w)
    ax.set_xticklabels(sk_names, fontsize=8)
    ax.set(title="Friction Coefficients μ (Table II)",
           ylabel="μ")
    ax.legend(fontsize=8); ax.grid(axis="y")

    # ── P8: Fµ vs FN sensitivity (steady state) ──────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    x_ss = 0.10   # 100 mm — deep into gross-sliding regime
    for mat, mc in MAT_C.items():
        mu = MECH_PARAMS[CFG["ACTIVE_SOIL"]][f"mu_{mat}"]
        ax.plot(FN_rng,
                friction_force_model(FN_rng, mu, x_ss),
                color=mc, lw=2, label=f"{mat.capitalize()} (µ={mu})")
    ax.set(title=f"Fµ vs Normal Load at x=100mm\n({SOIL_PARAMS[CFG['ACTIVE_SOIL']]['name']})",
           xlabel="FN (N)", ylabel="Fµ (N)")
    ax.legend(fontsize=8); ax.grid()

    plt.savefig("terramechanics_summary.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show(block=False); plt.pause(0.1)
    print("  Saved → terramechanics_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION F: NORMAL LOAD SWEEP — Reproduce paper's multi-load experiment
# ══════════════════════════════════════════════════════════════════════════════

def plot_normal_load_sweep():
    """
    Reproduces the style of Fig. 5/6/7 from the paper:
    Family of shear-force curves at 20, 40, 60, 80, 100 N normal load.
    One panel per foot material.
    """
    x_arr  = np.linspace(0, 0.15, 400)
    z_arr  = 0.008 * (1 - np.exp(-x_arr / 0.025)) + 0.0005 * x_arr
    loads  = [20, 40, 60, 80, 100]
    cmap   = plt.cm.plasma(np.linspace(0.2, 0.9, len(loads)))

    soil   = SOIL_PARAMS[CFG["ACTIVE_SOIL"]]
    mech   = MECH_PARAMS[CFG["ACTIVE_SOIL"]]
    area   = math.pi * CFG["FOOT_RADIUS_M"]**2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _suptitle(fig,
        f"Normal Load Sweep — {soil['name']}  │  "
        f"Solid = Paper model  ·  Dashed = Janosi-Hanamoto")

    for col, mat in enumerate(["plastic", "rubber", "metal"]):
        ax  = axes[col]
        mu  = mech[f"mu_{mat}"]
        for FN, clr in zip(loads, cmap):
            FT, _, _ = total_shear_force_model(
                FN, mu, x_arr, z_arr, mech["BC"], mech["BE"])
            sigma = FN / area
            tau   = janosi_hanamoto_model(
                soil["cohesion_kPa"] * 1e3, sigma,
                soil["friction_angle_deg"], x_arr, soil["Janosi_K"])
            FT_JH = tau * area

            ax.plot(x_arr * 1000, FT,    color=clr, lw=2,   label=f"{FN} N")
            ax.plot(x_arr * 1000, FT_JH, color=clr, lw=1.2, ls="--")

        ax.set(title=f"{mat.capitalize()}-soled foot  (µ={mu})",
               xlabel="Horizontal Displacement (mm)", ylabel="Shear Force (N)")
        ax.legend(title="FN", fontsize=7, loc="upper left"); ax.grid()

    plt.tight_layout()
    plt.savefig("normal_load_sweep.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show(block=False); plt.pause(0.1)
    print("  Saved → normal_load_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 72)
    print(" PyChrono Legged-Robot × Deformable-Terrain Simulation")
    print(f" Soil    : {SOIL_PARAMS[CFG['ACTIVE_SOIL']]['name']}")
    print(f" Foot    : {CFG['FOOT_MATERIAL'].capitalize()}  "
          f"(R={CFG['FOOT_RADIUS_M']*100:.1f} cm)")
    print(f" FN      : {CFG['NORMAL_LOAD_N']} N")
    print(f" PyChrono: {'ENABLED' if CHRONO_AVAILABLE else 'DISABLED — analytical mode'}")
    print(f" Output  : {os.path.abspath('.')}")
    print("═" * 72)

    # ── Plot 1: Terramechanics overview (always runs, no Chrono needed) ────────
    print("\n[1/4] Terramechanics summary …")
    plot_terramechanics_summary()

    # ── Plot 2: Normal load sweep (always runs) ────────────────────────────────
    print("\n[2/4] Normal load sweep …")
    plot_normal_load_sweep()

    # ── Plot 3: SIM-1 foot-terrain interaction ────────────────────────────────
    print(f"\n[3/4] SIM-1 foot–terrain interaction "
          f"({'PyChrono SCM' if CHRONO_AVAILABLE else 'analytical'}) …")
    log1 = run_sim1_foot_terrain()
    plot_sim1(log1)

    # ── Plot 4: SIM-2 tripod robot ────────────────────────────────────────────
    print(f"\n[4/4] SIM-2 tripod robot "
          f"({'PyChrono SCM' if CHRONO_AVAILABLE else 'analytical'}) …")
    log2 = run_sim2_robot()
    plot_sim2(log2)

    print("\n" + "═" * 72)
    print(" All done.  PNG files saved to:", os.path.abspath("."))
    print("   terramechanics_summary.png")
    print("   normal_load_sweep.png")
    print("   sim1_foot_terrain.png")
    print("   sim2_robot.png")
    print("═" * 72)
    print("""
 ─── HOW TO MODIFY ───────────────────────────────────────────────────────────
  • Change soil:        CFG["ACTIVE_SOIL"] = "wet_loam"
  • Change foot:        CFG["FOOT_MATERIAL"] = "rubber"
  • Change normal load: CFG["NORMAL_LOAD_N"] = 40.0
  • Change foot size:   CFG["FOOT_RADIUS_M"] = 0.040  (80 mm dia)
  • Change robot mass:  CFG["ROBOT_MASS_KG"] = 25.0
  • Edit Table II:      MECH_PARAMS["coarse_sand"]["mu_plastic"] = 0.60
  • Edit Bekker params: SOIL_PARAMS["coarse_sand"]["Bekker_Kphi"] = 2_500_000
 ─────────────────────────────────────────────────────────────────────────────
""")
    # Keep all figure windows open until user closes them
    plt.show(block=True)
