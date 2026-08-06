"""Central configuration: every constant, seed, and column list lives here.

Nothing in this project reads a magic number from anywhere else.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "latent_fish_data.xlsx"
DATA_SHEET = "data"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- seeds
SEED = 20260805
PERMUTATION_SEED = 90210
BOOTSTRAP_SEED = 31337

# ------------------------------------------------------------- outcome
DURATION_COL = "time_to_stageIII_s"
EVENT_COL = "event"
CENSOR_TIME_COL = "censor_time_s"

# ------------------------------------------------------------ grouping
SUBJECT_COL = "fish_id"
CLUTCH_COL = "clutch_id"          # biological replicate -> baseline stratum
STRATUM_COL = "drop_height_cm"    # injury dose stratum
LABEL_COL = "condition"           # label only, never a variable

# ---------------------------------------------------------- predictors
PRESSURE = "max_pressure_kPa"

# Every lfp_* column except the QC channel. Confirmed by the user:
# take the prefix literally.
LFP_FEATURES = [
    "lfp_rec_time_h_post_injury",
    "lfp_ap_slope",
    "lfp_ap_offset",
    "lfp_ap_fit_error",
    "lfp_variance_uV2",
    "lfp_autocorr_lag1",
    "lfp_discharge_rate_per_min",
    "lfp_power_low_norm",
    "lfp_power_mid_norm",
    "lfp_power_high_norm",
]

BEH_FEATURES = [
    "beh_baseline_speed_mm_s",
    "beh_baseline_immobile_frac",
    "beh_baseline_burst_rate_per_min",
]

# Quality control -- diagnostics, never predictors.
QC_COLUMNS = ["lfp_clean_fraction", "dlc_tracking_error_px"]

# Constant across all 100 larvae; carry no information.
CONSTANT_COLUMNS = [
    "age_dpf",
    "ptz_conc_mM",
    "drop_weight_g",
    "censor_time_s",
    "scorer_blinded",
]

# ------------------------------------------------------------- models
M0_FEATURES = [PRESSURE]
M1_FEATURES = [PRESSURE] + LFP_FEATURES

# max_pressure_kPa is never shrunk: M0 vs M1 and the injury-gradient
# checks are only meaningful if the dose term is guaranteed to survive.
UNPENALIZED = {PRESSURE}

# --------------------------------------------------- elastic-net grid
PENALIZER_GRID = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
L1_RATIO_GRID = [0.1, 0.5, 0.9, 1.0]
INNER_FOLDS = 5

# ----------------------------------------------------------- inference
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 2000
ALPHA = 0.05                      # -> 95% intervals
PH_ALPHA = 0.05                   # Schoenfeld global test threshold
EPV_THRESHOLD = 10.0              # events per predictor warning line

# ------------------------------------------------------------- solver
MAX_ITER = 500
TOL = 1e-9

# ---------------------------------------- expected biological direction
# Sign of the expected log-hazard coefficient.
#   +1 : rising feature should SHORTEN latency (raise hazard), HR > 1
#   -1 : rising feature should LENGTHEN latency (lower hazard), HR < 1
#
# lfp_ap_slope is the aperiodic exponent. Flattening (a SMALLER slope)
# means a shift toward excitation, so a smaller slope should raise the
# hazard -- i.e. the coefficient on slope itself is expected negative.
EXPECTED_DIRECTION = {
    "lfp_ap_slope": -1,
    "lfp_variance_uV2": +1,
    "lfp_autocorr_lag1": +1,
    "lfp_discharge_rate_per_min": +1,
}

DIRECTION_RATIONALE = {
    "lfp_ap_slope": (
        "Aperiodic exponent; flattening indexes a shift toward excitation "
        "in the excitation-inhibition balance, so a LOWER slope should "
        "predict a SHORTER latency to seizure (HR < 1 per unit slope)."
    ),
    "lfp_variance_uV2": (
        "Critical slowing indicator; variance rises as a network "
        "approaches instability, so HIGHER variance should predict a "
        "SHORTER latency (HR > 1)."
    ),
    "lfp_autocorr_lag1": (
        "Critical slowing indicator; lag-1 autocorrelation rises as "
        "recovery from perturbation slows near a bifurcation, so HIGHER "
        "autocorrelation should predict a SHORTER latency (HR > 1)."
    ),
    "lfp_discharge_rate_per_min": (
        "Interictal-like discharge rate; more spontaneous discharges "
        "indicate a more irritable network, so a HIGHER rate should "
        "predict a SHORTER latency (HR > 1)."
    ),
}

# The two mechanistically linked critical-slowing features (question 3).
CRITICAL_SLOWING_PAIR = ("lfp_variance_uV2", "lfp_autocorr_lag1")

# A coefficient this small is a shrunk-to-nothing coefficient, not a
# direction. |log HR| < 0.01 means a hazard ratio inside 0.99-1.01 per
# standard deviation, which carries no biological claim. Asking whether
# such a coefficient "points the right way" reads sign noise in the third
# decimal place, so questions 2 and 3 treat these as no-direction rather
# than as agreement.
NEGLIGIBLE_LOG_HR = 0.01
