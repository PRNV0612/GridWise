"""GridWise configuration: paths, constants and thresholds.

Single source of truth so ETL, training, evaluation and the API agree.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_UAE = ROOT / "data" / "uae"
MODELS_DIR = ROOT / "models"

# BDG2 checkout lives under data/raw/bdg2 (gitignored).
BDG2_DIR = DATA_RAW / "bdg2"
BDG2_METERS_CLEANED = BDG2_DIR / "data" / "meters" / "cleaned"
BDG2_METERS_KAGGLE = BDG2_DIR / "data" / "meters" / "kaggle"
BDG2_METADATA = BDG2_DIR / "data" / "metadata"
BDG2_WEATHER = BDG2_DIR / "data" / "weather"

PARQUET_PATH = DATA_PROCESSED / "electricity.parquet"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "model.joblib"

# --- ETL --------------------------------------------------------------------
TARGET_METER_TYPE = "electricity"          # chilledwater/steam/hotwater excluded
MAX_MISSING_SHARE = 0.20                   # drop meters with >20% missing hours
MIN_COVERAGE_BOTH_YEARS = 0.80             # require >=80% coverage in 2016 AND 2017
STUCK_ZERO_RUN_HOURS = 24                  # drop runs of zeros >= this length
STUCK_VALUE_RUN_HOURS = 48                 # drop runs of identical readings
SUBSET_N_BUILDINGS = 200                   # iterate fast, scale later
RANDOM_SEED = 42

# --- Model ------------------------------------------------------------------
TRAIN_YEAR = 2016
TEST_YEAR = 2017
VALIDATION_MONTHS = (10, 11, 12)           # Oct-Dec 2016 for early stopping

# --- Anomalies --------------------------------------------------------------
ANOMALY_Z_THRESHOLD = 3.0                  # robust z-score on residuals
ANOMALY_MIN_RUN_HOURS = 3                  # consecutive hours to call an event
NIGHT_BASELOAD_DRIFT = 0.25                # 00:00-05:00 median vs 4-week median
ROLLED_STATS_WINDOW_HOURS = 168            # 1 week of history for MAD/median

# --- Carbon-aware shifting --------------------------------------------------
DEFAULT_FLEXIBLE_SHARE = 0.15              # HVAC pre-cooling, EV charging, DHW
DEFAULT_HEADROOM_SHARE = 0.20              # per-hour cap = L[t] * (1 + headroom)
MIN_CARBON_GAIN_G = 10.0                   # gCO2/kWh; stop shifting below this

# --- UAE scenario -----------------------------------------------------------
# DEWA grid emission factor, kgCO2/kWh (published DEWA sustainability figures).
# Falling with the Mohammed bin Rashid Solar Park rollout.
DEWA_EMISSION_FACTOR_2026 = 0.35           # kgCO2/kWh, labelled estimate
KAPSARC_ANCHOR_CSV = DATA_UAE / "kapsarc_dewa_annual.csv"
