"""Central configuration for the WattCast pipeline.

Everything path- or constant-related lives here so the individual scripts
(``ingest``, ``features``, ``train``, ``evaluate``, ``predict``) stay short and
share a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

WEB_DATA_DIR = ROOT / "web" / "public" / "data"

# Raw artefacts written by ``ingest.py``.
RAW_CONSUMPTION = RAW_DIR / "consumption.parquet"
RAW_WEATHER = RAW_DIR / "weather.parquet"
# Feature matrix written by ``features.py``.
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# Web artefacts written by ``predict.py``.
PREDICTIONS_JSON = WEB_DATA_DIR / "predictions.json"
ACTUALS_JSON = WEB_DATA_DIR / "actuals.json"
METRICS_JSON = WEB_DATA_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
TARGET = "consumption"  # MW, hourly national consumption
TZ = "Europe/Paris"

# Lags (in hours) used as features. H-24 = same hour yesterday, H-168 = same
# hour last week — the two strongest autoregressive signals for load curves.
LAGS = (24, 25, 48, 168, 169)
# Rolling-mean windows (in hours).
ROLL_WINDOWS = (24, 168)
# Horizon of the operational forecast (hours ahead).
HORIZON = 24

# Fraction of the most recent data held out for the final test split.
TEST_SIZE = 0.15
# Number of folds for the walk-forward (TimeSeriesSplit) evaluation.
N_SPLITS = 5

# --------------------------------------------------------------------------- #
# Geography — population-weighted temperature.
#
# French load is driven by temperature; a single point (Paris) is a decent
# proxy but a population-weighted blend of the largest urban areas tracks the
# national curve more faithfully. Weights are rough population shares.
# --------------------------------------------------------------------------- #
CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522, "weight": 0.30},
    "Lyon": {"lat": 45.7640, "lon": 4.8357, "weight": 0.18},
    "Marseille": {"lat": 43.2965, "lon": 5.3698, "weight": 0.18},
    "Lille": {"lat": 50.6292, "lon": 3.0573, "weight": 0.17},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442, "weight": 0.17},
}

# --------------------------------------------------------------------------- #
# Data sources (no API key required).
# --------------------------------------------------------------------------- #
ODRE_BASE = "https://opendata.reseaux-energies.fr/api/explore/v2.1/catalog/datasets"
ODRE_CONS_DEF = "eco2mix-national-cons-def"  # consolidated/definitive history
ODRE_CONS_TR = "eco2mix-national-tr"  # real-time (recent actuals)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Network timeout (seconds) for every HTTP call.
HTTP_TIMEOUT = 60


def ensure_dirs() -> None:
    """Create every output directory if it does not exist yet."""
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, WEB_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
