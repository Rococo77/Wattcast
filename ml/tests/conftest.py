"""Shared fixtures: a synthetic dataset and a tmp-redirected config."""

from __future__ import annotations

import pytest

from ml import config, ingest
from ml.features import build_features

# A compact but realistic window — long enough for H-168 lags and a few folds,
# short enough to keep the suite fast. Built once per session.
START, END = "2022-01-01", "2023-03-31"


@pytest.fixture(scope="session")
def synthetic_raw():
    return ingest.generate_synthetic(START, END)


@pytest.fixture(scope="session")
def features_df(synthetic_raw):
    return build_features(synthetic_raw)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect every config path into a tmp dir so tests touch no repo files."""
    raw = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    model = tmp_path / "model"
    web = tmp_path / "web" / "public" / "data"
    for d in (raw, proc, model, web):
        d.mkdir(parents=True)

    patches = {
        "RAW_DIR": raw,
        "PROCESSED_DIR": proc,
        "MODEL_DIR": model,
        "WEB_DATA_DIR": web,
        "RAW_CONSUMPTION": raw / "consumption.parquet",
        "RAW_WEATHER": raw / "weather.parquet",
        "FEATURES_PATH": proc / "features.parquet",
        "MODEL_PATH": model / "model.pkl",
        "METRICS_PATH": model / "metrics.json",
        "PREDICTIONS_JSON": web / "predictions.json",
        "ACTUALS_JSON": web / "actuals.json",
        "METRICS_JSON": web / "metrics.json",
    }
    for name, value in patches.items():
        monkeypatch.setattr(config, name, value)
    return config
