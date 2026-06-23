"""Produce the operational J+1 forecast and the dashboard JSON.

Because the horizon (24 h) equals the shortest lag, every feature for the next
24 hours is already known at run time: the autoregressive lags read from recent
*actual* consumption and the temperature comes from the Open-Meteo **forecast**
(or its deterministic offline stand-in). No recursive multi-step prediction is
needed — one clean batch ``predict`` over the future block.

Writes three files the static dashboard reads client-side:

* ``web/public/data/predictions.json`` — model curve over a recent window + the
  next 24 h forecast.
* ``web/public/data/actuals.json``     — realised consumption over that window.
* ``web/public/data/metrics.json``     — headline metrics + feature importance.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from ml import config, ingest
from ml.features import build_features

# How much realised history to show alongside the forecast.
HISTORY_HOURS = 7 * 24


def _load_model():
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], bundle["features"]  # forecaster, feature list


def _future_temperature(idx: pd.DatetimeIndex) -> pd.Series:
    """Forecast temperature for ``idx`` from Open-Meteo, offline fallback aside."""
    try:
        fc = ingest.fetch_weather(str(idx[0].date()), str(idx[-1].date()), forecast=True)
        return fc["temperature"].reindex(idx).interpolate()
    except Exception as exc:  # noqa: BLE001 — offline / API down
        print(f"[predict] weather forecast unavailable ({exc!s}); using seasonal model")
        return ingest.seasonal_temperature(idx)


def build_forecast_frame() -> pd.DataFrame:
    """Assemble recent actuals + a 24 h future block, then compute features."""
    consumption = pd.read_parquet(config.RAW_CONSUMPTION).sort_index()
    weather = pd.read_parquet(config.RAW_WEATHER).sort_index()
    raw = consumption.join(weather, how="left")
    raw = raw[~raw.index.duplicated(keep="first")].asfreq("1h")
    raw["temperature"] = raw["temperature"].interpolate(limit=6)

    now = raw.index.max()
    future_idx = pd.date_range(
        now + pd.Timedelta(hours=1), periods=config.HORIZON, freq="1h", tz=config.TZ
    )
    future = pd.DataFrame(index=future_idx)
    future["temperature"] = _future_temperature(future_idx)

    combined = pd.concat([raw, future])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return build_features(combined), now


def run() -> dict:
    config.ensure_dirs()
    forecaster, _features = _load_model()
    feats, now = build_forecast_frame()

    window_start = now - pd.Timedelta(hours=HISTORY_HOURS)
    feats = feats[feats.index >= window_start]
    feats = feats.assign(predicted=np.round(forecaster.predict(feats), 1))

    is_future = feats.index > now
    predictions = [
        {
            "datetime": ts.isoformat(),
            "predicted": round(float(row.predicted), 1),
            "temperature": round(float(row.temperature), 1),
            "is_forecast": bool(is_future[i]),
        }
        for i, (ts, row) in enumerate(feats.iterrows())
    ]
    actuals = [
        {"datetime": ts.isoformat(), "actual": round(float(v), 1)}
        for ts, v in feats.loc[~is_future, config.TARGET].items()
        if pd.notna(v)
    ]

    predictions_payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "horizon_hours": config.HORIZON,
        "now": now.isoformat(),
        "points": predictions,
    }
    config.PREDICTIONS_JSON.write_text(json.dumps(predictions_payload, indent=2))
    config.ACTUALS_JSON.write_text(json.dumps({"points": actuals}, indent=2))

    # Surface training metrics + the model bake-off to the dashboard (best-effort).
    if config.METRICS_PATH.exists():
        shutil.copyfile(config.METRICS_PATH, config.METRICS_JSON)
    evaluation_path = config.MODEL_DIR / "evaluation.json"
    if evaluation_path.exists():
        shutil.copyfile(evaluation_path, config.WEB_DATA_DIR / "evaluation.json")

    forecast_pts = sum(p["is_forecast"] for p in predictions)
    print(
        f"[predict] now={now}  history={len(actuals)}h  forecast={forecast_pts}h"
    )
    print(f"  → {config.PREDICTIONS_JSON}\n  → {config.ACTUALS_JSON}\n  → {config.METRICS_JSON}")
    return predictions_payload


if __name__ == "__main__":
    run()
