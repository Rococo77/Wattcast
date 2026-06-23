"""Turn raw consumption + weather into a model-ready feature matrix.

Leakage discipline
------------------
The operational task is a **J+1** forecast: standing at the end of day *D* we
predict every hour of day *D+1*. So the freshest consumption we may use for an
hour *H* is *H-24*. Every autoregressive feature therefore reads from *H-24* or
earlier (``LAGS`` start at 24; rolling means are shifted by ``HORIZON``).
Temperature is the **forecast** for hour *H*, which is legitimately known ahead
of time, so it is used un-shifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_pkg
except ImportError:  # pragma: no cover
    holidays_pkg = None

from ml import config


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar + holiday features derived purely from the timestamp."""
    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype("int8")

    # Cyclical encodings keep midnight close to 23:00 and Dec close to Jan.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    if holidays_pkg is not None:
        years = range(idx.year.min(), idx.year.max() + 1)
        fr = holidays_pkg.France(years=years)
        out["is_holiday"] = pd.Series(idx.date, index=idx).isin(fr).astype("int8")
    else:  # pragma: no cover
        out["is_holiday"] = 0
    return out


def _weather_features(temperature: pd.Series) -> pd.DataFrame:
    """Temperature and its U-shaped heating/cooling transforms (forecast-safe)."""
    out = pd.DataFrame(index=temperature.index)
    out["temperature"] = temperature
    # The load/temperature curve is a U: heating below ~15 °C, cooling above ~20.
    out["heating_degrees"] = np.maximum(0.0, 15.0 - temperature)
    out["cooling_degrees"] = np.maximum(0.0, temperature - 20.0)
    out["temp_roll24"] = temperature.rolling(24, min_periods=1).mean()
    return out


def _autoregressive_features(consumption: pd.Series) -> pd.DataFrame:
    """Lagged consumption + rolling means, all anchored at ``H-HORIZON`` or older."""
    out = pd.DataFrame(index=consumption.index)
    for lag in config.LAGS:
        out[f"lag_{lag}"] = consumption.shift(lag)
    # Rolling means must not peek past H-HORIZON, hence the shift.
    shifted = consumption.shift(config.HORIZON)
    for window in config.ROLL_WINDOWS:
        out[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=window // 2).mean()
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from a raw frame.

    ``raw`` must be hourly-indexed and contain at least ``consumption`` and
    ``temperature`` columns. ``rte_forecast`` is carried through untouched (used
    only by :mod:`ml.evaluate` as a reference, never as a feature).
    """
    raw = raw.sort_index()
    idx = raw.index

    parts = [
        _calendar_features(idx),
        _weather_features(raw["temperature"]),
        _autoregressive_features(raw[config.TARGET]),
    ]
    features = pd.concat(parts, axis=1)
    features[config.TARGET] = raw[config.TARGET]
    if "rte_forecast" in raw.columns:
        features["rte_forecast"] = raw["rte_forecast"]

    # Drop the warm-up window where the longest lag/rolling mean is undefined.
    return features.dropna(subset=[c for c in features.columns if c.startswith(("lag_", "roll_"))])


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns = everything except the target and the RTE reference."""
    return [c for c in df.columns if c not in (config.TARGET, "rte_forecast")]


def run() -> pd.DataFrame:
    config.ensure_dirs()
    consumption = pd.read_parquet(config.RAW_CONSUMPTION)
    weather = pd.read_parquet(config.RAW_WEATHER)

    raw = consumption.join(weather, how="inner")
    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw.asfreq("1h")
    # Bridge short gaps so lags stay aligned; long gaps are dropped downstream.
    raw[config.TARGET] = raw[config.TARGET].interpolate(limit=3)
    raw["temperature"] = raw["temperature"].interpolate(limit=6)

    features = build_features(raw)
    features.to_parquet(config.FEATURES_PATH)
    print(
        f"[features] {len(features):>6} rows × {len(feature_columns(features))} features"
        f"  [{features.index.min()} → {features.index.max()}]"
    )
    print(f"  → {config.FEATURES_PATH}")
    return features


if __name__ == "__main__":
    run()
