"""Ingest raw data into ``data/raw/``.

Two sources, no API key required:

* **éCO2mix (RTE via ODRÉ)** — realised national consumption + RTE's own J-1
  forecast, used later as a reference to beat.
* **Open-Meteo** — temperature, population-weighted over the five largest French
  urban areas (see :data:`ml.config.CITIES`).

Real HTTP fetching is the default. When the network is unavailable (sandbox,
offline CI) the pipeline transparently falls back to a *deterministic synthetic*
generator that reproduces the salient structure of the real series — daily and
weekly cycles plus the U-shaped temperature/load relationship — so every
downstream step (features → train → predict → dashboard) can run end-to-end.
Pass ``--synthetic`` to force it.

Usage
-----
    python ml/ingest.py --start 2022-01-01 --end 2024-12-31
    python ml/ingest.py --recent          # last ~30 days, for the daily job
    python ml/ingest.py --synthetic        # force the offline generator
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml import config

try:  # requests is only needed for the real fetch path
    import requests
except ImportError:  # pragma: no cover
    requests = None


# --------------------------------------------------------------------------- #
# Real fetch — éCO2mix consumption (ODRÉ Opendatasoft v2.1)
# --------------------------------------------------------------------------- #
def fetch_consumption(start: str, end: str, *, realtime: bool = False) -> pd.DataFrame:
    """Download hourly national consumption (+ RTE J-1 forecast) from ODRÉ.

    The source is half-hourly; we resample to hourly means and drop rows where
    consumption is missing (the dataset pads the future with NaNs).
    """
    if requests is None:  # pragma: no cover
        raise RuntimeError("the 'requests' package is required for the real fetch path")

    dataset = config.ODRE_CONS_TR if realtime else config.ODRE_CONS_DEF
    url = f"{config.ODRE_BASE}/{dataset}/exports/csv"
    params = {
        "select": "date_heure,consommation,prevision_j1",
        "where": f"date_heure >= '{start}' AND date_heure <= '{end}'",
        "timezone": config.TZ,
        "delimiter": ";",
    }
    resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text), sep=";")
    df = df.rename(
        columns={
            "date_heure": "datetime",
            "consommation": config.TARGET,
            "prevision_j1": "rte_forecast",
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(config.TZ)
    df = df.dropna(subset=[config.TARGET]).set_index("datetime").sort_index()
    # Half-hourly → hourly.
    df = df.resample("1h").mean(numeric_only=True)
    return df.dropna(subset=[config.TARGET])


# --------------------------------------------------------------------------- #
# Real fetch — Open-Meteo temperature (population-weighted)
# --------------------------------------------------------------------------- #
def fetch_weather(start: str, end: str, *, forecast: bool = False) -> pd.DataFrame:
    """Download hourly temperature and population-weight it across cities."""
    if requests is None:  # pragma: no cover
        raise RuntimeError("the 'requests' package is required for the real fetch path")

    series: list[pd.Series] = []
    for name, c in config.CITIES.items():
        if forecast:
            url = config.OPEN_METEO_FORECAST
            params = {
                "latitude": c["lat"],
                "longitude": c["lon"],
                "hourly": "temperature_2m",
                "forecast_days": 2,
                "timezone": config.TZ,
            }
        else:
            url = config.OPEN_METEO_ARCHIVE
            params = {
                "latitude": c["lat"],
                "longitude": c["lon"],
                "hourly": "temperature_2m",
                "start_date": start,
                "end_date": end,
                "timezone": config.TZ,
            }
        resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()["hourly"]
        s = pd.Series(
            payload["temperature_2m"],
            index=pd.to_datetime(payload["time"]),
            name=name,
        )
        series.append(s * c["weight"])

    temperature = pd.concat(series, axis=1).sum(axis=1)
    temperature.index.name = "datetime"
    out = temperature.to_frame("temperature")
    out.index = out.index.tz_localize(config.TZ)
    return out


# --------------------------------------------------------------------------- #
# Synthetic fallback — deterministic, reproducible
# --------------------------------------------------------------------------- #
def seasonal_temperature(idx: pd.DatetimeIndex) -> pd.Series:
    """Deterministic, smooth seasonal + diurnal temperature for any index.

    Used both as the backbone of the synthetic generator and as the offline
    fallback for the J+1 temperature forecast in :mod:`ml.predict`.
    """
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    seasonal = 13.0 - 9.0 * np.cos(2 * np.pi * (doy - 10) / 365.25)  # coldest ~10 Jan
    diurnal = 3.0 * np.sin(2 * np.pi * (hour - 15) / 24)  # warmest mid-afternoon
    return pd.Series(seasonal + diurnal, index=idx, name="temperature")


def generate_synthetic(start: str, end: str, *, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic consumption + temperature series.

    The point is not to fake results but to give the *machinery* something with
    the right shape to chew on when the data APIs are unreachable: a seasonal +
    diurnal temperature curve and a load that responds to it in the
    characteristic U shape, with morning/evening peaks and lighter weekends.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="1h", tz=config.TZ)
    n = len(idx)

    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()  # 0 = Monday

    # --- Temperature: seasonal + diurnal backbone + AR(1) weather noise -------
    noise = np.zeros(n)
    for i in range(1, n):  # AR(1) weather noise → smooth, autocorrelated
        noise[i] = 0.92 * noise[i - 1] + rng.normal(0, 1.1)
    temperature = seasonal_temperature(idx).to_numpy() + noise

    # --- Consumption: U-shaped temperature response + calendar profile --------
    heating = np.maximum(0.0, 15.0 - temperature) * 1400.0  # winter heating
    cooling = np.maximum(0.0, temperature - 21.0) * 700.0  # summer A/C
    base = 47000.0

    # Diurnal load profile: trough at night, morning + evening peaks.
    daily = (
        6500 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 4200 * np.sin(2 * np.pi * (hour - 19) / 12)
    )
    weekend = np.where(dow >= 5, -5500.0, 0.0)  # lighter weekends
    holiday_dip = np.where((doy >= 358) | (doy <= 2), -3500.0, 0.0)  # year-end lull

    load = base + heating + cooling + daily + weekend + holiday_dip
    load += rng.normal(0, 900, n)  # measurement noise
    load = np.clip(load, 30000, None)

    # A plausible RTE J-1 forecast: the truth seen one day ago + small AR error.
    rte_err = np.zeros(n)
    for i in range(1, n):
        rte_err[i] = 0.8 * rte_err[i - 1] + rng.normal(0, 350)
    rte_forecast = load + rte_err

    return pd.DataFrame(
        {
            config.TARGET: load,
            "rte_forecast": rte_forecast,
            "temperature": temperature,
        },
        index=idx,
    ).rename_axis("datetime")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save(consumption: pd.DataFrame, weather: pd.DataFrame) -> None:
    config.ensure_dirs()
    consumption.to_parquet(config.RAW_CONSUMPTION)
    weather.to_parquet(config.RAW_WEATHER)
    print(
        f"  consumption: {len(consumption):>6} rows  "
        f"[{consumption.index.min()} → {consumption.index.max()}]"
    )
    print(
        f"  weather:     {len(weather):>6} rows  "
        f"[{weather.index.min()} → {weather.index.max()}]"
    )
    print(f"  → {config.RAW_CONSUMPTION}\n  → {config.RAW_WEATHER}")


def run(
    start: str,
    end: str,
    *,
    synthetic: bool = False,
    realtime: bool = False,
    allow_fallback: bool = True,
) -> None:
    if synthetic:
        print(f"[ingest] synthetic generation {start} → {end}")
        syn = generate_synthetic(start, end)
        _save(syn[[config.TARGET, "rte_forecast"]], syn[["temperature"]])
        return

    try:
        print(f"[ingest] fetching real data {start} → {end} (realtime={realtime})")
        consumption = fetch_consumption(start, end, realtime=realtime)
        weather = fetch_weather(start, end, forecast=False)
        _save(consumption, weather)
    except Exception as exc:  # network/HTTP/parse failure
        if not allow_fallback:
            raise
        print(f"[ingest] real fetch failed ({exc!s}); falling back to synthetic", file=sys.stderr)
        syn = generate_synthetic(start, end)
        _save(syn[[config.TARGET, "rte_forecast"]], syn[["temperature"]])


def _default_window(recent: bool) -> tuple[str, str]:
    today = datetime.now()
    if recent:
        start = (today - timedelta(days=45)).strftime("%Y-%m-%d")
    else:
        start = (today - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    return start, today.strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest WattCast raw data.")
    parser.add_argument("--start", help="YYYY-MM-DD (default: 3 years ago)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--recent", action="store_true", help="short window for the daily refresh"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="force the offline generator"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="use the éCO2mix real-time dataset"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="do not fall back to synthetic data on fetch failure",
    )
    args = parser.parse_args(argv)

    d_start, d_end = _default_window(args.recent)
    run(
        args.start or d_start,
        args.end or d_end,
        synthetic=args.synthetic,
        realtime=args.realtime or args.recent,
        allow_fallback=not args.strict,
    )


if __name__ == "__main__":
    main()
