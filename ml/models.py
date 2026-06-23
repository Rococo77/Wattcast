"""Forecaster abstraction — one interface, several model families.

This is what lets WattCast *compare* models on an equal footing. Every
forecaster exposes the same two methods::

    forecaster.fit(train_df, features)   # train_df is the engineered matrix
    forecaster.predict(df) -> np.ndarray

so :mod:`ml.train`, :mod:`ml.predict` and the walk-forward bake-off in
:mod:`ml.evaluate` are all model-agnostic.

* :class:`XGBoostForecaster` — gradient-boosted trees over the full engineered
  feature set (lags, rolling means, calendar, U-shaped temperature). The
  production model.
* :class:`ProphetForecaster` — additive trend + daily/weekly/yearly seasonality
  with temperature, heating/cooling degrees and recent lags as extra
  regressors. Optional benchmark (requires the ``prophet`` package).
"""

from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pandas as pd

from ml import config

# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}


class BaseForecaster:
    """Common interface shared by every model family."""

    name: str = "base"

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> BaseForecaster:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self, *, early_stopping: bool = True, params: dict | None = None):
        self.early_stopping = early_stopping
        self.params = {**XGB_PARAMS, **(params or {})}
        self.features: list[str] = []
        self.model = None

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> XGBoostForecaster:
        from xgboost import XGBRegressor

        from ml.utils import time_split

        self.features = features
        if self.early_stopping:
            inner_train, inner_val = time_split(train_df, test_size=0.1)
            self.model = XGBRegressor(
                early_stopping_rounds=40, eval_metric="mae", **self.params
            )
            self.model.fit(
                inner_train[features],
                inner_train[config.TARGET],
                eval_set=[(inner_val[features], inner_val[config.TARGET])],
                verbose=False,
            )
        else:
            self.model = XGBRegressor(**self.params)
            self.model.fit(train_df[features], train_df[config.TARGET], verbose=False)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.features)

    @property
    def best_iteration(self) -> int:
        return int(getattr(self.model, "best_iteration", self.params["n_estimators"]))


class ProphetForecaster(BaseForecaster):
    name = "Prophet"

    # Extra regressors (beyond Prophet's built-in seasonalities), kept only if
    # present in the feature frame. Gives Prophet the non-linear temperature
    # response and recent autocorrelation — a fair fight against XGBoost.
    CANDIDATE_REGRESSORS = (
        "temperature",
        "heating_degrees",
        "cooling_degrees",
        "is_holiday",
        "lag_24",
        "lag_168",
    )

    def __init__(self):
        self.model = None
        self.regressors: list[str] = []

    @staticmethod
    def is_available() -> bool:
        try:
            import prophet  # noqa: F401

            return True
        except Exception:
            return False

    def _frame(self, df: pd.DataFrame, *, with_target: bool) -> pd.DataFrame:
        out = pd.DataFrame({"ds": df.index.tz_localize(None)})
        if with_target:
            out["y"] = df[config.TARGET].to_numpy()
        for r in self.regressors:
            out[r] = df[r].to_numpy()
        return out

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> ProphetForecaster:
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        self.regressors = [r for r in self.CANDIDATE_REGRESSORS if r in features]

        self.model = Prophet(
            growth="flat",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
        )
        for r in self.regressors:
            self.model.add_regressor(r)

        # Prophet's fitter is chatty; keep the pipeline output clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(self._frame(train_df, with_target=True))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        forecast = self.model.predict(self._frame(df, with_target=False))
        return forecast["yhat"].to_numpy()


def build_forecaster(name: str, **kwargs) -> BaseForecaster:
    """Factory: ``build_forecaster("xgboost")`` / ``build_forecaster("prophet")``."""
    name = name.lower()
    if name in ("xgboost", "xgb"):
        return XGBoostForecaster(**kwargs)
    if name == "prophet":
        if not ProphetForecaster.is_available():
            raise RuntimeError("the 'prophet' package is not installed")
        return ProphetForecaster()
    raise ValueError(f"unknown model '{name}' (choose: xgboost, prophet)")
