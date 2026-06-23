import numpy as np
import pytest

from ml.features import feature_columns
from ml.models import XGBoostForecaster, build_forecaster
from ml.utils import mae, time_split


def test_build_forecaster_factory():
    assert build_forecaster("xgboost").name == "XGBoost"
    assert build_forecaster("xgb").name == "XGBoost"
    with pytest.raises(ValueError):
        build_forecaster("nope")


def test_xgboost_beats_naive_baseline(features_df):
    """The core promise: the model must beat the same-hour-last-week baseline."""
    features = feature_columns(features_df)
    train_df, test_df = time_split(features_df, test_size=0.2)
    model = XGBoostForecaster(early_stopping=False).fit(train_df, features)
    preds = model.predict(test_df)

    model_mae = mae(test_df["consumption"], preds)
    baseline_mae = mae(test_df["consumption"], test_df["lag_168"])
    assert np.isfinite(preds).all()
    assert model_mae < baseline_mae


def test_xgboost_feature_importance(features_df):
    features = feature_columns(features_df)
    model = XGBoostForecaster(early_stopping=False).fit(features_df, features)
    imp = model.feature_importance
    assert len(imp) == len(features)
    assert (imp >= 0).all()


def test_prophet_optional(features_df):
    pytest.importorskip("prophet")
    from ml.models import ProphetForecaster

    features = feature_columns(features_df)
    window = features_df.iloc[: 24 * 60]  # 60 days keeps the fit quick
    model = ProphetForecaster().fit(window.iloc[:-24], features)
    preds = model.predict(window.iloc[-24:])
    assert len(preds) == 24
    assert np.isfinite(preds).all()
    # Prophet got the engineered temperature regressors, not just seasonality.
    assert "temperature" in model.regressors
