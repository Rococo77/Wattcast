import pandas as pd

from ml import config
from ml.features import build_features, feature_columns


def test_no_nan_in_feature_columns(features_df):
    cols = feature_columns(features_df)
    assert not features_df[cols].isna().any().any()


def test_expected_feature_set(features_df):
    cols = set(feature_columns(features_df))
    for expected in ("hour", "is_weekend", "is_holiday", "temperature",
                     "heating_degrees", "cooling_degrees", "lag_24", "lag_168",
                     "roll_mean_24", "roll_mean_168"):
        assert expected in cols, expected


def test_target_and_reference_excluded_from_features(features_df):
    cols = feature_columns(features_df)
    assert config.TARGET not in cols
    assert "rte_forecast" not in cols


def test_lag_alignment_no_leakage(synthetic_raw, features_df):
    """lag_24 at time t must equal the *actual* consumption at t-24."""
    t = features_df.index[500]
    expected = synthetic_raw.loc[t - pd.Timedelta(hours=24), config.TARGET]
    assert abs(features_df.loc[t, "lag_24"] - expected) < 1e-6


def test_weekend_flag(features_df):
    # 2022-01-15 is a Saturday.
    sat = features_df[features_df.index.normalize() == pd.Timestamp("2022-01-15", tz=config.TZ)]
    assert (sat["is_weekend"] == 1).all()


def test_french_holiday_flag(features_df):
    # 14 July (Bastille Day) is a public holiday in France.
    bastille = features_df[features_df.index.normalize() == pd.Timestamp("2022-07-14", tz=config.TZ)]
    assert len(bastille) > 0 and (bastille["is_holiday"] == 1).all()


def test_heating_cooling_non_negative(features_df):
    assert (features_df["heating_degrees"] >= 0).all()
    assert (features_df["cooling_degrees"] >= 0).all()


def test_build_features_is_sorted(synthetic_raw):
    shuffled = synthetic_raw.sample(frac=1, random_state=0)
    out = build_features(shuffled)
    assert out.index.is_monotonic_increasing
