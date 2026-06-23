import numpy as np
import pandas as pd

from ml.utils import mae, mape, regression_metrics, time_split


def test_mae_zero_for_identical():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    assert mae([0, 0, 0], [1, 2, 3]) == 2.0


def test_mape_percentage():
    # 10% error on every point → 10.0
    assert mape([100, 200], [110, 220]) == 10.0


def test_mape_ignores_zeros():
    # the zero-truth point is masked out, leaving a clean 10%.
    assert mape([0, 100], [5, 110]) == 10.0


def test_regression_metrics_keys():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert set(m) == {"mae", "mape"}


def test_time_split_is_chronological():
    df = pd.DataFrame({"x": range(100)}, index=pd.date_range("2022-01-01", periods=100, freq="h"))
    train, test = time_split(df, test_size=0.2)
    assert len(train) == 80 and len(test) == 20
    # No leakage: every train timestamp precedes every test timestamp.
    assert train.index.max() < test.index.min()
