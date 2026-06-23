"""Shared helpers: error metrics and the strict temporal split."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (MW)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%), guarded against zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Round-tripped {mae, mape} dict for JSON serialisation."""
    return {"mae": round(mae(y_true, y_pred), 2), "mape": round(mape(y_true, y_pred), 3)}


def time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: the last ``test_size`` fraction is the test set.

    No shuffling — training always happens on the past, evaluation on the most
    recent period. This is the single most important guard against optimistic
    metrics in a time-series problem.
    """
    df = df.sort_index()
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]
