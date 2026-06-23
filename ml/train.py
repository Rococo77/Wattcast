"""Train the production load-forecasting model.

Strict temporal validation: fit on the past, test on the most recent
``TEST_SIZE`` slice (never a random shuffle). The model is compared head-to-head
with the naive *same-hour-last-week* baseline (``lag_168``) and, when available,
RTE's own J-1 forecast — the two references the README promises to report.

Defaults to XGBoost; ``--model prophet`` trains Prophet instead (see
:mod:`ml.evaluate` for the full bake-off).

Outputs
-------
* ``model/model.pkl``    — the fitted forecaster + feature list (joblib).
* ``model/metrics.json`` — test metrics, baseline/RTE comparison, feature
  importance and run metadata. Copied to the dashboard by ``predict.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import pandas as pd

from ml import config
from ml.features import feature_columns
from ml.models import build_forecaster
from ml.utils import regression_metrics, time_split


def _baseline_metrics(test_df: pd.DataFrame) -> dict:
    """Naive references that the model must beat / be compared against."""
    out = {"baseline_naive": regression_metrics(test_df[config.TARGET], test_df["lag_168"])}
    if "rte_forecast" in test_df.columns:
        out["rte_official"] = regression_metrics(
            test_df[config.TARGET], test_df["rte_forecast"]
        )
    return out


def run(model_name: str = "xgboost") -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.FEATURES_PATH)
    features = feature_columns(df)

    train_df, test_df = time_split(df, config.TEST_SIZE)
    print(f"[train] model={model_name}  train={len(train_df)}  test={len(test_df)}  features={len(features)}")

    forecaster = build_forecaster(model_name).fit(train_df, features)
    preds = forecaster.predict(test_df)

    model_metrics = regression_metrics(test_df[config.TARGET], preds)
    references = _baseline_metrics(test_df)
    baseline_mae = references["baseline_naive"]["mae"]
    improvement = round(100 * (baseline_mae - model_metrics["mae"]) / baseline_mae, 1)

    metrics = {
        "model": forecaster.name,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_period": {"start": str(test_df.index.min()), "end": str(test_df.index.max())},
        "metrics": model_metrics,
        "references": references,
        "improvement_vs_baseline_pct": improvement,
    }
    # Feature importance is XGBoost-specific.
    if hasattr(forecaster, "feature_importance"):
        metrics["best_iteration"] = forecaster.best_iteration
        metrics["feature_importance"] = (
            forecaster.feature_importance.sort_values(ascending=False).round(4).to_dict()
        )

    joblib.dump(
        {"model": forecaster, "features": features, "model_name": forecaster.name},
        config.MODEL_PATH,
    )
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"[train] MAE={model_metrics['mae']} MW  MAPE={model_metrics['mape']} %")
    print(f"[train] baseline MAE={baseline_mae} MW  →  {improvement:+.1f}% error vs naive baseline")
    if "rte_official" in references:
        print(f"[train] RTE J-1 MAE={references['rte_official']['mae']} MW (reference)")
    print(f"  → {config.MODEL_PATH}\n  → {config.METRICS_PATH}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the WattCast model.")
    parser.add_argument(
        "--model", default="xgboost", choices=["xgboost", "prophet"], help="model family"
    )
    args = parser.parse_args(argv)
    run(args.model)


if __name__ == "__main__":
    main()
