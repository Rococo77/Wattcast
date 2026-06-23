"""Walk-forward model bake-off with ``TimeSeriesSplit``.

A single train/test split can be lucky. This retrains each candidate model
across ``N_SPLITS`` expanding-window folds and reports the mean ± std of
MAE/MAPE, side by side with the naive ``lag_168`` baseline and RTE's J-1
forecast — an honest, leak-free read on generalisation.

Models compared:
* **XGBoost** — the production model.
* **Prophet** — additive seasonality + regressors (skipped if not installed,
  or with ``--no-prophet``).

Outputs ``model/evaluation.json`` (consumed by the dashboard) and prints a
comparison table.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ml import config
from ml.features import feature_columns
from ml.models import ProphetForecaster, build_forecaster
from ml.utils import mae, mape


def _evaluate_model(name: str, df: pd.DataFrame, features: list[str], splitter) -> dict:
    """Run one forecaster across every fold and aggregate its errors."""
    maes, mapes, fit_times = [], [], []
    for fold, (tr, te) in enumerate(splitter.split(df), start=1):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        t0 = time.time()
        forecaster = build_forecaster(name).fit(train_df, features)
        preds = forecaster.predict(test_df)
        fit_times.append(time.time() - t0)
        y = test_df[config.TARGET].to_numpy()
        maes.append(mae(y, preds))
        mapes.append(mape(y, preds))
        print(f"[eval] {name:<8} fold {fold}/{splitter.n_splits}  MAE={maes[-1]:8.1f}  ({fit_times[-1]:.1f}s)")
    return {
        "mae": {"mean": round(np.mean(maes), 2), "std": round(np.std(maes), 2)},
        "mape": {"mean": round(np.mean(mapes), 3), "std": round(np.std(mapes), 3)},
        "fit_seconds": round(float(np.mean(fit_times)), 1),
        "fold_mae": [round(v, 1) for v in maes],
    }


def _reference(col: str, df: pd.DataFrame, splitter) -> dict:
    """Aggregate a non-trained reference column (baseline / RTE) over the folds."""
    maes, mapes = [], []
    for _tr, te in splitter.split(df):
        test_df = df.iloc[te]
        y = test_df[config.TARGET].to_numpy()
        maes.append(mae(y, test_df[col].to_numpy()))
        mapes.append(mape(y, test_df[col].to_numpy()))
    return {
        "mae": {"mean": round(np.mean(maes), 2), "std": round(np.std(maes), 2)},
        "mape": {"mean": round(np.mean(mapes), 3), "std": round(np.std(mapes), 3)},
    }


def run(*, with_prophet: bool = True) -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.FEATURES_PATH).sort_index()
    features = feature_columns(df)
    splitter = TimeSeriesSplit(n_splits=config.N_SPLITS)

    model_names = ["xgboost"]
    if with_prophet and ProphetForecaster.is_available():
        model_names.append("prophet")
    elif with_prophet:
        print("[eval] prophet not installed → skipping (pip install prophet)")

    results = {name: _evaluate_model(name, df, features, splitter) for name in model_names}

    references = {"baseline_naive": _reference("lag_168", df, splitter)}
    if "rte_forecast" in df.columns:
        references["rte_official"] = _reference("rte_forecast", df, splitter)

    baseline_mae = references["baseline_naive"]["mae"]["mean"]
    best_name = min(results, key=lambda n: results[n]["mae"]["mean"])
    summary = {
        "n_splits": config.N_SPLITS,
        "models": results,
        "references": references,
        "best_model": best_name,
        "improvement_vs_baseline_pct": round(
            100 * (baseline_mae - results[best_name]["mae"]["mean"]) / baseline_mae, 1
        ),
    }

    out_path = config.MODEL_DIR / "evaluation.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Walk-forward bake-off (mean ± std MAE over folds) ===")
    rows = {**{f"{n} (model)": results[n] for n in results}, **references}
    for label, r in rows.items():
        print(f"  {label:<22} MAE {r['mae']['mean']:8.1f} ± {r['mae']['std']:6.1f} MW   MAPE {r['mape']['mean']:.2f} %")
    print(f"  → best: {best_name}  ({summary['improvement_vs_baseline_pct']:+.1f}% vs naive baseline)")
    print(f"  → {out_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward model bake-off.")
    parser.add_argument("--no-prophet", action="store_true", help="skip the Prophet benchmark")
    args = parser.parse_args(argv)
    run(with_prophet=not args.no_prophet)


if __name__ == "__main__":
    main()
