"""End-to-end smoke test: ingest → features → train → evaluate → predict.

Everything runs against a tmp-redirected config (no repo files touched) and the
synthetic generator (no network). Forces the offline temperature path so the
test is deterministic and isolated.
"""

import json

from ml import evaluate, features, ingest, predict, train


def test_full_pipeline(tmp_config, monkeypatch):
    # Force predict's weather forecast onto the deterministic offline path.
    monkeypatch.setattr(
        ingest, "fetch_weather", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    ingest.run("2022-01-01", "2023-03-31", synthetic=True)
    assert tmp_config.RAW_CONSUMPTION.exists()
    assert tmp_config.RAW_WEATHER.exists()

    features.run()
    assert tmp_config.FEATURES_PATH.exists()

    metrics = train.run("xgboost")
    assert tmp_config.MODEL_PATH.exists()
    # The headline claim, asserted on held-out data.
    assert metrics["metrics"]["mae"] < metrics["references"]["baseline_naive"]["mae"]
    assert metrics["improvement_vs_baseline_pct"] > 0

    summary = evaluate.run(with_prophet=False)  # xgboost-only keeps it fast
    assert summary["best_model"] == "xgboost"
    assert (tmp_config.MODEL_DIR / "evaluation.json").exists()

    predict.run()
    payload = json.loads(tmp_config.PREDICTIONS_JSON.read_text())
    assert payload["horizon_hours"] == tmp_config.HORIZON
    assert len(payload["points"]) > tmp_config.HORIZON
    # There must be a forecast block (is_forecast) beyond the last actual.
    assert sum(p["is_forecast"] for p in payload["points"]) == tmp_config.HORIZON

    actuals = json.loads(tmp_config.ACTUALS_JSON.read_text())
    assert len(actuals["points"]) > 0
    # The dashboard's evaluation file is propagated for the bake-off table.
    assert (tmp_config.WEB_DATA_DIR / "evaluation.json").exists()
