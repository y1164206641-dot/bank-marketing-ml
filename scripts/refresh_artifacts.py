"""Recompute descriptive artifacts without retraining the models.

    python scripts/refresh_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src import config, data, evaluate, train
from src.data import save_json
from src.features import make_xy


def main() -> None:
    df = data.load_dataset()
    model = joblib.load(config.ARTIFACT_DIR / "model.joblib")
    meta = json.loads((config.ARTIFACT_DIR / "model_meta.json").read_text("utf-8"))
    pred = pd.read_csv(config.ARTIFACT_DIR / "test_predictions.csv")

    # 1) audit + EDA figures
    audit = data.audit(df)
    save_json(audit, config.ARTIFACT_DIR / "data_audit.json")
    evaluate.plot_target_and_campaign(df)
    evaluate.plot_eda_rates(df)
    evaluate.plot_monthly_trend(df)
    evaluate.plot_label_shift(df)
    split = json.loads((config.ARTIFACT_DIR / "split_summary.json").read_text("utf-8"))
    evaluate.plot_split_drift(split)

    # 2) validation comparison
    strategy = train.validation_strategy_experiment(df)
    strategy.to_csv(config.ARTIFACT_DIR / "validation_strategy.csv", index=False)

    # 3) PDP with the calibrated final model (probabilities on the right scale)
    _, test = data.grouped_month_split(df)
    X_test, _ = make_xy(test)
    pdp_features = ["euribor3m", "nr_employed", "age", "campaign"]
    X_pdp = X_test.sample(n=1500, random_state=config.RANDOM_STATE).copy()
    X_pdp[pdp_features] = X_pdp[pdp_features].astype(float)
    pdp_rows = train.compute_pdp(model, X_pdp, pdp_features, grid_resolution=25)
    save_json({"curves": pdp_rows}, config.ARTIFACT_DIR / "pdp.json")
    evaluate.plot_pdp_curves(pdp_rows, X_test)

    # 4) profit curve on the holdout (matches the reported business numbers)
    curve = evaluate.profit_curve(pred["y"], pred["p"])
    curve.to_csv(config.ARTIFACT_DIR / "profit_curve.csv", index=False)
    evaluate.plot_threshold_profit(
        curve,
        {
            "profit-optimal": meta["thresholds"]["profit_optimal"]["threshold"],
            "top 20% capacity": float(np.quantile(pred["p"], 1 - config.CAPACITY_FRACTION)),
        },
    )

    metrics = json.loads((config.ARTIFACT_DIR / "metrics.json").read_text("utf-8"))
    metrics["dataset"] = audit
    metrics["validation_strategy"] = strategy.to_dict("records")
    save_json(metrics, config.ARTIFACT_DIR / "metrics.json")
    print("refreshed")


if __name__ == "__main__":
    main()
