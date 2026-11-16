"""Fast, dependency-light checks for the pipeline.

Run directly (no pytest needed):

    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, features, models  # noqa: E402


def _small_frame(n: int = 400):
    df = data.load_dataset()
    return df.head(n).copy()


def test_domain_features():
    df = _small_frame()
    tr = features.DomainFeatures().fit(df)
    out = tr.transform(df)
    assert "duration" not in out.columns, "duration must be dropped (leakage)"
    assert out["contacted_before"].isin([0, 1]).all()
    never = out.loc[df["pdays"].values == 999, "pdays_known"]
    assert never.isna().all(), "pdays=999 must become missing, not 999"
    assert (out["campaign_log"] >= 0).all()


def test_pipeline_fit_predict():
    df = _small_frame()
    X, y = features.make_xy(df)
    pipe = models.SPECS["logistic"].build().fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(df),)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(p).all()


def test_metrics_and_thresholds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.2 * y + rng.normal(0.3, 0.15, 500), 0, 1)
    m = evaluate.classification_metrics(y, p, 0.4)
    for key in ("roc_auc", "average_precision", "brier", "f1"):
        assert 0 <= m[key] <= 1
    thr = evaluate.capacity_threshold(p, 0.2)
    assert 0 <= thr <= 1
    assert evaluate.precision_at_fraction(y, p, 0.2) >= 0


def test_no_leaky_feature_in_pipeline():
    names = features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES
    assert "duration" not in names


if __name__ == "__main__":
    for fn in [test_domain_features, test_pipeline_fit_predict,
               test_metrics_and_thresholds, test_no_leaky_feature_in_pipeline]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all tests passed")
