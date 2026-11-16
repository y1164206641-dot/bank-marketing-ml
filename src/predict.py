"""Batch scoring for new campaign lists.

    python -m src.predict --input new_calls.csv --output scored.csv

The input must contain the 20 raw modelling columns (``duration`` is optional
and ignored).  Missing optional columns raise a clear error instead of silently
scoring NaNs, because a silent mismatch is the most common production failure
for tabular models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config

REQUIRED = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor3m", "nr_employed",
]


def load_model(path: Path | None = None):
    path = path or config.ARTIFACT_DIR / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run run_all.py first")
    return joblib.load(path)


def score(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    # Accept raw UCI headers (emp.var.rate) as well as normalised ones.
    df = df.rename(columns=lambda c: str(c).strip().replace(".", "_"))
    df = df.drop(columns=[c for c in ("y",) if c in df.columns])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    X = df.copy()
    if "duration" in X.columns:
        X = X.drop(columns=["duration"])
    out = df.copy()
    out["p_subscribe"] = model.predict_proba(X)[:, 1]
    out["contact"] = (out["p_subscribe"] >= threshold).astype(int)
    out["rank"] = out["p_subscribe"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("p_subscribe", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="scored.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    model = load_model(Path(args.model) if args.model else None)
    meta = pd.read_json(config.ARTIFACT_DIR / "model_meta.json", typ="series")
    threshold = args.threshold or float(meta["thresholds"]["capacity"]["threshold"])

    df = pd.read_csv(args.input, sep=None, engine="python")
    scored = score(df, model, threshold)
    scored.to_csv(args.output, index=False)
    print(f"scored {len(scored)} rows -> {args.output} "
          f"(threshold={threshold:.4f}, flagged={scored['contact'].sum()})")


if __name__ == "__main__":
    main()
