"""Data loading, integrity checks and train/test splitting.

Design notes
------------
* The source file is ordered by campaign date (May 2008 -> Nov 2010) and the
  positive rate rises from ~3% to ~46% across that period.  A pure temporal
  split is therefore a label-shift stress test, not a like-for-like benchmark.
  The primary split is a stratified random holdout; the temporal split is kept
  as an explicit robustness experiment (see ``train.validation_strategy_experiment``).
* ``duration`` is dropped at the feature-engineering step because it is only
  observed after a call.  ``data.load_dataset`` keeps it so the leakage
  experiment can put it back deliberately.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _download(url: str, dest: Path, attempts: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(part, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if not zipfile.is_zipfile(part):
                raise IOError("downloaded file is not a zip archive")
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            part.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url}: {last}")


def ensure_raw_data() -> Path:
    """Download and unpack the UCI archive if the CSV is not present."""
    if config.DATA_FILE.exists():
        return config.DATA_FILE
    archive = config.RAW_DIR / "bank+marketing.zip"
    if not archive.exists():
        _download(config.RAW_ZIP_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.RAW_DIR)
    for inner in ("bank.zip", "bank-additional.zip"):
        p = config.RAW_DIR / inner
        if p.exists():
            with zipfile.ZipFile(p) as zf:
                zf.extractall(config.RAW_DIR)
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"expected {config.DATA_FILE} after extraction")
    return config.DATA_FILE


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV, normalise names and encode the target as 0/1."""
    path = path or ensure_raw_data()
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_").strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df[config.TARGET] = (df[config.TARGET] == config.POSITIVE).astype(np.int8)
    return df


def audit(df: pd.DataFrame) -> dict:
    """Summarise data-quality facts used in the written report."""
    # pandas >= 3.0 stores strings as the ``str`` dtype, so ``dtype == object``
    # is not a reliable categorical test any more.
    cat_cols = [
        c for c in df.columns
        if c != config.TARGET and not pd.api.types.is_numeric_dtype(df[c])
    ]
    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_rate": float(df[config.TARGET].mean()),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "categorical_columns": len(cat_cols),
        "numeric_columns": int(df.shape[1] - len(cat_cols)),
        "cardinality": {c: int(df[c].nunique()) for c in cat_cols},
        "unknown_share": {
            c: float((df[c] == "unknown").mean())
            for c in cat_cols
            if (df[c] == "unknown").any()
        },
        "pdays_999_share": float((df["pdays"] == 999).mean()),
        "duration_mean": float(df["duration"].mean()),
    }
    return audit


def temporal_split(
    df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by row order, which is chronological in the source file."""
    cut = int(round(len(df) * (1.0 - test_fraction)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def random_split(
    df: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split, kept for the validation-strategy comparison."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=test_fraction, random_state=seed, stratify=df[config.TARGET]
    )
    return train.copy(), test.copy()


def campaign_month_id(df: pd.DataFrame) -> pd.Series:
    """Consecutive campaign-month blocks (the file is ordered by date)."""
    return (df["month"] != df["month"].shift()).cumsum()


def grouped_month_split(
    df: pd.DataFrame,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole campaign months, so no month appears on both sides.

    Macro columns (euribor3m, nr_employed, ...) are constant within a month,
    so a plain random split lets the model see the test period's macro state
    during training.  Grouping by month removes that shortcut.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = campaign_month_id(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                 random_state=seed)
    tr_idx, te_idx = next(splitter.split(df, groups=groups))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Compare covariate distributions across a split (drift evidence)."""
    numeric = [
        "age",
        "campaign",
        "previous",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]
    drift = {}
    for col in numeric:
        tr_mean, te_mean = float(train[col].mean()), float(test[col].mean())
        pooled = float(pd.concat([train[col], test[col]]).std()) or 1.0
        drift[col] = {
            "train_mean": tr_mean,
            "test_mean": te_mean,
            "std_diff": (te_mean - tr_mean) / pooled,
        }
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train[config.TARGET].mean()),
        "test_positive_rate": float(test[config.TARGET].mean()),
        "drift": drift,
        "train_months": sorted(train["month"].unique().tolist()),
        "test_months": sorted(test["month"].unique().tolist()),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
