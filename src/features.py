"""Feature engineering and preprocessing.

The transformer chain is deliberately explicit so that every derived column is
auditable:

    DomainFeatures  ->  ColumnTransformer  ->  model

``DomainFeatures`` adds three campaign-specific signals and removes the
post-call leak (``duration``).  The ``ColumnTransformer`` imputes, scales (only
where the model needs it) and one-hot encodes with rare-level grouping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "campaign_log",
    "previous",
    "previous_log",
    "pdays_known",
    "contacted_before",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

FEATURE_DOCS = {
    "pdays_known": "距离上次联系的天数；从未联系过的样本为缺失（原值 999 是哨兵值）",
    "contacted_before": "是否曾被联系过（pdays != 999）",
    "campaign_log": "本次活动中已联系次数取 log1p，压缩长尾",
    "previous_log": "此前活动中的联系次数取 log1p",
}


class DomainFeatures(BaseEstimator, TransformerMixin):
    """Add campaign-level derived features and drop post-call leakage.

    Parameters
    ----------
    drop_leaky:
        When True (default) the ``duration`` column is removed.  Set to False
        only for the explicit leakage experiment.
    """

    def __init__(self, drop_leaky: bool = True):
        self.drop_leaky = drop_leaky

    def fit(self, X: pd.DataFrame, y=None) -> "DomainFeatures":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatures expects a pandas DataFrame")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatures expects a pandas DataFrame")
        out = X.copy()

        pdays = pd.to_numeric(out["pdays"], errors="coerce").replace(999, np.nan)
        out["pdays_known"] = pdays
        out["contacted_before"] = (pdays.notna()).astype(np.int8)
        out["campaign_log"] = np.log1p(pd.to_numeric(out["campaign"], errors="coerce"))
        out["previous_log"] = np.log1p(pd.to_numeric(out["previous"], errors="coerce"))

        if self.drop_leaky:
            out = out.drop(columns=[c for c in config.LEAKY_FEATURES if c in out])
        return out

    def get_feature_names_out(self, input_features=None):
        cols = list(self.feature_names_in_)
        if self.drop_leaky:
            cols = [c for c in cols if c not in config.LEAKY_FEATURES]
        return np.asarray(cols + ["pdays_known", "contacted_before", "campaign_log",
                                  "previous_log"], dtype=object)


def build_preprocessor(scale_numeric: bool = True, min_frequency: int | float = 0.002,
                       drop_leaky: bool = True):
    """Column-wise preprocessing used by every model.

    ``min_frequency`` merges rare categorical levels (e.g. a handful of
    ``job=unknown`` records) into one ``infrequent_sklearn`` bucket, which keeps
    the one-hot matrix compact and avoids coefficients fitted on 3 rows.

    ``drop_leaky=False`` adds ``duration`` to the numeric block; it exists only
    for the leakage experiment.
    """
    numeric_features = list(NUMERIC_FEATURES)
    if not drop_leaky:
        numeric_features.append("duration")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(model, *, scale_numeric: bool = True, drop_leaky: bool = True,
                   min_frequency: int | float = 0.002) -> Pipeline:
    """Full estimator: domain features -> preprocessing -> classifier."""
    return Pipeline(
        [
            ("domain", DomainFeatures(drop_leaky=drop_leaky)),
            ("prep", build_preprocessor(scale_numeric=scale_numeric,
                                        drop_leaky=drop_leaky,
                                        min_frequency=min_frequency)),
            ("clf", model),
        ]
    )


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET].astype(int)
    return X, y


def expanded_feature_names(pipeline: Pipeline) -> list[str]:
    """Human-readable names of the columns the classifier actually sees."""
    prep = pipeline.named_steps["prep"]
    names = prep.get_feature_names_out()
    return [str(n).replace("num__", "").replace("cat__", "") for n in names]


def transform_frame(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted domain+prep steps and return a labelled DataFrame."""
    domain = pipeline.named_steps["domain"]
    prep = pipeline.named_steps["prep"]
    arr = prep.transform(domain.transform(X))
    return pd.DataFrame(arr, columns=expanded_feature_names(pipeline),
                        index=X.index)
