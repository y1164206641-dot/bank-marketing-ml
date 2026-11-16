"""Model zoo, hyper-parameter spaces and CV search helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
)

from . import config
from .features import build_pipeline


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    scale_numeric: bool = True
    space: dict = field(default_factory=dict)

    def build(self):
        return build_pipeline(self._estimator(), scale_numeric=self.scale_numeric)

    def _estimator(self):
        if self.key == "dummy":
            return DummyClassifier(strategy="prior")
        if self.key == "logistic":
            return LogisticRegression(max_iter=2000, solver="lbfgs",
                                      random_state=config.RANDOM_STATE)
        if self.key == "random_forest":
            return RandomForestClassifier(
                n_estimators=300, n_jobs=1, random_state=config.RANDOM_STATE
            )
        if self.key == "hist_gb":
            return HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.06,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=config.RANDOM_STATE,
            )
        raise KeyError(self.key)


SPECS = {
    "dummy": ModelSpec(
        key="dummy",
        label="Dummy（先验概率）",
        description="只输出训练集正类比例，作为最低基准",
    ),
    "logistic": ModelSpec(
        key="logistic",
        label="逻辑回归",
        description="线性可解释基准，L2 正则，数值特征标准化",
        space={
            "clf__C": loguniform(1e-3, 1e2),
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="随机森林",
        description="Bagging 树集成，不做超参搜索（作为未调参参照）",
        scale_numeric=False,
    ),
    "hist_gb": ModelSpec(
        key="hist_gb",
        label="HistGradientBoosting",
        description="直方图梯度提升树，主力模型，处理非线性与交互",
        scale_numeric=False,
        space={
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "clf__max_leaf_nodes": randint(15, 64),
            "clf__min_samples_leaf": randint(20, 101),
            "clf__l2_regularization": loguniform(1e-3, 10.0),
        },
    ),
}

# Models that are tuned; the rest are fitted once with defaults.
TUNED = ["logistic", "hist_gb"]


def default_pipelines() -> dict:
    return {k: spec.build() for k, spec in SPECS.items()}


def make_cv(n_splits: int = 4) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_STATE)


def search_space(key: str) -> dict:
    return dict(SPECS[key].space)


def tune(key: str, X, y, *, n_iter: int = 20, cv_splits: int = 4,
         n_jobs: int = 1, verbose: int = 1, groups=None) -> RandomizedSearchCV:
    """Random search on average precision with a small multi-metric panel.

    ``n_jobs`` defaults to 1: the DSH sandbox blocks the named pipes joblib's
    process backend needs, so nested parallelism fails with WinError 5.
    When ``groups`` is given (campaign month), folds never split a month.
    """
    cv = (GroupKFold(n_splits=cv_splits) if groups is not None
          else make_cv(cv_splits))
    search = RandomizedSearchCV(
        SPECS[key].build(),
        param_distributions=search_space(key),
        n_iter=n_iter,
        scoring={
            "ap": "average_precision",
            "roc_auc": "roc_auc",
            "neg_log_loss": "neg_log_loss",
        },
        refit="ap",
        cv=cv,
        n_jobs=n_jobs,
        random_state=config.RANDOM_STATE,
        verbose=verbose,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)
    return search


def describe_models() -> list[dict]:
    return [
        {"key": s.key, "label": s.label, "description": s.description,
         "tuned": s.key in TUNED}
        for s in SPECS.values()
    ]


def top_cv_rows(search: RandomizedSearchCV, n: int = 5):
    """Return the best few candidates as plain dicts for reporting."""
    res = search.cv_results_
    order = np.argsort(-res["mean_test_ap"])[:n]
    rows = []
    for i in order:
        params = {k: res["params"][i][k] for k in res["params"][i]}
        rows.append(
            {
                "params": params,
                "ap": float(res["mean_test_ap"][i]),
                "ap_std": float(res["std_test_ap"][i]),
                "roc_auc": float(res["mean_test_roc_auc"][i]),
                "log_loss": float(-res["mean_test_neg_log_loss"][i]),
            }
        )
    return rows
