"""End-to-end training entry point.

Run ``python run_all.py`` (or ``python -m src.train``) to reproduce every
number and figure in ``reports/``.  The pipeline is intentionally staged so
each artifact can be inspected on its own:

    1. audit + EDA
    2. 5-fold CV of the model zoo on the training period
    3. random search for the two tuned models
    4. model selection by CV average precision (never by the holdout)
    5. calibration choice on an inner, time-ordered validation slice
    6. threshold choice on out-of-fold training predictions
    7. one-shot holdout evaluation + bootstrap intervals
    8. leakage / validation-strategy experiments
    9. explanations and artifact export
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from . import config, data, evaluate, features, models
from .data import save_json
from .features import build_pipeline, expanded_feature_names, make_xy


# ------------------------------------------------------------------ utils --
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _fold_scores(estimator, X, y, cv, groups=None) -> pd.DataFrame:
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring={
            "ap": "average_precision",
            "roc_auc": "roc_auc",
            "neg_log_loss": "neg_log_loss",
            "brier": make_scorer(brier_score_loss, greater_is_better=False,
                                 response_method="predict_proba"),
        },
        n_jobs=1,
        return_train_score=False,
    )
    return pd.DataFrame(
        {
            "fold": np.arange(1, len(scores["test_ap"]) + 1),
            "average_precision": scores["test_ap"],
            "roc_auc": scores["test_roc_auc"],
            "log_loss": -scores["test_neg_log_loss"],
            "brier": -scores["test_brier"],
        }
    )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_pdp(model, X: pd.DataFrame, features: list[str],
                grid_resolution: int = 25) -> list[dict]:
    """Average partial dependence, computed manually.

    ``sklearn.inspection.partial_dependence`` hung indefinitely in this
    sandbox (likely its internal threading interacting with the blocked
    joblib backend), so the brute-force definition is implemented directly:
    for every grid value, overwrite one column and average the predictions.
    """
    rows = []
    for f in features:
        grid = np.quantile(X[f].astype(float), np.linspace(0.05, 0.95, grid_resolution))
        grid = np.unique(np.round(grid, 6))
        avg = []
        for v in grid:
            Xt = X.copy()
            Xt[f] = v
            avg.append(float(model.predict_proba(Xt)[:, 1].mean()))
        rows.append({"feature": f, "grid": grid.tolist(), "average": avg})
    return rows


# ------------------------------------------------------------- experiments --
def validation_strategy_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Random holdout vs. temporal holdout: what changes under drift?"""
    rows = []
    for name, splitter in (
        ("random stratified", lambda d: data.random_split(d)),
        ("grouped by campaign month", lambda d: data.grouped_month_split(d)),
        ("temporal (drift stress test)", lambda d: data.temporal_split(d)),
    ):
        tr, te = splitter(df)
        X_tr, y_tr = make_xy(tr)
        X_te, y_te = make_xy(te)
        est = models.SPECS["hist_gb"].build().fit(X_tr, y_tr)
        p = evaluate.proba(est, X_te)
        m = evaluate.classification_metrics(y_te, p, 0.5)
        rows.append(
            {
                "strategy": name,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_positive_rate": float(y_tr.mean()),
                "test_positive_rate": float(y_te.mean()),
                "average_precision": m["average_precision"],
                "roc_auc": m["roc_auc"],
                "brier": m["brier"],
                "precision_at_0.5": m["precision"],
                "recall_at_0.5": m["recall"],
            }
        )
    return pd.DataFrame(rows)


def leakage_experiment(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Quantify the optimism caused by the post-call ``duration`` feature."""
    X_tr, y_tr = make_xy(train)
    X_te, y_te = make_xy(test)
    rows = []
    for key in ("logistic", "hist_gb"):
        for drop in (True, False):
            est = build_pipeline(
                models.SPECS[key]._estimator(),  # fresh, untuned estimator
                scale_numeric=models.SPECS[key].scale_numeric,
                drop_leaky=drop,
            ).fit(X_tr, y_tr)
            p = evaluate.proba(est, X_te)
            m = evaluate.classification_metrics(y_te, p, 0.5)
            rows.append(
                {
                    "model": models.SPECS[key].label,
                    "with_duration": not drop,
                    "average_precision": m["average_precision"],
                    "roc_auc": m["roc_auc"],
                }
            )
    wide = (
        pd.DataFrame(rows)
        .pivot(index="model", columns="with_duration",
               values=["average_precision", "roc_auc"])
        .reset_index()
    )
    wide.columns = [
        "model",
        "ap_without_duration",
        "ap_with_duration",
        "auc_without_duration",
        "auc_with_duration",
    ]
    return wide[["model", "ap_without_duration", "ap_with_duration",
                 "auc_without_duration", "auc_with_duration"]]


def choose_calibration(base, X_tr, y_tr) -> tuple[str, dict]:
    """Pick raw / sigmoid / isotonic on an inner holdout slice."""
    from sklearn.model_selection import train_test_split

    Xa, Xb, ya, yb = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=config.RANDOM_STATE, stratify=y_tr
    )

    candidates = {"raw": clone(base).fit(Xa, ya)}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(base), method=method, cv=5)
        candidates[method] = cal.fit(Xa, ya)

    scores = {
        name: float(brier_score_loss(yb, est.predict_proba(Xb)[:, 1]))
        for name, est in candidates.items()
    }
    best = min(scores, key=scores.get)
    return best, scores


def fit_final(base, method: str, X_tr, y_tr):
    """Refit the tuned estimator (optionally calibrated) on the full train set."""
    if method == "raw":
        return clone(base).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(clone(base), method=method, cv=5)
    return cal.fit(X_tr, y_tr)


def threshold_decisions(y_val, p_val) -> dict:
    profit = evaluate.best_profit_threshold(y_val, p_val)
    capacity = evaluate.capacity_threshold(p_val, config.CAPACITY_FRACTION)
    return {
        "profit_optimal": profit,
        "capacity": {
            "threshold": capacity,
            "fraction": config.CAPACITY_FRACTION,
            "validation_profit_per_contact": evaluate.profit_per_contact(
                y_val, p_val, capacity
            ),
        },
    }


# -------------------------------------------------------------------- main --
def main(fast: bool = False, n_iter: dict | None = None) -> dict:
    t0 = time.time()
    rng_seed = config.RANDOM_STATE
    np.random.seed(rng_seed)

    n_iter = n_iter or {"logistic": 16, "hist_gb": 20}
    if fast:
        n_iter = {"logistic": 4, "hist_gb": 4}

    print("== 1. load + audit ==")
    df = data.load_dataset()
    audit = data.audit(df)
    save_json(_jsonable(audit), config.ARTIFACT_DIR / "data_audit.json")
    print(f"rows={audit['rows']} positive_rate={audit['target_rate']:.3%} "
          f"duplicates={audit['duplicate_rows']}")

    evaluate.plot_target_and_campaign(df)
    evaluate.plot_eda_rates(df)
    evaluate.plot_monthly_trend(df)
    evaluate.plot_label_shift(df)

    print("== 2. primary split: whole campaign months held out ==")
    train, test = data.grouped_month_split(df)
    groups_tr = data.campaign_month_id(train).to_numpy()
    split = data.split_summary(train, test)
    save_json(_jsonable(split), config.ARTIFACT_DIR / "split_summary.json")
    evaluate.plot_split_drift(split)
    X_tr, y_tr = make_xy(train)
    X_te, y_te = make_xy(test)
    print(f"train={len(train)} test={len(test)} months={len(np.unique(groups_tr))} "
          f"positive rate {y_tr.mean():.2%} -> {y_te.mean():.2%}")

    print("== 3. model-zoo CV (GroupKFold by campaign month) ==")
    cv = GroupKFold(n_splits=5)
    cv_parts = []
    for key, spec in models.SPECS.items():
        part = _fold_scores(spec.build(), X_tr, y_tr, cv, groups_tr)
        part.insert(0, "model", spec.label)
        part.insert(1, "key", key)
        cv_parts.append(part)
        print(f"  {spec.label:26s} AP={part['average_precision'].mean():.4f} "
              f"+/- {part['average_precision'].std():.4f}")
    cv_df = pd.concat(cv_parts, ignore_index=True)
    cv_df.to_csv(config.ARTIFACT_DIR / "cv_model_zoo.csv", index=False)
    evaluate.plot_model_comparison(cv_df)

    print("== 4. random search ==")
    tuning_summary = {}
    searches = {}
    for key in models.TUNED:
        n = n_iter.get(key, 20)
        print(f"  tuning {key} ({n} candidates x 4 folds)")
        search = models.tune(key, X_tr, y_tr, n_iter=n, cv_splits=4, verbose=0,
                             groups=groups_tr)
        searches[key] = search
        tuning_summary[key] = {
            "best_cv_ap": float(search.best_score_),
            "best_cv_se": float(
                search.cv_results_["std_test_ap"][search.best_index_]
            ),
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
            "n_candidates": n,
            "top": _jsonable(models.top_cv_rows(search, 5)),
        }
        joblib.dump(search, config.ARTIFACT_DIR / f"search_{key}.joblib")
        print(f"  best AP={search.best_score_:.4f} params={search.best_params_}")
    save_json(_jsonable(tuning_summary), config.ARTIFACT_DIR / "tuning.json")

    # --- select by CV with the one-standard-error rule --------------------
    # If a simpler model is within 1 SE of the best CV score, prefer it: the
    # extra complexity is not supported by the data.
    best_key = max(tuning_summary, key=lambda k: tuning_summary[k]["best_cv_ap"])
    threshold_ap = (tuning_summary[best_key]["best_cv_ap"]
                    - tuning_summary[best_key]["best_cv_se"])
    simplicity = {"logistic": 0, "hist_gb": 1}
    eligible = [k for k in tuning_summary
                if tuning_summary[k]["best_cv_ap"] >= threshold_ap]
    selected_key = min(eligible, key=lambda k: simplicity[k])
    print(f"== 5. selected model: {selected_key} "
          f"(CV AP {tuning_summary[selected_key]['best_cv_ap']:.4f}, "
          f"best {best_key} {tuning_summary[best_key]['best_cv_ap']:.4f}) ==")

    method, cal_scores = choose_calibration(searches[selected_key].best_estimator_,
                                            X_tr, y_tr)
    print(f"  calibration: {method} {cal_scores}")
    final_model = fit_final(searches[selected_key].best_estimator_, method, X_tr, y_tr)

    print("== 6. threshold selection on out-of-fold predictions ==")
    oof = cross_val_predict(final_model, X_tr, y_tr, cv=cv, groups=groups_tr,
                            method="predict_proba", n_jobs=1)[:, 1]
    decisions = threshold_decisions(y_tr, oof)
    print(f"  profit threshold={decisions['profit_optimal']['threshold']:.4f} "
          f"contact_rate={decisions['profit_optimal']['contact_rate']:.1%}")

    print("== 7. holdout evaluation ==")
    holdout_preds = {}
    for key in ("logistic", "hist_gb"):
        if key in searches:
            holdout_preds[models.SPECS[key].label] = evaluate.proba(
                searches[key].best_estimator_, X_te
            )
    holdout_preds[models.SPECS["random_forest"].label] = evaluate.proba(
        models.SPECS["random_forest"].build().fit(X_tr, y_tr), X_te
    )
    final_name = models.SPECS[selected_key].label + (
        " + 校准" if method != "raw" else ""
    )
    p_final = evaluate.proba(final_model, X_te)
    holdout_preds[final_name] = p_final

    # Capacity policy = top X% of *this* batch: the cut is a quantile of the
    # scored list, not a fixed probability threshold.
    batch_thresholds = {
        name: evaluate.capacity_threshold(p, config.CAPACITY_FRACTION)
        for name, p in holdout_preds.items()
    }
    holdout_metrics = {
        name: evaluate.classification_metrics(y_te, p, batch_thresholds[name])
        for name, p in holdout_preds.items()
    }
    # also at 0.5 for reference
    for name, p in holdout_preds.items():
        holdout_metrics[name]["metrics_at_0.5"] = evaluate.classification_metrics(
            y_te, p, 0.5
        )
    final_cap_threshold = batch_thresholds[final_name]

    ci_ap = evaluate.bootstrap_ci(y_te, p_final, "average_precision", 1000)
    ci_auc = evaluate.bootstrap_ci(y_te, p_final, "roc_auc", 1000)
    other_key = "hist_gb" if selected_key == "logistic" else "logistic"
    other_label = models.SPECS[other_key].label
    diff_vs_other = evaluate.bootstrap_paired_diff(
        y_te, p_final, holdout_preds[other_label], "average_precision", 1000
    )

    profit_profit = evaluate.compare_to_no_model(
        y_te, p_final, decisions["profit_optimal"]["threshold"]
    )
    profit_capacity = evaluate.compare_to_no_model(
        y_te, p_final, final_cap_threshold
    )
    profit_curve = evaluate.profit_curve(y_te, p_final)
    sensitivity = evaluate.profit_sensitivity(y_te, p_final)
    profit_curve.to_csv(config.ARTIFACT_DIR / "profit_curve.csv", index=False)
    sensitivity.to_csv(config.ARTIFACT_DIR / "profit_sensitivity.csv", index=False)

    lift = evaluate.lift_table(y_te, p_final, 10)
    lift.to_csv(config.ARTIFACT_DIR / "lift_table.csv", index=False)

    evaluate.plot_roc_pr(y_te, {k: v for k, v in holdout_preds.items()
                                if k != models.SPECS["dummy"].label})
    evaluate.plot_calibration(y_te, {
        other_label: holdout_preds[other_label],
        final_name: p_final,
    })
    evaluate.plot_confusion(y_te, p_final, final_cap_threshold,
                            title=f"{final_name} @ top-{config.CAPACITY_FRACTION:.0%}")
    evaluate.plot_lift(y_te, p_final)
    threshold_curve = evaluate.profit_curve(
        y_tr, oof, value=config.VALUE_PER_CONVERSION, cost=config.COST_PER_CONTACT
    )
    evaluate.plot_threshold_profit(
        threshold_curve,
        {
            "profit-optimal": decisions["profit_optimal"]["threshold"],
            "top 20% capacity": decisions["capacity"]["threshold"],
        },
    )
    evaluate.plot_profit_sensitivity(sensitivity)

    print("== 8. leakage + validation strategy ==")
    leakage = leakage_experiment(train, test)
    leakage.to_csv(config.ARTIFACT_DIR / "leakage_experiment.csv", index=False)
    evaluate.plot_leakage(leakage)
    strategy = validation_strategy_experiment(df)
    strategy.to_csv(config.ARTIFACT_DIR / "validation_strategy.csv", index=False)

    print("== 9. explanations ==", flush=True)
    perm = permutation_importance(
        final_model, X_te, y_te, scoring="average_precision", n_repeats=8,
        random_state=rng_seed, n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {"feature": X_te.columns, "mean": perm.importances_mean,
         "std": perm.importances_std}
    ).sort_values("mean", ascending=False)
    perm_df.to_csv(config.ARTIFACT_DIR / "permutation_importance.csv", index=False)
    evaluate.plot_permutation_importance(perm, list(X_te.columns), top_n=15)

    pdp_features = ["euribor3m", "nr_employed", "age", "campaign"]
    X_pdp = X_te.sample(n=min(1500, len(X_te)), random_state=rng_seed).copy()
    X_pdp[pdp_features] = X_pdp[pdp_features].astype(float)
    print("  partial dependence ...", flush=True)
    try:
        pdp_rows = compute_pdp(final_model, X_pdp, pdp_features,
                               grid_resolution=25)
        save_json(_jsonable({"curves": pdp_rows}), config.ARTIFACT_DIR / "pdp.json")
        evaluate.plot_pdp_curves(pdp_rows, X_te)
    except Exception as exc:  # noqa: BLE001 - never lose the whole run
        print(f"  PDP skipped: {exc}", flush=True)

    logit = searches.get("logistic")
    coef_rows = []
    if logit is not None:
        est = logit.best_estimator_
        names = expanded_feature_names(est)
        coefs = est.named_steps["clf"].coef_.ravel()
        coef_df = pd.DataFrame({"feature": names, "coef": coefs})
        coef_df["odds_ratio"] = np.exp(coef_df["coef"])
        coef_df = coef_df.sort_values("coef", ascending=False)
        coef_df.to_csv(config.ARTIFACT_DIR / "logistic_coefficients.csv", index=False)
        evaluate.plot_logistic_coefficients(coef_df)
        coef_rows = _jsonable(coef_df.head(10).to_dict("records"))

    print("== 10. save artifacts ==")
    joblib.dump(final_model, config.ARTIFACT_DIR / "model.joblib")
    meta = {
        "model_key": selected_key,
        "model_label": final_name,
        "calibration": method,
        "calibration_brier": cal_scores,
        "thresholds": decisions,
        "business": {
            "value_per_conversion": config.VALUE_PER_CONVERSION,
            "cost_per_contact": config.COST_PER_CONTACT,
            "capacity_fraction": config.CAPACITY_FRACTION,
        },
        "features": features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256_16": _sha256(config.DATA_FILE),
        "sklearn_version": __import__("sklearn").__version__,
    }
    save_json(_jsonable(meta), config.ARTIFACT_DIR / "model_meta.json")

    test_out = test[["job", "month", "poutcome", "contact", "age", "campaign"]].copy()
    test_out["y"] = y_te.values
    test_out["p"] = p_final
    test_out.to_csv(config.ARTIFACT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_sec": round(time.time() - t0, 1),
        "dataset": audit,
        "split": split,
        "model_zoo_cv": {
            label: {
                "ap_mean": float(g["average_precision"].mean()),
                "ap_std": float(g["average_precision"].std()),
                "auc_mean": float(g["roc_auc"].mean()),
                "log_loss_mean": float(g["log_loss"].mean()),
            }
            for label, g in cv_df.groupby("model")
        },
        "tuning": tuning_summary,
        "selected": {"key": selected_key, "label": final_name, "calibration": method},
        "holdout": holdout_metrics,
        "final": {
            "name": final_name,
            "metrics_at_capacity": holdout_metrics[final_name],
            "metrics_at_0.5": holdout_metrics[final_name]["metrics_at_0.5"],
            "bootstrap": {"ap_ci95": ci_ap, "auc_ci95": ci_auc},
            "paired_ap_vs_other": diff_vs_other,
            "other_model": other_label,
            "capacity_threshold": final_cap_threshold,
            "profit": {"profit_optimal": profit_profit,
                       "capacity": profit_capacity},
            "precision_at_20pct": evaluate.precision_at_fraction(y_te, p_final, 0.2),
        },
        "leakage": _jsonable(leakage.to_dict("records")),
        "validation_strategy": _jsonable(strategy.to_dict("records")),
        "top_coefficients": coef_rows,
    }
    save_json(_jsonable(metrics), config.ARTIFACT_DIR / "metrics.json")
    print(f"done in {metrics['runtime_sec']}s -> {config.ARTIFACT_DIR}")
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="tiny run for smoke tests")
    args = ap.parse_args()
    main(fast=args.fast)

