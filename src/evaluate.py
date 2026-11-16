"""Metrics, business simulation and all diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config

# ---------------------------------------------------------------- style ----
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#c0392b"
BLUE = "#2c6e9b"
GREEN = "#2e7d5b"
GRID = "#dfe3e8"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)

PALETTE = [BLUE, ACCENT, GREEN, "#8e6c8a", "#c9a227", MUTED]


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------- metrics ----
def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alert_rate": float(yhat.mean()),
        "accuracy": float((tp + tn) / len(y)),
    }


def precision_at_fraction(y, p, fraction: float) -> float:
    k = max(1, int(round(len(p) * fraction)))
    order = np.argsort(-np.asarray(p))[:k]
    return float(np.asarray(y)[order].mean())


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                           n_bins, labels=range(1, n_bins + 1))
    base = df["y"].mean()
    rows = []
    for d, g in df.groupby("decile", observed=True):
        rows.append(
            {
                "decile": int(d),
                "n": int(len(g)),
                "positive_rate": float(g["y"].mean()),
                "lift": float(g["y"].mean() / base) if base else 0.0,
                "captured_positive": float(g["y"].sum()),
                "capture_share": float(g["y"].sum() / df["y"].sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, metric: str = "average_precision", n_boot: int = 1000,
                 seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    fn = average_precision_score if metric == "average_precision" else roc_auc_score
    values = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            values[i] = np.nan
            continue
        values[i] = fn(y[idx], p[idx])
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_paired_diff(y, p_a, p_b, metric: str = "average_precision",
                          n_boot: int = 1000,
                          seed: int = config.RANDOM_STATE) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on the same holdout rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    fn = average_precision_score if metric == "average_precision" else roc_auc_score
    diffs = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            diffs[i] = np.nan
            continue
        diffs[i] = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])
    diffs = diffs[~np.isnan(diffs)]
    observed = fn(y, p_a) - fn(y, p_b)
    return {
        "observed": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


# ------------------------------------------------------------ business ----
def profit_per_contact(y, p, threshold: float, value: float = config.VALUE_PER_CONVERSION,
                       cost: float = config.COST_PER_CONTACT) -> float:
    y = np.asarray(y)
    contacted = np.asarray(p) >= threshold
    if contacted.sum() == 0:
        return 0.0
    return float((y[contacted] * value - cost).sum() / len(y))


def profit_curve(y, p, value: float = config.VALUE_PER_CONVERSION,
                 cost: float = config.COST_PER_CONTACT,
                 n_points: int = 200) -> pd.DataFrame:
    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, n_points)))
    rows = []
    for t in thresholds:
        contacted = np.asarray(p) >= t
        n = int(contacted.sum())
        conv = float(np.asarray(y)[contacted].sum())
        profit = conv * value - n * cost
        rows.append(
            {
                "threshold": float(t),
                "contact_rate": n / len(p),
                "n_contacted": n,
                "conversions": conv,
                "profit": profit,
                "profit_per_contact": profit / len(p),
            }
        )
    return pd.DataFrame(rows)


def best_profit_threshold(y, p, value: float = config.VALUE_PER_CONVERSION,
                          cost: float = config.COST_PER_CONTACT) -> dict:
    curve = profit_curve(y, p, value, cost, n_points=300)
    row = curve.loc[curve["profit"].idxmax()]
    return {
        "threshold": float(row["threshold"]),
        "contact_rate": float(row["contact_rate"]),
        "profit": float(row["profit"]),
        "profit_per_contact": float(row["profit_per_contact"]),
    }


def capacity_threshold(p, fraction: float = config.CAPACITY_FRACTION) -> float:
    return float(np.quantile(p, 1.0 - fraction))


def compare_to_no_model(y, p, threshold: float,
                        value: float = config.VALUE_PER_CONVERSION,
                        cost: float = config.COST_PER_CONTACT) -> dict:
    y = np.asarray(y)
    n = len(y)
    base_profit = float(y.sum() * value - n * cost)
    model_profit = float((y[(np.asarray(p) >= threshold)] * value - cost).sum())
    return {
        "contact_everyone_profit": base_profit,
        "model_profit": model_profit,
        "uplift": model_profit - base_profit,
        "contact_everyone_per_person": base_profit / n,
        "model_per_person": model_profit / n,
        "uplift_per_person": (model_profit - base_profit) / n,
        "contact_everyone_rate": float(y.mean()),
    }


def profit_sensitivity(y, p, values=(40, 80, 160), costs=(2, 5, 10)) -> pd.DataFrame:
    rows = []
    for v in values:
        for c in costs:
            best = best_profit_threshold(y, p, v, c)
            base = float(np.asarray(y).sum() * v - len(y) * c)
            rows.append(
                {
                    "value": v,
                    "cost": c,
                    "threshold": best["threshold"],
                    "contact_rate": best["contact_rate"],
                    "profit_per_contact": best["profit_per_contact"],
                    "contact_everyone": base / len(y),
                    "uplift_per_contact": best["profit_per_contact"] - base / len(y),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------- plots ----
def plot_target_and_campaign(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    counts = df[config.TARGET].value_counts().sort_index()
    axes[0].bar(["No (0)", "Yes (1)"], counts.values, color=[MUTED, ACCENT], width=0.55)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom",
                     fontsize=8)
    axes[0].set_title("Target balance")
    axes[0].set_ylim(0, counts.max() * 1.22)

    axes[1].hist(df["age"], bins=30, color=BLUE, alpha=0.85)
    axes[1].axvline(df["age"].median(), color=ACCENT, ls="--", lw=1)
    axes[1].set_title("Age distribution")
    axes[1].set_xlabel("age")

    axes[2].hist(np.log1p(df["duration"]), bins=40, color=GREEN, alpha=0.85)
    axes[2].set_title("Call duration (log1p)")
    axes[2].set_xlabel("log1p(seconds)")
    axes[2].text(0.97, 0.92, "leaky feature - dropped", transform=axes[2].transAxes,
                 ha="right", fontsize=7.5, color=ACCENT)
    return _save(fig, "01_target_and_duration")


def plot_eda_rates(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    specs = [
        ("job", axes[0, 0]),
        ("month", axes[0, 1]),
        ("poutcome", axes[1, 0]),
        ("contact", axes[1, 1]),
    ]
    overall = df[config.TARGET].mean()
    for col, ax in specs:
        rate = df.groupby(col)[config.TARGET].mean().sort_values()
        n = df[col].value_counts().reindex(rate.index)
        bars = ax.barh(rate.index.astype(str), rate.values, height=0.62,
                       color=[ACCENT if v > overall else BLUE for v in rate.values],
                       alpha=0.9)
        ax.axvline(overall, color=INK, ls=":", lw=1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"Subscription rate by {col}")
        ax.tick_params(axis="y", labelsize=7.5)
        xmax = max(rate.max() * 1.62, overall * 1.8)
        for bar, v, cnt in zip(bars, rate.values, n.values):
            ax.text(v + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%} (n={cnt:,})", va="center", fontsize=6.4, color=INK)
        ax.set_xlim(0, xmax)
    fig.suptitle(f"Overall subscription rate = {overall:.2%}", y=1.0, fontsize=9,
                 color=MUTED)
    return _save(fig, "02_eda_subscription_rates")


def plot_monthly_trend(df: pd.DataFrame) -> Path:
    """Campaign months are contiguous blocks in the source order."""
    seg = (df["month"] != df["month"].shift()).cumsum()
    g = df.groupby(seg).agg(
        month=("month", "first"),
        rate=(config.TARGET, "mean"),
        euribor=("euribor3m", "mean"),
        n=(config.TARGET, "size"),
    )
    x = np.arange(len(g))
    fig, ax1 = plt.subplots(figsize=(9.2, 3.4))
    ax1.plot(x, g["rate"], color=ACCENT, marker="o", ms=3, lw=1.6,
             label="subscription rate")
    ax1.set_ylabel("subscription rate", color=ACCENT)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.tick_params(axis="y", colors=ACCENT)
    ax2 = ax1.twinx()
    ax2.plot(x, g["euribor"], color=BLUE, lw=1.4, ls="--", label="euribor3m")
    ax2.set_ylabel("euribor3m", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(g["month"], rotation=90, fontsize=7)
    ax1.set_title("Monthly subscription rate vs. 3-month Euribor (source order)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=7)
    return _save(fig, "03_monthly_trend")


def plot_split_drift(summary: dict) -> Path:
    drift = summary["drift"]
    cols = list(drift)
    vals = [drift[c]["std_diff"] for c in cols]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    colors = [ACCENT if abs(v) > 0.2 else BLUE for v in vals]
    ax.bar(cols, vals, color=colors, alpha=0.9)
    ax.axhline(0, color=INK, lw=0.8)
    for thresh, ls in ((0.2, "--"), (-0.2, "--")):
        ax.axhline(thresh, color=MUTED, lw=0.8, ls=ls)
    ax.set_ylabel("standardised mean difference")
    ax.set_title("Train vs. holdout covariate balance")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "04_split_drift")


def plot_label_shift(df: pd.DataFrame, n_blocks: int = 10) -> Path:
    """The campaign's success rate is not stationary over time."""
    block = pd.cut(np.arange(len(df)), n_blocks, labels=False)
    g = df.assign(_block=block).groupby("_block").agg(
        rate=(config.TARGET, "mean"),
        month=("month", "first"),
        euribor=("euribor3m", "mean"),
    )
    x = np.arange(len(g))
    fig, ax1 = plt.subplots(figsize=(7.4, 3.1))
    bars = ax1.bar(x, g["rate"], color=ACCENT, alpha=0.85, width=0.7)
    for xi, v in zip(x, g["rate"]):
        ax1.text(xi, v + 0.008, f"{v:.1%}", ha="center", fontsize=7, color=INK)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("subscription rate", color=ACCENT)
    ax1.set_xticks(x, g["month"], fontsize=7.5)
    ax1.set_xlabel("chronological block (each ~10% of rows)")
    ax1.set_title("Non-stationary target: subscription rate by time block")
    ax2 = ax1.twinx()
    ax2.plot(x, g["euribor"], color=BLUE, marker="o", ms=3, lw=1.3)
    ax2.set_ylabel("euribor3m", color=BLUE)
    ax2.grid(False)
    return _save(fig, "04b_label_shift")


def plot_model_comparison(cv_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    order = cv_df.groupby("model")["average_precision"].mean().sort_values().index
    data = [cv_df.loc[cv_df["model"] == m, "average_precision"].values for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(order) + 1), order)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for med in bp["medians"]:
        med.set_color(INK)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, s=14, color=INK, alpha=0.7,
                   zorder=3)
    ax.set_ylabel("average precision (5-fold CV)")
    ax.set_title("Model comparison on the training period")
    ax.tick_params(axis="x", labelsize=8)
    return _save(fig, "05_model_comparison")


def plot_roc_pr(y, preds: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for i, (name, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.6,
                     label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=1.6,
                     label=f"{name} (AP={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                title="ROC curve (holdout)")
    axes[1].axhline(np.mean(y), color=MUTED, ls=":", lw=1, label="base rate")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall (holdout)")
    for ax in axes:
        ax.legend(fontsize=7, loc="lower left" if ax is axes[0] else "upper right")
    return _save(fig, "06_roc_pr")


def plot_calibration(y, preds: dict, n_bins: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="perfect")
    for i, (name, p) in enumerate(preds.items()):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=3.5, lw=1.5,
                color=PALETTE[i % len(PALETTE)],
                label=f"{name} (Brier={brier_score_loss(y, p):.3f})")
    ax.set(xlabel="mean predicted probability", ylabel="observed positive rate",
           title="Calibration on holdout")
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, "07_calibration")


def plot_confusion(y, p, threshold: float, title: str = "Confusion matrix") -> Path:
    yhat = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10,
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["pred no", "pred yes"])
    ax.set_yticks([0, 1], ["actual no", "actual yes"])
    ax.set_title(f"{title}\nthreshold={threshold:.3f}", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, "08_confusion")


def plot_lift(y, p) -> Path:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values(
        "p", ascending=False
    )
    cum = df["y"].cumsum() / df["y"].sum()
    frac = np.arange(1, len(df) + 1) / len(df)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(frac, cum, color=ACCENT, lw=1.8)
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    axes[0].set(xlabel="share of list contacted (ranked by score)",
                ylabel="share of all conversions captured", title="Cumulative gains")
    axes[0].axvline(config.CAPACITY_FRACTION, color=BLUE, ls="--", lw=1)
    k = int(len(df) * config.CAPACITY_FRACTION)
    axes[0].annotate(f"top {config.CAPACITY_FRACTION:.0%}: {cum.iloc[k - 1]:.1%}",
                     xy=(config.CAPACITY_FRACTION, cum.iloc[k - 1]),
                     xytext=(config.CAPACITY_FRACTION + 0.08, cum.iloc[k - 1] - 0.12),
                     fontsize=7.5, color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    table = lift_table(y, p, 10).sort_values("decile")
    axes[1].bar(table["decile"].astype(str), table["lift"], color=BLUE, alpha=0.85)
    axes[1].axhline(1.0, color=INK, lw=0.8, ls="--")
    axes[1].set(xlabel="decile (1 = highest score)", ylabel="lift vs. base rate",
                title="Lift by decile")
    for x, v in zip(table["decile"].astype(str), table["lift"]):
        axes[1].text(x, v + 0.05, f"{v:.1f}x", ha="center", fontsize=7)
    return _save(fig, "09_lift")


def plot_threshold_profit(curve: pd.DataFrame, chosen: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(curve["contact_rate"], curve["profit_per_contact"], color=ACCENT, lw=1.8)
    top = float(curve["profit_per_contact"].max())
    ax.set_ylim(min(0, float(curve["profit_per_contact"].min())), top * 1.22)
    for name, t in chosen.items():
        row = curve.iloc[(curve["threshold"] - t).abs().idxmin()]
        ax.scatter(row["contact_rate"], row["profit_per_contact"], s=30,
                   color=BLUE, zorder=3)
        dy = 10 if "profit" in name else -16
        ax.annotate(name, (row["contact_rate"], row["profit_per_contact"]),
                    textcoords="offset points", xytext=(7, dy), fontsize=7.5,
                    color=BLUE)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="share of list contacted", ylabel="profit per person (EUR)",
           title="Campaign profit vs. contact volume (holdout)")
    return _save(fig, "10_threshold_profit")


def plot_profit_sensitivity(table: pd.DataFrame) -> Path:
    pivot = table.pivot(index="value", columns="cost", values="uplift_per_contact")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{c:.0f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:.0f}" for v in pivot.index])
    ax.set(xlabel="cost per contact (EUR)", ylabel="value per conversion (EUR)",
           title="Profit uplift vs. contacting everyone")
    for (i, j), v in np.ndenumerate(pivot.values):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="black" if abs(v) < 25 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, "11_profit_sensitivity")


def plot_permutation_importance(result, feature_names, top_n: int = 15) -> Path:
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("mean")
    df = df.tail(top_n)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(df["feature"], df["mean"], xerr=df["std"], color=BLUE, alpha=0.85,
            error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    ax.set(xlabel="drop in average precision when shuffled",
           title="Permutation importance (holdout)")
    ax.tick_params(axis="y", labelsize=7.5)
    return _save(fig, "12_permutation_importance")


def plot_pdp(display) -> Path:
    fig = plt.figure(figsize=(9.4, 6.2))
    display.plot(kind="average", ax=fig.gca(), line_kw={"color": ACCENT, "lw": 1.6})
    fig.suptitle("Partial dependence of the tuned model", fontsize=10, y=0.98)
    return _save(fig, "13_partial_dependence")


def plot_pdp_curves(pdp_rows, X: pd.DataFrame) -> Path:
    """2x2 partial-dependence panels with a rug of the observed distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6))
    for ax, row in zip(axes.ravel(), pdp_rows):
        grid = np.asarray(row["grid"])
        avg = np.asarray(row["average"])
        ax.plot(grid, avg, color=ACCENT, lw=1.8)
        ax.fill_between(grid, avg, alpha=0.12, color=ACCENT)
        ax.set_title(row["feature"])
        ax.set_xlabel(row["feature"])
        ax.set_ylabel("P(subscribe)")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        values = pd.to_numeric(X[row["feature"]], errors="coerce").dropna()
        if len(values):
            qs = np.quantile(values, np.linspace(0.05, 0.95, 18))
            ymin = ax.get_ylim()[0]
            ax.plot(qs, np.full_like(qs, ymin), "|", color=MUTED, ms=5,
                    transform=ax.get_xaxis_transform())
    return _save(fig, "13_partial_dependence")


def plot_logistic_coefficients(coefs: pd.DataFrame) -> Path:
    df = pd.concat([coefs.head(10), coefs.tail(10)]).sort_values("coef")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = [GREEN if v > 0 else ACCENT for v in df["coef"]]
    ax.barh(df["feature"], df["coef"], color=colors, alpha=0.85)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="log-odds coefficient", title="Logistic regression: strongest effects")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, "14_logistic_coefficients")


def plot_leakage(leak: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(leak))
    ax.barh(y - 0.2, leak["ap_without_duration"], height=0.38, color=BLUE,
            label="without duration (deployed)")
    ax.barh(y + 0.2, leak["ap_with_duration"], height=0.38, color=ACCENT,
            label="with duration (leaky)")
    for i, (a, b) in enumerate(zip(leak["ap_without_duration"], leak["ap_with_duration"])):
        ax.text(a + 0.01, i - 0.2, f"{a:.3f}", va="center", fontsize=7, color=BLUE)
        ax.text(b + 0.01, i + 0.2, f"{b:.3f}", va="center", fontsize=7, color=ACCENT)
    ax.set_yticks(y, leak["model"])
    ax.set_xlabel("average precision on holdout")
    ax.set_xlim(0, 1.0)
    ax.set_title("Why duration must be dropped: an artificial AP jump")
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, "15_leakage")
