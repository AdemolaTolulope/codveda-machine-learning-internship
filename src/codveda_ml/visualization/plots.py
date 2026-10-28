"""Matplotlib/Seaborn plotting suite producing clean, labelled figures from verified results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "amber": "#D95F02",
    "purple": "#6A51A3",
    "crimson": "#B22222",
    "slate": "#5C6B73",
    "green": "#2E7D32",
    "light_bg": "#F8FAFC",
}


def apply_plot_style() -> None:
    """Apply consistent high-readability matplotlib styling across all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.92,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
        }
    )


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_dataset_quality_audit(
    inventory_df: pd.DataFrame,
    stock_missing_by_col: dict[str, int],
    churn_train_df: pd.DataFrame,
    churn_test_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 1: Dataset audit — missingness, class balance, and churn correlation heatmap."""
    apply_plot_style()
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    # Panel A: Missing cells across datasets + Stock OHLC breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    cols = list(stock_missing_by_col.keys())
    vals = [stock_missing_by_col[c] for c in cols]
    bars = ax1.bar(cols, vals, color=[PALETTE["crimson"] if v > 0 else PALETTE["slate"] for v in vals])
    for bar, v in zip(bars, vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.3,
            str(v),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_title("A. Genuine Missing Values in Stock Prices Dataset (N = 497,472)")
    ax1.set_ylabel("Missing Observation Count (rows)")
    ax1.set_xlabel("Raw Stock Column")
    ax1.set_ylim(0, max(vals) + 2.5)

    # Panel B: Class distribution in Churn 80/20 splits
    ax2 = fig.add_subplot(gs[0, 1])
    train_counts = churn_train_df["Churn"].value_counts(normalize=True).reindex([False, True]) * 100
    test_counts = churn_test_df["Churn"].value_counts(normalize=True).reindex([False, True]) * 100
    x = np.arange(2)
    w = 0.34
    b1 = ax2.bar(x - w / 2, train_counts.values, w, label="Train: churn-bigml-80 (n=2,666)", color=PALETTE["navy"])
    b2 = ax2.bar(x + w / 2, test_counts.values, w, label="Test: churn-bigml-20 (n=667)", color=PALETTE["teal"])
    for b in list(b1) + list(b2):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{b.get_height():.2f}%",
            ha="center",
            fontsize=9,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Retained (Churn = False)", "Churned (Churn = True)"])
    ax2.set_ylabel("Percentage of Partition (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("B. Stratified Target Balance Across Official Churn Partitions")
    ax2.legend(loc="upper right")

    # Panel C: Exact collinearity heatmap on Churn minutes vs charge pairs
    ax3 = fig.add_subplot(gs[1, 0])
    pair_cols = [
        "Total day minutes",
        "Total day charge",
        "Total eve minutes",
        "Total eve charge",
        "Total night minutes",
        "Total night charge",
        "Total intl minutes",
        "Total intl charge",
        "Customer service calls",
    ]
    corr = churn_train_df[pair_cols].corr()
    short_labels = [
        "Day Min",
        "Day Chg",
        "Eve Min",
        "Eve Chg",
        "Night Min",
        "Night Chg",
        "Intl Min",
        "Intl Chg",
        "Cust Svc",
    ]
    sns.heatmap(
        corr,
        xticklabels=short_labels,
        yticklabels=short_labels,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.82, "label": "Pearson r"},
        ax=ax3,
        annot_kws={"size": 8},
    )
    ax3.set_title("C. Churn Usage vs Billing Charge Deterministic Collinearity (r = 1.00)")

    # Panel D: Dataset inventory row counts (log10 scale)
    ax4 = fig.add_subplot(gs[1, 1])
    short_names = [
        "Iris\n(150)",
        "House Prices\n(506)",
        "Churn Test\n(667)",
        "Sentiment\n(732)",
        "Churn Train\n(2,666)",
        "Stock Prices\n(497,472)",
    ]
    row_vals = sorted(inventory_df["n_rows"].tolist())
    bars4 = ax4.barh(short_names, row_vals, color=PALETTE["purple"], alpha=0.88)
    ax4.set_xscale("log")
    ax4.set_xlabel("Observation Count (log10 scale)")
    ax4.set_title("D. Supplied Codveda Archive Dataset Scale Comparison")
    for b, rv in zip(bars4, row_vals):
        ax4.text(rv * 1.15, b.get_y() + b.get_height() / 2, f"{rv:,}", va="center", fontsize=9)
    ax4.set_xlim(80, 2_000_000)

    fig.suptitle(
        "Codveda ML Internship — Raw Dataset Audit, Missingness & Structural Collinearity",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    return _save(fig, out_path)


def plot_scaling_comparison(
    raw_train_series: pd.Series,
    std_train_vals: np.ndarray,
    rob_train_vals: np.ndarray,
    churn_raw_df: pd.DataFrame,
    churn_scaled_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 2: Effect of training-only scaling on skewed Stock volume and Churn features."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.26)

    # Stock close raw vs scaled (clipped 99th pct for visual clarity with annotation)
    p99_raw = float(np.nanpercentile(raw_train_series, 99))
    axes[0, 0].hist(
        raw_train_series.dropna().clip(upper=p99_raw),
        bins=45,
        color=PALETTE["navy"],
        alpha=0.85,
        edgecolor="white",
    )
    axes[0, 0].set_title("A. Stock Close Price — Raw Train Distribution (2014–2016)")
    axes[0, 0].set_xlabel("Raw Price in USD (display clipped at 99th pct = $" + f"{p99_raw:.0f})")
    axes[0, 0].set_ylabel("Trading-Day Frequency")

    axes[0, 1].hist(
        np.clip(std_train_vals, -2.5, 4.5),
        bins=45,
        alpha=0.65,
        label="StandardScaler (mean=0, SD=1)",
        color=PALETTE["teal"],
    )
    axes[0, 1].hist(
        np.clip(rob_train_vals, -2.5, 4.5),
        bins=45,
        alpha=0.55,
        label="RobustScaler (median=0, IQR=1)",
        color=PALETTE["amber"],
    )
    axes[0, 1].set_title("B. Stock Close Price — Post-Scaling Comparison (Train Fit Only)")
    axes[0, 1].set_xlabel("Transformed Feature Value (display clipped [-2.5, 4.5])")
    axes[0, 1].set_ylabel("Trading-Day Frequency")
    axes[0, 1].legend()

    # Churn raw boxplots showing wildly heterogeneous feature scales
    key_churn_cols = [
        "Account length",
        "Total day minutes",
        "Total intl minutes",
        "Customer service calls",
    ]
    sns.boxplot(
        data=churn_raw_df[key_churn_cols],
        orient="h",
        ax=axes[1, 0],
        palette=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"], PALETTE["purple"]],
    )
    axes[1, 0].set_title("C. Churn Numeric Features Before Scaling (Heterogeneous Units)")
    axes[1, 0].set_xlabel("Raw Feature Magnitude (days / minutes / calls)")

    sns.boxplot(
        data=churn_scaled_df[key_churn_cols],
        orient="h",
        ax=axes[1, 1],
        palette=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"], PALETTE["purple"]],
    )
    axes[1, 1].axvline(0.0, color=PALETTE["crimson"], linestyle="--", linewidth=1.1, label="Zero Mean")
    axes[1, 1].set_title("D. Churn Numeric Features After StandardScaler (Fit on Train 80%)")
    axes[1, 1].set_xlabel("Standardized Z-Score (SD units)")
    axes[1, 1].legend(loc="lower right")

    fig.suptitle(
        "Task 1 — Leakage-Free Feature Standardization & Scaling Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    return _save(fig, out_path)


def plot_knn_k_sweep(
    cv_df: pd.DataFrame,
    selected_k: int,
    selected_weights: str,
    out_path: Path,
) -> Path:
    """Figure 3: KNN CV Macro-F1 across K values, scaled vs unscaled and weight schemes."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)

    for ax, weight_mode, panel_lbl in [
        (axes[0], "uniform", "A. Uniform Neighbor Weighting"),
        (axes[1], "distance", "B. Inverse-Distance Neighbor Weighting"),
    ]:
        sub = cv_df[cv_df["weights"] == weight_mode]
        for scaled_flag, label, color, marker in [
            (True, "StandardScaler Inside Pipeline", PALETTE["navy"], "o"),
            (False, "Unscaled Raw Features (cm)", PALETTE["amber"], "s"),
        ]:
            s = sub[sub["scaled"] == scaled_flag].sort_values("k")
            ax.plot(
                s["k"],
                s["cv_f1_macro_mean"],
                marker=marker,
                linewidth=2.1,
                color=color,
                label=label,
            )
            ax.fill_between(
                s["k"],
                s["cv_f1_macro_mean"] - s["cv_f1_macro_std"],
                np.minimum(1.0, s["cv_f1_macro_mean"] + s["cv_f1_macro_std"]),
                color=color,
                alpha=0.14,
            )
        if weight_mode == selected_weights:
            ax.axvline(
                selected_k,
                color=PALETTE["crimson"],
                linestyle="--",
                linewidth=1.5,
                label=f"Selected Final K = {selected_k}",
            )
        ax.set_title(panel_lbl)
        ax.set_xlabel("Number of Neighbors (K)")
        ax.set_xticks(sorted(cv_df["k"].unique()))
        ax.set_ylim(0.84, 1.01)
        ax.legend(loc="lower left")

    axes[0].set_ylabel("5-Fold Stratified CV Macro-F1 (Train n = 120)")
    fig.suptitle(
        "Task 2 — KNN Sensitivity to K, Feature Scaling, and Weighting on Iris Training Data",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)


def plot_multiclass_confusion_and_scatter(
    cm: list[list[int]],
    class_names: list[str],
    iris_df: pd.DataFrame,
    title_prefix: str,
    subtitle_metrics: str,
    out_path: Path,
) -> Path:
    """Figure 4 / 10: Confusion Matrix + Iris Petal Feature Space."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
        ax=axes[0],
    )
    axes[0].set_title(f"A. Held-Out Test Confusion Matrix (n = 30)\n{subtitle_metrics}")
    axes[0].set_xlabel("Predicted Iris Species")
    axes[0].set_ylabel("Actual Iris Species")

    colors = {"setosa": PALETTE["navy"], "versicolor": PALETTE["teal"], "virginica": PALETTE["amber"]}
    for sp, grp in iris_df.groupby("species"):
        axes[1].scatter(
            grp["petal_length"],
            grp["petal_width"],
            label=sp.capitalize(),
            color=colors[sp],
            s=48,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
    axes[1].set_title("B. Iris Petal Feature Space & Versicolor–Virginica Boundary Zone")
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].set_ylabel("Petal Width (cm)")
    axes[1].legend(title="Species")

    fig.suptitle(title_prefix, fontsize=13.5, fontweight="bold", y=1.02)
    return _save(fig, out_path)


def plot_logreg_roc_pr_threshold(
    eval_default: dict[str, Any],
    eval_opt: dict[str, Any],
    oof_threshold_info: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 5: Logistic Regression ROC, Precision-Recall, and OOF Threshold Sweep."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panel A: ROC curve
    fpr = eval_default["curves"]["fpr"]
    tpr = eval_default["curves"]["tpr"]
    auc_val = eval_default["roc_auc"]
    axes[0].plot(fpr, tpr, color=PALETTE["navy"], lw=2.3, label=f"Logistic Regression (AUC = {auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["slate"], lw=1.2, label="Random Classifier (AUC = 0.5000)")
    axes[0].set_title("A. Held-Out ROC Curve (churn-bigml-20)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    # Panel B: Precision-Recall curve
    rec = eval_default["curves"]["pr_recall"]
    prec = eval_default["curves"]["pr_precision"]
    ap = eval_default["pr_auc"]
    prev = eval_default["positive_prevalence"]
    axes[1].plot(rec, prec, color=PALETTE["teal"], lw=2.3, label=f"Logistic Regression (PR-AUC = {ap:.4f})")
    axes[1].axhline(prev, linestyle="--", color=PALETTE["slate"], lw=1.2, label=f"No-Skill Baseline ({prev:.3f})")
    axes[1].scatter(
        [eval_default["recall"]],
        [eval_default["precision"]],
        color=PALETTE["amber"],
        s=70,
        zorder=5,
        label=f"Default Thr = 0.50 (F1={eval_default['f1']:.3f})",
    )
    axes[1].scatter(
        [eval_opt["recall"]],
        [eval_opt["precision"]],
        color=PALETTE["crimson"],
        marker="D",
        s=70,
        zorder=5,
        label=f"OOF-Tuned Thr = {eval_opt['threshold']:.2f} (F1={eval_opt['f1']:.3f})",
    )
    axes[1].set_title("B. Held-Out Precision–Recall Curve")
    axes[1].set_xlabel("Recall (Churn Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right", fontsize=8.8)

    # Panel C: OOF Threshold Sweep on Training Data
    sweep = oof_threshold_info["sweep_table"]
    opt_thr = oof_threshold_info["optimal_f1_threshold"]
    axes[2].plot(sweep["threshold"], sweep["precision"], label="OOF Precision", color=PALETTE["navy"], lw=1.8)
    axes[2].plot(sweep["threshold"], sweep["recall"], label="OOF Recall", color=PALETTE["teal"], lw=1.8)
    axes[2].plot(sweep["threshold"], sweep["f1"], label="OOF F1-Score", color=PALETTE["amber"], lw=2.4)
    axes[2].axvline(opt_thr, color=PALETTE["crimson"], linestyle="--", lw=1.6, label=f"Selected Thr = {opt_thr:.2f}")
    axes[2].set_title("C. Out-of-Fold Training Threshold Calibration")
    axes[2].set_xlabel("Classification Probability Threshold")
    axes[2].set_ylabel("5-Fold OOF Metric Value (Train n = 2,666)")
    axes[2].legend(loc="best", fontsize=8.8)

    fig.suptitle(
        "Task 3 — Logistic Regression Discrimination & Leak-Free Threshold Selection on Customer Churn",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)


def plot_logreg_odds_ratios_and_vif(
    odds_df: pd.DataFrame,
    vif_before: pd.DataFrame,
    vif_after: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 6: Odds Ratios with 95% Bootstrap CIs + VIF Multicollinearity Remediation."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.subplots_adjust(wspace=0.38)

    # Top 12 non-State or strongest predictors by |coef|
    top = odds_df.reindex(odds_df["coef"].abs().sort_values(ascending=False).index).head(12)
    top = top.sort_values("odds_ratio", ascending=True)
    y_pos = np.arange(len(top))
    or_vals = top["odds_ratio"].to_numpy()
    err_low = or_vals - top["or_ci_low"].to_numpy()
    err_high = top["or_ci_high"].to_numpy() - or_vals
    colors = [PALETTE["crimson"] if v > 1.0 else PALETTE["teal"] for v in or_vals]

    axes[0].errorbar(
        or_vals,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color=PALETTE["navy"],
        ecolor=PALETTE["slate"],
        elinewidth=1.6,
        capsize=3.5,
        markersize=6.5,
    )
    for yp, val, c in zip(y_pos, or_vals, colors):
        axes[0].scatter([val], [yp], color=c, s=55, zorder=4)
    axes[0].axvline(1.0, color="#334155", linestyle="--", lw=1.3, label="Neutral Odds (OR = 1.0)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top["clean_feature"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Odds Ratio exp(beta) [Log Scale, 95% Bootstrap CI]")
    axes[0].set_title("A. Top 12 Logistic Regression Odds Ratios (Train n = 2,666)")
    axes[0].legend(loc="lower right")

    # Panel B: VIF before vs after dropping redundant charge columns
    merged_vif = pd.merge(
        vif_before[["feature", "vif"]].rename(columns={"vif": "vif_all_15"}),
        vif_after[["feature", "vif"]].rename(columns={"vif": "vif_pruned_11"}),
        on="feature",
        how="left",
    ).sort_values("vif_all_15", ascending=True)

    y_v = np.arange(len(merged_vif))
    axes[1].barh(
        y_v - 0.18,
        merged_vif["vif_all_15"],
        height=0.35,
        color=PALETTE["amber"],
        label="Raw 15 Numeric Features (Minutes + Charges)",
    )
    axes[1].barh(
        y_v + 0.18,
        merged_vif["vif_pruned_11"].fillna(0.0),
        height=0.35,
        color=PALETTE["navy"],
        label="Pruned 11 Numeric Features (Charges Dropped)",
    )
    axes[1].axvline(5.0, color=PALETTE["crimson"], linestyle="--", lw=1.3, label="Conventional VIF = 5 Threshold")
    axes[1].set_xscale("log")
    axes[1].set_yticks(y_v)
    axes[1].set_yticklabels(merged_vif["feature"], fontsize=8.8)
    axes[1].set_xlabel("Variance Inflation Factor (VIF, log10 scale)")
    axes[1].set_title("B. Multicollinearity Elimination (Max VIF: >10^6 -> 1.01)")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Task 3 — Statistical Interpretability: Odds Ratios & Collinearity Diagnostics",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    return _save(fig, out_path)


def plot_binary_confusion_pair(
    eval_a: dict[str, Any],
    eval_b: dict[str, Any],
    label_a: str,
    label_b: str,
    suptitle: str,
    out_path: Path,
) -> Path:
    """Figure 7: Side-by-side binary confusion matrices."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    tick_labels = ["Retained (0)", "Churned (1)"]

    for ax, ev, title in [(axes[0], eval_a, label_a), (axes[1], eval_b, label_b)]:
        cm = np.array(ev["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(
            f"{title}\nAcc={ev['accuracy']:.3f} | Prec={ev['precision']:.3f} | Rec={ev['recall']:.3f} | F1={ev['f1']:.3f}"
        )
        ax.set_xlabel("Predicted Customer Outcome")
        ax.set_ylabel("Actual Customer Outcome")

    fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", y=1.03)
    return _save(fig, out_path)


def plot_single_binary_confusion(
    eval_res: dict[str, Any],
    title: str,
    out_path: Path,
) -> Path:
    """Figure 13: Single clean binary confusion matrix with counts and row percentages."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    cm = np.array(eval_res["confusion_matrix"])
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = cm / np.maximum(row_sums, 1) * 100.0
    annot = np.array(
        [[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
    )
    tick_labels = ["Retained (False)", "Churned (True)"]
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(
        f"{title}\nAcc={eval_res['accuracy']:.3f} | Prec={eval_res['precision']:.3f} | "
        f"Rec={eval_res['recall']:.3f} | F1={eval_res['f1']:.3f} | AUC={eval_res['roc_auc']:.3f}"
    )
    ax.set_xlabel("Predicted Customer Class")
    ax.set_ylabel("Actual Customer Class")
    return _save(fig, out_path)


def plot_decision_trees_comparison(
    unrestricted_clf: Any,
    pruned_clf: Any,
    feature_names: list[str],
    class_names: list[str],
    out_path: Path,
) -> Path:
    """Figure 8: Side-by-side Unrestricted vs Cost-Complexity Pruned Decision Trees."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8))

    plot_tree(
        unrestricted_clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=axes[0],
    )
    axes[0].set_title(
        f"A. Unrestricted Tree (Depth = {unrestricted_clf.get_depth()}, "
        f"Leaves = {unrestricted_clf.get_n_leaves()}, ccp_alpha = 0.0)"
    )

    plot_tree(
        pruned_clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=axes[1],
    )
    axes[1].set_title(
        f"B. Pruned Tree (Depth = {pruned_clf.get_depth()}, "
        f"Leaves = {pruned_clf.get_n_leaves()}, Regularised via CV)"
    )

    fig.suptitle(
        "Task 4 — Decision Tree Structural Complexity: Unrestricted Overfitting vs Pruned Generalisation",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    return _save(fig, out_path)


def plot_ccp_pruning_curve(
    ccp_df: pd.DataFrame,
    selected_alpha: float,
    out_path: Path,
) -> Path:
    """Figure 9: Cost-Complexity Pruning Path on Iris Training Set."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].plot(
        ccp_df["ccp_alpha"],
        ccp_df["n_leaves"],
        marker="o",
        drawstyle="steps-post",
        color=PALETTE["navy"],
        lw=2.0,
        label="Number of Terminal Leaves",
    )
    axes[0].plot(
        ccp_df["ccp_alpha"],
        ccp_df["depth"],
        marker="s",
        drawstyle="steps-post",
        color=PALETTE["teal"],
        lw=2.0,
        label="Tree Depth",
    )
    axes[0].axvline(selected_alpha, color=PALETTE["crimson"], linestyle="--", lw=1.5, label=f"Selected alpha = {selected_alpha:.4f}")
    axes[0].set_title("A. Structural Simplification Along Effective Alpha Path")
    axes[0].set_xlabel("Cost-Complexity Parameter (ccp_alpha)")
    axes[0].set_ylabel("Count (Leaves / Depth)")
    axes[0].legend()

    axes[1].plot(
        ccp_df["ccp_alpha"],
        ccp_df["train_f1_macro"],
        marker="o",
        drawstyle="steps-post",
        color=PALETTE["amber"],
        lw=2.0,
        label="In-Sample Training Macro-F1",
    )
    axes[1].plot(
        ccp_df["ccp_alpha"],
        ccp_df["cv_f1_macro_mean"],
        marker="D",
        drawstyle="steps-post",
        color=PALETTE["navy"],
        lw=2.2,
        label="5-Fold Stratified CV Macro-F1",
    )
    axes[1].fill_between(
        ccp_df["ccp_alpha"],
        ccp_df["cv_f1_macro_mean"] - ccp_df["cv_f1_macro_std"],
        np.minimum(1.0, ccp_df["cv_f1_macro_mean"] + ccp_df["cv_f1_macro_std"]),
        color=PALETTE["navy"],
        alpha=0.14,
        step="post",
    )
    axes[1].axvline(selected_alpha, color=PALETTE["crimson"], linestyle="--", lw=1.5, label=f"Selected alpha = {selected_alpha:.4f}")
    axes[1].set_title("B. Overfitting vs Generalisation Trade-Off (Train n = 120)")
    axes[1].set_xlabel("Cost-Complexity Parameter (ccp_alpha)")
    axes[1].set_ylabel("Macro-F1 Score")
    axes[1].legend(loc="lower left")

    fig.suptitle(
        "Task 4 — Minimal Cost-Complexity Pruning Analysis on Iris Training Split",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)


def plot_rf_feature_importances(
    mdi_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 11: Impurity-based (MDI) vs Held-Out Permutation Feature Importance."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.subplots_adjust(wspace=0.38)

    top_mdi = mdi_df.head(12).iloc[::-1]
    axes[0].barh(
        top_mdi["feature"],
        top_mdi["mdi_importance"],
        xerr=top_mdi["mdi_std"],
        color=PALETTE["navy"],
        alpha=0.88,
        capsize=3,
    )
    axes[0].set_title("A. Impurity-Based Feature Importance (Mean Decrease in Gini)")
    axes[0].set_xlabel("Mean Decrease in Impurity Across Forest Trees")

    top_perm = perm_df.head(12).iloc[::-1]
    axes[1].barh(
        top_perm["feature"],
        top_perm["perm_importance_mean"],
        xerr=top_perm["perm_importance_std"],
        color=PALETTE["teal"],
        alpha=0.88,
        capsize=3,
    )
    axes[1].axvline(0.0, color=PALETTE["slate"], linestyle="--", lw=1.0)
    axes[1].set_title("B. Permutation Importance (30 Shuffles, ROC-AUC Drop)")
    axes[1].set_xlabel("Mean Decrease in ROC-AUC Upon Feature Permutation")

    fig.suptitle(
        "Task 5 — Random Forest Feature Importance: In-Sample Gini (MDI) vs Permutation Drop",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    return _save(fig, out_path)


def plot_rf_vs_logreg_and_stability(
    rf_eval: dict[str, Any],
    logreg_eval: dict[str, Any],
    stability_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 12: Random Forest vs Logistic Regression ROC/PR + 10-Seed Stability."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panel A: ROC comparison
    axes[0].plot(
        rf_eval["curves"]["fpr"],
        rf_eval["curves"]["tpr"],
        color=PALETTE["navy"],
        lw=2.4,
        label=f"Random Forest (AUC = {rf_eval['roc_auc']:.4f})",
    )
    axes[0].plot(
        logreg_eval["curves"]["fpr"],
        logreg_eval["curves"]["tpr"],
        color=PALETTE["amber"],
        lw=2.0,
        linestyle="-.",
        label=f"Logistic Regression (AUC = {logreg_eval['roc_auc']:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["slate"], lw=1.1)
    axes[0].set_title("A. Held-Out ROC Curve Comparison")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    # Panel B: PR comparison
    axes[1].plot(
        rf_eval["curves"]["pr_recall"],
        rf_eval["curves"]["pr_precision"],
        color=PALETTE["navy"],
        lw=2.4,
        label=f"Random Forest (PR-AUC = {rf_eval['pr_auc']:.4f})",
    )
    axes[1].plot(
        logreg_eval["curves"]["pr_recall"],
        logreg_eval["curves"]["pr_precision"],
        color=PALETTE["amber"],
        lw=2.0,
        linestyle="-.",
        label=f"Logistic Regression (PR-AUC = {logreg_eval['pr_auc']:.4f})",
    )
    axes[1].axhline(rf_eval["positive_prevalence"], linestyle="--", color=PALETTE["slate"], lw=1.1, label="Baseline Prevalence")
    axes[1].set_title("B. Held-Out Precision–Recall Comparison")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    # Panel C: Stability across 10 random seeds
    melted = stability_df.melt(
        id_vars=["seed"],
        value_vars=["roc_auc", "pr_auc", "f1"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1-Score"}
    melted["Metric"] = melted["metric"].map(metric_labels)
    sns.boxplot(
        data=melted,
        x="Metric",
        y="score",
        width=0.4,
        color="#DCE7F5",
        ax=axes[2],
    )
    sns.stripplot(
        data=melted,
        x="Metric",
        y="score",
        color=PALETTE["navy"],
        size=7,
        jitter=0.12,
        ax=axes[2],
    )
    axes[2].set_title("C. Random Forest Stability Across 10 Seeds")
    axes[2].set_xlabel("Held-Out Test Metric")
    axes[2].set_ylabel("Metric Value")
    axes[2].set_ylim(0.75, 0.98)

    fig.suptitle(
        "Task 5 — Random Forest Predictive Lift Over Logistic Regression & Multi-Seed Stability",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)


def plot_svm_decision_boundaries(
    boundary_data: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 14: 2D SVM Decision Boundaries (Linear vs RBF Feature Pair + PCA Slice)."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    xx = boundary_data["xx"]
    yy = boundary_data["yy"]
    X_pts = boundary_data["X_test_2d"]
    y_pts = boundary_data["y_test"]

    panels = [
        (axes[0], boundary_data["Z_linear"], "A. Linear SVM (2D Feature Space)\nCannot Isolate Curved Churn Pockets"),
        (axes[1], boundary_data["Z_rbf"], "B. RBF SVM (2D Feature Space)\nCaptures High-Day-Min OR High-Svc-Calls"),
    ]
    for ax, Z, title in panels:
        cs = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.58)
        ax.contour(xx, yy, Z, levels=[0.5], colors=[PALETTE["crimson"]], linewidths=2.0)
        retained = y_pts == 0
        churned = y_pts == 1
        ax.scatter(
            X_pts[retained, 0],
            X_pts[retained, 1],
            c=PALETTE["navy"],
            s=22,
            alpha=0.55,
            label="Retained (Test)",
        )
        ax.scatter(
            X_pts[churned, 0],
            X_pts[churned, 1],
            c=PALETTE["amber"],
            marker="^",
            s=42,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label="Churned (Test)",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Standardized Total Day Minutes (Z-score)")
        ax.set_ylabel("Standardized Customer Service Calls (Z-score)")
        ax.legend(loc="upper left", fontsize=8.5)

    # Panel C: Full high-dimensional RBF SVM conditional slice on PCA1 vs PCA2
    xx_p = boundary_data["xx_pca"]
    yy_p = boundary_data["yy_pca"]
    Z_p = boundary_data["Z_pca_slice"]
    Xp = boundary_data["X_test_pca2"]
    cs3 = axes[2].contourf(xx_p, yy_p, Z_p, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.58)
    axes[2].contour(xx_p, yy_p, Z_p, levels=[0.5], colors=[PALETTE["crimson"]], linewidths=2.0)
    axes[2].scatter(Xp[y_pts == 0, 0], Xp[y_pts == 0, 1], c=PALETTE["navy"], s=22, alpha=0.50, label="Retained")
    axes[2].scatter(
        Xp[y_pts == 1, 0],
        Xp[y_pts == 1, 1],
        c=PALETTE["amber"],
        marker="^",
        s=42,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.90,
        label="Churned",
    )
    axes[2].set_title(
        f"C. 2D PCA Projection RBF SVM ({(boundary_data['pc1_var']+boundary_data['pc2_var'])*100:.1f}% Var)\n"
        f"Discarding {boundary_data['n_full_features']-2} Dimensions Obscures Class Separation",
        fontsize=10.5,
    )
    axes[2].set_xlabel(f"Principal Component 1 ({boundary_data['pc1_var']*100:.1f}% Var)")
    axes[2].set_ylabel(f"Principal Component 2 ({boundary_data['pc2_var']*100:.1f}% Var)")
    cbar = fig.colorbar(cs3, ax=axes[2], shrink=0.86)
    cbar.set_label("Predicted P(Churn = True)")

    fig.suptitle(
        "Task 6 — SVM Decision Boundary Visualisation & High-Dimensional Projection Geometry",
        fontsize=13.5,
        fontweight="bold",
        y=1.03,
    )
    return _save(fig, out_path)


def plot_svm_kernel_comparison(
    kernel_summary_df: pd.DataFrame,
    rbf_eval: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 15: Linear vs RBF SVM Metric Comparison + Selected RBF Confusion Matrix."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    metrics_cols = ["cv_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1", "test_recall", "test_precision"]
    pretty = ["CV AUC", "Test AUC", "Test PR-AUC", "Test F1", "Test Recall", "Test Precision"]
    x = np.arange(len(metrics_cols))
    w = 0.35

    lin_row = kernel_summary_df[kernel_summary_df["kernel"] == "linear"].iloc[0]
    rbf_row = kernel_summary_df[kernel_summary_df["kernel"] == "rbf"].iloc[0]

    b1 = axes[0].bar(x - w / 2, [lin_row[c] for c in metrics_cols], w, label="Linear Kernel SVM", color=PALETTE["amber"])
    b2 = axes[0].bar(x + w / 2, [rbf_row[c] for c in metrics_cols], w, label="RBF Kernel SVM", color=PALETTE["navy"])
    for b in list(b1) + list(b2):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.012,
            f"{b.get_height():.2f}",
            ha="center",
            fontsize=8.2,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pretty, rotation=15)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A. Linear vs RBF Kernel Performance (Churn Dataset)")
    axes[0].legend(loc="lower right")

    cm = np.array(rbf_eval["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
        ax=axes[1],
    )
    axes[1].set_title(
        f"B. Selected RBF SVM Held-Out Confusion Matrix (Thr = {rbf_eval['threshold']:.2f})\n"
        f"Acc={rbf_eval['accuracy']:.3f} | F1={rbf_eval['f1']:.3f} | AUC={rbf_eval['roc_auc']:.3f}"
    )
    axes[1].set_xlabel("Predicted Customer Class")
    axes[1].set_ylabel("Actual Customer Class")

    fig.suptitle(
        "Task 6 — Support Vector Machine Kernel Comparison & Held-Out Test Performance",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)


def plot_portfolio_benchmark(
    churn_curves: dict[str, dict[str, Any]],
    comparison_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 16: Cross-task executive model comparison across Churn and Iris benchmarks."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))
    fig.subplots_adjust(wspace=0.42)

    style_map = {
        "Logistic Regression": (PALETTE["amber"], "-.", 2.0),
        "SVM (Linear)": (PALETTE["purple"], ":", 2.0),
        "SVM (RBF)": (PALETTE["teal"], "--", 2.2),
        "Random Forest": (PALETTE["navy"], "-", 2.5),
    }

    for name, ev in churn_curves.items():
        color, ls, lw = style_map.get(name, (PALETTE["slate"], "-", 1.8))
        axes[0].plot(
            ev["curves"]["fpr"],
            ev["curves"]["tpr"],
            color=color,
            linestyle=ls,
            lw=lw,
            label=f"{name} (AUC={ev['roc_auc']:.3f})",
        )
        axes[1].plot(
            ev["curves"]["pr_recall"],
            ev["curves"]["pr_precision"],
            color=color,
            linestyle=ls,
            lw=lw,
            label=f"{name} (AP={ev['pr_auc']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1.0)
    axes[0].set_title("A. Churn Benchmark — Held-Out ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8.5)

    axes[1].axhline(0.1424, linestyle="--", color="#94A3B8", lw=1.0, label="Prevalence (0.142)")
    axes[1].set_title("B. Churn Benchmark — Held-Out PR Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8.5)

    churn_sub = comparison_df[
        (comparison_df["dataset"] == "Customer Churn (churn-bigml-20)")
        & (~comparison_df["model_name"].str.contains("Dummy"))
    ].copy()
    short_map = {
        "Logistic Regression (Default Thr=0.50)": "LogReg (Thr=0.50)",
        "Logistic Regression (OOF-Calibrated)": "LogReg (OOF Thr)",
        "SVM (Linear Kernel, OOF-Calibrated)": "SVM Linear (OOF)",
        "SVM (RBF Kernel, OOF-Calibrated)": "SVM RBF (OOF)",
        "Random Forest (Untuned Baseline)": "RF (Untuned)",
        "Random Forest (Tuned, OOF-Calibrated)": "RF (Tuned, OOF)",
    }
    churn_sub["short_label"] = churn_sub["model_name"].map(lambda s: short_map.get(s, s))
    churn_sub = churn_sub.sort_values("f1_primary", ascending=True)
    y_pos = np.arange(len(churn_sub))
    axes[2].barh(y_pos - 0.17, churn_sub["roc_auc"], height=0.33, color=PALETTE["navy"], label="Test ROC-AUC")
    axes[2].barh(y_pos + 0.17, churn_sub["f1_primary"], height=0.33, color=PALETTE["teal"], label="Test F1-Score")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(churn_sub["short_label"], fontsize=9.2)
    axes[2].set_xlim(0, 1.05)
    axes[2].set_xlabel("Held-Out Test Metric (n = 667)")
    axes[2].set_title("C. Churn Model Family Progression")
    axes[2].legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Codveda ML Internship — Unified Benchmark Comparison Across Levels 1–3",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, out_path)
