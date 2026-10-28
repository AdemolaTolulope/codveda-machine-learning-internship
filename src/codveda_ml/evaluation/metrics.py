"""Rigorous classification metrics and leak-free out-of-fold probability thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_multiclass_classifier(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: pd.Series | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute comprehensive multiclass metrics with verified class ordering."""
    y_arr = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(model.predict(X), dtype=int)
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_arr, y_pred, labels=labels)
    report_dict = classification_report(
        y_arr,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    result: dict[str, Any] = {
        "n_samples": int(len(y_arr)),
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "precision_macro": float(
            precision_score(y_arr, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_arr, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_arr, y_pred, average="weighted", zero_division=0)
        ),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cname: {
                "precision": float(report_dict[cname]["precision"]),
                "recall": float(report_dict[cname]["recall"]),
                "f1": float(report_dict[cname]["f1-score"]),
                "support": int(report_dict[cname]["support"]),
            }
            for cname in class_names
        },
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_arr, proba, multi_class="ovr", average="macro")
        )
        result["log_loss"] = float(log_loss(y_arr, proba, labels=labels))

    return result


def evaluate_binary_classifier(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: pd.Series | np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Evaluate a binary classifier at a specified probability threshold.
    Confusion matrix strictly uses `labels=[0, 1]` -> `[[TN, FP], [FN, TP]]`.
    """
    y_arr = np.asarray(y_true, dtype=int)

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        y_pred = (y_score >= threshold).astype(int)
        has_proba = True
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X), dtype=float)
        y_pred = np.asarray(model.predict(X), dtype=int)
        has_proba = False
    else:
        y_pred = np.asarray(model.predict(X), dtype=int)
        y_score = y_pred.astype(float)
        has_proba = False

    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_arr, y_score, pos_label=1)
    prec_curve, rec_curve, _ = precision_recall_curve(y_arr, y_score, pos_label=1)

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_arr)),
        "positive_prevalence": float(y_arr.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "precision": float(precision_score(y_arr, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_arr, y_score)),
        "pr_auc": float(average_precision_score(y_arr, y_score, pos_label=1)),
        "brier_score": float(brier_score_loss(y_arr, y_score)) if has_proba else None,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "curves": {
            "fpr": [round(float(x), 6) for x in fpr],
            "tpr": [round(float(x), 6) for x in tpr],
            "pr_precision": [round(float(x), 6) for x in prec_curve],
            "pr_recall": [round(float(x), 6) for x in rec_curve],
        },
    }
    return metrics


def select_optimal_threshold_oof(
    y_train: pd.Series | np.ndarray,
    oof_proba: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, Any]:
    """
    Determine the F1-optimal and cost-optimal classification thresholds strictly
    from out-of-fold training probability predictions (`oof_proba`), preventing
    any inspection or leakage of the held-out test partition.
    """
    y_arr = np.asarray(y_train, dtype=int)
    grid = np.round(np.arange(0.10, 0.91, 0.02), 2)
    rows: list[dict[str, float]] = []

    for thr in grid:
        preds = (oof_proba >= thr).astype(int)
        cm = confusion_matrix(y_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(precision_score(y_arr, preds, zero_division=0))
        rec = float(recall_score(y_arr, preds, zero_division=0))
        f1 = float(f1_score(y_arr, preds, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        youden_j = rec + spec - 1.0
        expected_cost_per_customer = float((fn * fn_cost + fp * fp_cost) / len(y_arr))
        rows.append(
            {
                "threshold": float(thr),
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "youden_j": float(youden_j),
                "expected_cost_per_customer": expected_cost_per_customer,
            }
        )

    sweep_df = pd.DataFrame(rows)
    best_f1_row = sweep_df.sort_values(
        ["f1", "recall"], ascending=[False, False]
    ).iloc[0]
    best_cost_row = sweep_df.sort_values(
        ["expected_cost_per_customer", "f1"], ascending=[True, False]
    ).iloc[0]

    return {
        "optimal_f1_threshold": float(best_f1_row["threshold"]),
        "oof_f1_at_optimal": float(best_f1_row["f1"]),
        "oof_precision_at_optimal": float(best_f1_row["precision"]),
        "oof_recall_at_optimal": float(best_f1_row["recall"]),
        "optimal_cost_threshold": float(best_cost_row["threshold"]),
        "cost_ratio_fn_to_fp": float(fn_cost / fp_cost),
        "sweep_table": sweep_df,
    }
