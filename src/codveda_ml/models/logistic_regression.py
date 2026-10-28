"""Task 3 (Level 2 Intermediate): Logistic Regression for Binary Customer Churn Classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from codveda_ml.config import ensure_directories, get_runtime_metadata, load_config, save_json
from codveda_ml.data.loaders import (
    CHURN_NUMERIC_COLS_ALL,
    CHURN_NUMERIC_COLS_PRUNED,
    load_churn_splits,
)
from codveda_ml.evaluation.metrics import (
    evaluate_binary_classifier,
    select_optimal_threshold_oof,
)
from codveda_ml.preprocessing.pipelines import build_churn_preprocessor, compute_vif_table
from codveda_ml.visualization.plots import (
    plot_binary_confusion_pair,
    plot_logreg_odds_ratios_and_vif,
    plot_logreg_roc_pr_threshold,
)


def _clean_feature_label(raw_name: str) -> str:
    return (
        raw_name.replace("International plan_Yes", "International Plan = Yes")
        .replace("Voice mail plan_Yes", "Voice Mail Plan = Yes")
        .replace("Area code_", "Area Code = ")
        .replace("State_", "State = ")
    )


def run_task3_logistic_regression(root: Path | None = None) -> dict[str, Any]:
    """Train, tune, interpret, and evaluate Logistic Regression on Customer Churn."""
    dirs = ensure_directories(root)
    cfg = load_config(dirs["root"])
    seed = int(cfg["project"]["random_seed"])
    c_grid = [float(c) for c in cfg["tasks"]["task3_logistic_regression"]["c_grid"]]
    n_boot = int(cfg["tasks"]["task3_logistic_regression"]["bootstrap_iterations"])

    churn = load_churn_splits(dirs["root"])
    X_train, X_test = churn["X_train"], churn["X_test"]
    y_train, y_test = churn["y_train"], churn["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Multicollinearity (VIF) diagnostic justifying removal of redundant *charge columns
    vif_before = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_ALL])
    vif_after = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_PRUNED])

    # 2. Majority-class Dummy Baseline (demonstrates the accuracy paradox under 85.8%/14.2% split)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_eval = evaluate_binary_classifier(dummy, X_test, y_test, threshold=0.50)

    # 3. Untuned Baseline Logistic Regression (C=1.0, class_weight=None, threshold=0.50)
    untuned_pipe = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]
    )
    untuned_pipe.fit(X_train, y_train)
    untuned_eval = evaluate_binary_classifier(untuned_pipe, X_test, y_test, threshold=0.50)

    # 4. Hyperparameter Grid Search via 5-fold Stratified CV on churn-bigml-80 ONLY
    base_pipe = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
        ]
    )
    param_grid = {
        "clf__C": c_grid,
        "clf__class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid.cv_results_)[
        [
            "param_clf__C",
            "param_clf__class_weight",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_pr_auc",
            "mean_test_f1",
        ]
    ].sort_values("mean_test_roc_auc", ascending=False)
    cv_results_df.to_csv(dirs["tables_dir"] / "task3_logreg_cv_results.csv", index=False)

    best_pipe: Pipeline = grid.best_estimator_

    # 5. Leak-free Out-of-Fold Probability Threshold Calibration on Training Split ONLY
    oof_proba = cross_val_predict(
        clone(best_pipe),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    oof_thr_info = select_optimal_threshold_oof(y_train, oof_proba, fn_cost=5.0, fp_cost=1.0)
    oof_thr_info["sweep_table"].to_csv(
        dirs["tables_dir"] / "task3_logreg_oof_threshold_sweep.csv", index=False
    )
    optimal_thr = float(oof_thr_info["optimal_f1_threshold"])

    # 6. Held-out test evaluation at default 0.50 AND OOF-selected optimal threshold
    test_eval_default = evaluate_binary_classifier(best_pipe, X_test, y_test, threshold=0.50)
    test_eval_optimal = evaluate_binary_classifier(best_pipe, X_test, y_test, threshold=optimal_thr)

    # 7. Coefficients, Odds Ratios, and 95% Stratified Bootstrap Confidence Intervals
    feat_names = list(best_pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out())
    point_coefs = best_pipe.named_steps["clf"].coef_.ravel()

    # Pre-transform training matrix once and bootstrap refit the identical regularized logit
    # (equivalent within bootstrap sample up to minor scaling noise, fast & exact for 1,000 draws)
    rng = np.random.default_rng(seed)
    X_tr_tx = best_pipe.named_steps["prep"].transform(X_train)
    y_tr_np = y_train.to_numpy()
    pos_idx = np.where(y_tr_np == 1)[0]
    neg_idx = np.where(y_tr_np == 0)[0]

    boot_coefs = np.zeros((n_boot, len(feat_names)), dtype=float)
    base_clf = clone(best_pipe.named_steps["clf"])
    for b in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        s_idx = np.concatenate([s_pos, s_neg])
        m = clone(base_clf)
        m.fit(X_tr_tx[s_idx], y_tr_np[s_idx])
        boot_coefs[b] = m.coef_.ravel()

    ci_low_coef = np.percentile(boot_coefs, 2.5, axis=0)
    ci_high_coef = np.percentile(boot_coefs, 97.5, axis=0)

    odds_df = pd.DataFrame(
        {
            "feature": feat_names,
            "clean_feature": [_clean_feature_label(f) for f in feat_names],
            "scale_basis": [
                "Per +1 Train SD" if f in CHURN_NUMERIC_COLS_PRUNED else "Binary Indicator (1 vs 0)"
                for f in feat_names
            ],
            "coef": point_coefs,
            "coef_ci_low": ci_low_coef,
            "coef_ci_high": ci_high_coef,
            "odds_ratio": np.exp(point_coefs),
            "or_ci_low": np.exp(ci_low_coef),
            "or_ci_high": np.exp(ci_high_coef),
        }
    ).sort_values("odds_ratio", ascending=False)
    odds_df.to_csv(dirs["tables_dir"] / "task3_logreg_odds_ratios_bootstrap.csv", index=False)

    # Persist pipeline & test predictions
    model_path = dirs["models_dir"] / "task3_logistic_regression_churn_pipeline.joblib"
    joblib.dump(best_pipe, model_path)

    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    preds_df = X_test[["State", "Area code", "International plan", "Total day minutes", "Customer service calls"]].copy()
    preds_df["y_true_churn"] = y_test.to_numpy()
    preds_df["pred_proba_churn"] = np.round(test_proba, 6)
    preds_df["y_pred_default_050"] = (test_proba >= 0.50).astype(int)
    preds_df["y_pred_oof_optimal"] = (test_proba >= optimal_thr).astype(int)
    preds_df.to_csv(dirs["predictions_dir"] / "task3_logreg_churn_test_predictions.csv", index=False)

    # Figures
    fig5 = plot_logreg_roc_pr_threshold(
        eval_default=test_eval_default,
        eval_opt=test_eval_optimal,
        oof_threshold_info=oof_thr_info,
        out_path=dirs["figures_dir"] / "fig05_logreg_roc_pr_and_threshold_sweep.png",
    )
    fig6 = plot_logreg_odds_ratios_and_vif(
        odds_df=odds_df,
        vif_before=vif_before,
        vif_after=vif_after,
        out_path=dirs["figures_dir"] / "fig06_logreg_odds_ratios_and_vif.png",
    )
    fig7 = plot_binary_confusion_pair(
        eval_a=test_eval_default,
        eval_b=test_eval_optimal,
        label_a="A. Default Threshold (tau = 0.50)",
        label_b=f"B. OOF-Calibrated Threshold (tau* = {optimal_thr:.2f})",
        suptitle="Task 3 — Logistic Regression Held-Out Confusion Matrices (churn-bigml-20, n = 667)",
        out_path=dirs["figures_dir"] / "fig07_logreg_confusion_matrices.png",
    )

    # Keep ROC and PR curve arrays in primary evaluation JSON for unified cross-model plotting.
    summary: dict[str, Any] = {
        "task_id": "Task 3",
        "official_level": "Level 2 (Intermediate) — Task 1",
        "title": "Logistic Regression for Binary Classification",
        "dataset": "Customer Churn (churn-bigml-80 train / churn-bigml-20 test)",
        "runtime": get_runtime_metadata(),
        "best_hyperparameters": {
            "C": float(grid.best_params_["clf__C"]),
            "class_weight": grid.best_params_["clf__class_weight"],
            "cv_roc_auc_mean": round(float(grid.best_score_), 6),
            "oof_selected_threshold": optimal_thr,
            "oof_f1_at_selected_threshold": round(float(oof_thr_info["oof_f1_at_optimal"]), 6),
            "oof_cost_optimal_threshold_fn5_fp1": float(oof_thr_info["optimal_cost_threshold"]),
        },
        "vif_summary": {
            "max_vif_all_15_numeric": float(vif_before["vif"].max()),
            "max_vif_pruned_11_numeric": float(vif_after["vif"].max()),
        },
        "top_positive_odds_ratios": odds_df.head(5)[
            ["clean_feature", "scale_basis", "coef", "odds_ratio", "or_ci_low", "or_ci_high"]
        ].to_dict(orient="records"),
        "top_negative_odds_ratios": odds_df.tail(5)[
            ["clean_feature", "scale_basis", "coef", "odds_ratio", "or_ci_low", "or_ci_high"]
        ].to_dict(orient="records"),
        "dummy_baseline_test": {k: v for k, v in dummy_eval.items() if k != "curves"},
        "untuned_logreg_test_050": {k: v for k, v in untuned_eval.items() if k != "curves"},
        "tuned_logreg_test_default_050": test_eval_default,
        "tuned_logreg_test_oof_optimal": test_eval_optimal,
        "model_artifact": str(model_path.relative_to(dirs["root"])),
        "figures": [
            str(fig5.relative_to(dirs["root"])),
            str(fig6.relative_to(dirs["root"])),
            str(fig7.relative_to(dirs["root"])),
        ],
    }
    save_json(summary, dirs["metrics_dir"] / "task3_logistic_regression_metrics.json")
    return summary
