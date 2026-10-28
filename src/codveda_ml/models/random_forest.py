"""Task 5 (Level 3 Advanced): Random Forest Classifier, Feature Importance & Stability on Churn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from codveda_ml.config import ensure_directories, get_runtime_metadata, load_config, save_json
from codveda_ml.data.loaders import CHURN_REDUNDANT_CHARGE_COLS, load_churn_splits
from codveda_ml.evaluation.metrics import (
    evaluate_binary_classifier,
    select_optimal_threshold_oof,
)
from codveda_ml.preprocessing.pipelines import build_churn_preprocessor
from codveda_ml.visualization.plots import (
    plot_rf_feature_importances,
    plot_rf_vs_logreg_and_stability,
    plot_single_binary_confusion,
)


def _aggregate_mdi_by_semantic_feature(
    rf_clf: RandomForestClassifier,
    transformed_feature_names: list[str],
    semantic_features: list[str],
) -> pd.DataFrame:
    """
    Aggregate tree-level Mean Decrease in Impurity (MDI) across one-hot columns
    belonging to the same parent categorical variable (e.g., `State_*`, `Area code_*`)
    so MDI can be compared directly on the exact same 15 parent features as
    raw-feature Permutation Importance.
    """
    n_trees = len(rf_clf.estimators_)
    tree_imps = np.array([est.feature_importances_ for est in rf_clf.estimators_])  # (n_trees, n_tx)

    rows = []
    for sem in semantic_features:
        if sem in ("State", "Area code", "International plan", "Voice mail plan"):
            prefix = f"{sem}_"
            idxs = [i for i, fn in enumerate(transformed_feature_names) if fn.startswith(prefix)]
        else:
            idxs = [i for i, fn in enumerate(transformed_feature_names) if fn == sem]
        per_tree_sum = tree_imps[:, idxs].sum(axis=1)
        rows.append(
            {
                "feature": sem,
                "n_encoded_columns": len(idxs),
                "mdi_importance": float(per_tree_sum.mean()),
                "mdi_std": float(per_tree_sum.std(ddof=1)) if n_trees > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("mdi_importance", ascending=False).reset_index(drop=True)


def run_task5_random_forest(root: Path | None = None) -> dict[str, Any]:
    """Train, cross-validate, analyse feature importance & stability for Random Forest on Churn."""
    dirs = ensure_directories(root)
    cfg = load_config(dirs["root"])
    seed = int(cfg["project"]["random_seed"])
    stability_seeds = [int(s) for s in cfg["tasks"]["task5_random_forest"]["stability_seeds"]]
    perm_repeats = int(cfg["tasks"]["task5_random_forest"]["permutation_repeats"])

    churn = load_churn_splits(dirs["root"])
    X_train, X_test = churn["X_train"], churn["X_test"]
    y_train, y_test = churn["y_train"], churn["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Baselines: Dummy + Untuned Random Forest
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_eval = evaluate_binary_classifier(dummy, X_test, y_test, threshold=0.50)

    untuned_rf = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=False)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ]
    )
    untuned_rf.fit(X_train, y_train)
    untuned_eval = evaluate_binary_classifier(untuned_rf, X_test, y_test, threshold=0.50)

    # 2. Hyperparameter Tuning via 5-Fold Stratified CV on churn-bigml-80 ONLY
    base_rf = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=False)),
            ("rf", RandomForestClassifier(random_state=seed, n_jobs=-1)),
        ]
    )
    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [12, 16, None],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", 0.4],
        "rf__class_weight": [None, "balanced_subsample"],
    }
    grid = GridSearchCV(
        base_rf,
        param_grid=param_grid,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        refit="pr_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    cv_df = pd.DataFrame(grid.cv_results_)[
        [
            "param_rf__n_estimators",
            "param_rf__max_depth",
            "param_rf__min_samples_leaf",
            "param_rf__max_features",
            "param_rf__class_weight",
            "mean_test_pr_auc",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_f1",
        ]
    ].sort_values("mean_test_pr_auc", ascending=False)
    cv_df.to_csv(dirs["tables_dir"] / "task5_random_forest_cv_results.csv", index=False)

    best_rf_pipe: Pipeline = grid.best_estimator_

    # 3. Leak-free Out-of-Fold Probability Threshold Selection on Training Split ONLY
    oof_proba = cross_val_predict(
        clone(best_rf_pipe),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    oof_thr_info = select_optimal_threshold_oof(y_train, oof_proba, fn_cost=5.0, fp_cost=1.0)
    opt_thr = float(oof_thr_info["optimal_f1_threshold"])

    test_eval_default = evaluate_binary_classifier(best_rf_pipe, X_test, y_test, threshold=0.50)
    test_eval_optimal = evaluate_binary_classifier(best_rf_pipe, X_test, y_test, threshold=opt_thr)

    # 4. Feature Importance Analysis: Impurity (MDI) vs Permutation Importance
    semantic_features = [c for c in X_train.columns if c not in CHURN_REDUNDANT_CHARGE_COLS]
    tx_names = list(best_rf_pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out())
    mdi_df = _aggregate_mdi_by_semantic_feature(
        best_rf_pipe.named_steps["rf"], tx_names, semantic_features
    )

    # Permutation importance on raw semantic columns (restricting to retained 15 non-redundant cols)
    perm_res = permutation_importance(
        best_rf_pipe,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=perm_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    perm_all_df = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "perm_importance_mean": perm_res.importances_mean,
            "perm_importance_std": perm_res.importances_std,
        }
    )
    perm_df = (
        perm_all_df[perm_all_df["feature"].isin(semantic_features)]
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    imp_merged = pd.merge(mdi_df, perm_df, on="feature")
    imp_merged.to_csv(dirs["tables_dir"] / "task5_rf_feature_importance_comparison.csv", index=False)

    # Collinearity dilution check: what happens to MDI if redundant *charge cols are NOT dropped?
    collinear_rf = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=False, scale_numeric=False)),
            ("rf", clone(best_rf_pipe.named_steps["rf"])),
        ]
    ).fit(X_train, y_train)
    col_tx_names = list(collinear_rf.named_steps["prep"].named_steps["columns"].get_feature_names_out())
    col_mdi = pd.Series(collinear_rf.named_steps["rf"].feature_importances_, index=col_tx_names)

    # 5. Model Stability Assessment Across 10 Independent Random Seeds
    stability_rows: list[dict[str, Any]] = []
    best_rf_params = best_rf_pipe.named_steps["rf"].get_params()
    for s in stability_seeds:
        params_s = dict(best_rf_params)
        params_s["random_state"] = s
        pipe_s = Pipeline(
            [
                ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=False)),
                ("rf", RandomForestClassifier(**params_s)),
            ]
        )
        pipe_s.fit(X_train, y_train)
        ev_s = evaluate_binary_classifier(pipe_s, X_test, y_test, threshold=opt_thr)
        stability_rows.append(
            {
                "seed": s,
                "accuracy": ev_s["accuracy"],
                "precision": ev_s["precision"],
                "recall": ev_s["recall"],
                "f1": ev_s["f1"],
                "roc_auc": ev_s["roc_auc"],
                "pr_auc": ev_s["pr_auc"],
            }
        )
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(dirs["tables_dir"] / "task5_rf_seed_stability.csv", index=False)

    # Load Task 3 Logistic Regression metrics for direct comparison plot
    logreg_metrics_path = dirs["metrics_dir"] / "task3_logistic_regression_metrics.json"
    if logreg_metrics_path.exists():
        with open(logreg_metrics_path, "r", encoding="utf-8") as fh:
            logreg_eval = json.load(fh)["tuned_logreg_test_oof_optimal"]
    else:
        from codveda_ml.models.logistic_regression import run_task3_logistic_regression
        logreg_eval = run_task3_logistic_regression(dirs["root"])["tuned_logreg_test_oof_optimal"]

    # Persist pipeline & test predictions
    model_path = dirs["models_dir"] / "task5_random_forest_churn_pipeline.joblib"
    joblib.dump(best_rf_pipe, model_path)

    test_proba = best_rf_pipe.predict_proba(X_test)[:, 1]
    preds_df = X_test[["State", "Area code", "International plan", "Total day minutes", "Customer service calls"]].copy()
    preds_df["y_true_churn"] = y_test.to_numpy()
    preds_df["pred_proba_churn"] = np.round(test_proba, 6)
    preds_df["y_pred_default_050"] = (test_proba >= 0.50).astype(int)
    preds_df["y_pred_oof_optimal"] = (test_proba >= opt_thr).astype(int)
    preds_df.to_csv(dirs["predictions_dir"] / "task5_rf_churn_test_predictions.csv", index=False)

    # Figures
    fig11 = plot_rf_feature_importances(
        mdi_df=mdi_df,
        perm_df=perm_df,
        out_path=dirs["figures_dir"] / "fig11_rf_feature_importance_mdi_vs_permutation.png",
    )
    fig12 = plot_rf_vs_logreg_and_stability(
        rf_eval=test_eval_optimal,
        logreg_eval=logreg_eval,
        stability_df=stability_df,
        out_path=dirs["figures_dir"] / "fig12_rf_roc_pr_and_seed_stability.png",
    )
    fig13 = plot_single_binary_confusion(
        eval_res=test_eval_optimal,
        title=f"Task 5 — Tuned Random Forest Held-Out Confusion Matrix (Thr = {opt_thr:.2f})",
        out_path=dirs["figures_dir"] / "fig13_rf_confusion_matrix.png",
    )

    summary: dict[str, Any] = {
        "task_id": "Task 5",
        "official_level": "Level 3 (Advanced) — Task 1",
        "title": "Random Forest Classifier",
        "dataset": "Customer Churn (churn-bigml-80 train / churn-bigml-20 test)",
        "runtime": get_runtime_metadata(),
        "best_hyperparameters": {
            "n_estimators": int(grid.best_params_["rf__n_estimators"]),
            "max_depth": grid.best_params_["rf__max_depth"],
            "min_samples_leaf": int(grid.best_params_["rf__min_samples_leaf"]),
            "max_features": grid.best_params_["rf__max_features"],
            "class_weight": grid.best_params_["rf__class_weight"],
            "cv_pr_auc_mean": round(float(grid.best_score_), 6),
            "oof_selected_threshold": opt_thr,
            "oof_f1_at_selected_threshold": round(float(oof_thr_info["oof_f1_at_optimal"]), 6),
        },
        "collinearity_mdi_dilution_demo": {
            "pruned_total_day_minutes_mdi": round(
                float(mdi_df.loc[mdi_df["feature"] == "Total day minutes", "mdi_importance"].iloc[0]), 6
            ),
            "unpruned_total_day_minutes_mdi": round(float(col_mdi["Total day minutes"]), 6),
            "unpruned_total_day_charge_mdi": round(float(col_mdi["Total day charge"]), 6),
        },
        "state_high_cardinality_bias_demo": {
            "state_mdi_sum_51_dummies": round(
                float(mdi_df.loc[mdi_df["feature"] == "State", "mdi_importance"].iloc[0]), 6
            ),
            "state_permutation_auc_drop": round(
                float(perm_df.loc[perm_df["feature"] == "State", "perm_importance_mean"].iloc[0]), 6
            ),
        },
        "stability_10_seeds": {
            "roc_auc_mean": round(float(stability_df["roc_auc"].mean()), 6),
            "roc_auc_std": round(float(stability_df["roc_auc"].std(ddof=1)), 6),
            "pr_auc_mean": round(float(stability_df["pr_auc"].mean()), 6),
            "pr_auc_std": round(float(stability_df["pr_auc"].std(ddof=1)), 6),
            "f1_mean": round(float(stability_df["f1"].mean()), 6),
            "f1_std": round(float(stability_df["f1"].std(ddof=1)), 6),
        },
        "dummy_baseline_test": {k: v for k, v in dummy_eval.items() if k != "curves"},
        "untuned_rf_test_050": {k: v for k, v in untuned_eval.items() if k != "curves"},
        "tuned_rf_test_default_050": {k: v for k, v in test_eval_default.items() if k != "curves"},
        "tuned_rf_test_oof_optimal": test_eval_optimal,
        "top_10_features_by_permutation": imp_merged.sort_values(
            "perm_importance_mean", ascending=False
        ).head(10).to_dict(orient="records"),
        "model_artifact": str(model_path.relative_to(dirs["root"])),
        "figures": [
            str(fig11.relative_to(dirs["root"])),
            str(fig12.relative_to(dirs["root"])),
            str(fig13.relative_to(dirs["root"])),
        ],
    }
    save_json(summary, dirs["metrics_dir"] / "task5_random_forest_metrics.json")
    return summary
