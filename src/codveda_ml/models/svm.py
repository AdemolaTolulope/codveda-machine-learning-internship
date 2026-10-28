"""Task 6 (Level 3 Advanced): Support Vector Machine (Linear vs RBF) & 2D Boundary Geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from codveda_ml.config import ensure_directories, get_runtime_metadata, load_config, save_json
from codveda_ml.data.loaders import load_churn_splits
from codveda_ml.evaluation.metrics import (
    evaluate_binary_classifier,
    select_optimal_threshold_oof,
)
from codveda_ml.preprocessing.pipelines import build_churn_preprocessor
from codveda_ml.visualization.plots import (
    plot_svm_decision_boundaries,
    plot_svm_kernel_comparison,
)


def run_task6_svm(root: Path | None = None) -> dict[str, Any]:
    """Train, tune, compare Linear vs RBF SVMs, and visualise 2D decision boundaries on Churn."""
    dirs = ensure_directories(root)
    cfg = load_config(dirs["root"])
    seed = int(cfg["project"]["random_seed"])
    lin_c_grid = [float(c) for c in cfg["tasks"]["task6_svm"]["linear_c_grid"]]
    rbf_c_grid = [float(c) for c in cfg["tasks"]["task6_svm"]["rbf_c_grid"]]
    rbf_gamma_grid = cfg["tasks"]["task6_svm"]["rbf_gamma_grid"]

    churn = load_churn_splits(dirs["root"])
    X_train, X_test = churn["X_train"], churn["X_test"]
    y_train, y_test = churn["y_train"], churn["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Dummy baseline
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_eval = evaluate_binary_classifier(dummy, X_test, y_test, threshold=0.50)

    # 2. Linear Kernel SVM GridSearchCV inside leak-free scaling pipeline
    lin_base = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)),
            ("svm", SVC(kernel="linear", probability=True, random_state=seed)),
        ]
    )
    lin_grid = GridSearchCV(
        lin_base,
        param_grid={"svm__C": lin_c_grid, "svm__class_weight": [None, "balanced"]},
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    lin_grid.fit(X_train, y_train)
    best_lin_pipe: Pipeline = lin_grid.best_estimator_

    lin_oof_proba = cross_val_predict(
        clone(best_lin_pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    lin_thr_info = select_optimal_threshold_oof(y_train, lin_oof_proba)
    lin_opt_thr = float(lin_thr_info["optimal_f1_threshold"])
    lin_test_default = evaluate_binary_classifier(best_lin_pipe, X_test, y_test, threshold=0.50)
    lin_test_optimal = evaluate_binary_classifier(best_lin_pipe, X_test, y_test, threshold=lin_opt_thr)

    # 3. RBF Kernel SVM GridSearchCV inside leak-free scaling pipeline
    rbf_base = Pipeline(
        [
            ("prep", build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]
    )
    rbf_grid = GridSearchCV(
        rbf_base,
        param_grid={
            "svm__C": rbf_c_grid,
            "svm__gamma": rbf_gamma_grid,
            "svm__class_weight": [None, "balanced"],
        },
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    rbf_grid.fit(X_train, y_train)
    best_rbf_pipe: Pipeline = rbf_grid.best_estimator_

    rbf_oof_proba = cross_val_predict(
        clone(best_rbf_pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    rbf_thr_info = select_optimal_threshold_oof(y_train, rbf_oof_proba)
    rbf_opt_thr = float(rbf_thr_info["optimal_f1_threshold"])
    rbf_test_default = evaluate_binary_classifier(best_rbf_pipe, X_test, y_test, threshold=0.50)
    rbf_test_optimal = evaluate_binary_classifier(best_rbf_pipe, X_test, y_test, threshold=rbf_opt_thr)

    # Save full CV tables
    pd.DataFrame(lin_grid.cv_results_)[
        ["param_svm__C", "param_svm__class_weight", "mean_test_roc_auc", "mean_test_pr_auc", "mean_test_f1"]
    ].sort_values("mean_test_roc_auc", ascending=False).to_csv(
        dirs["tables_dir"] / "task6_svm_linear_cv_results.csv", index=False
    )
    pd.DataFrame(rbf_grid.cv_results_)[
        [
            "param_svm__C",
            "param_svm__gamma",
            "param_svm__class_weight",
            "mean_test_roc_auc",
            "mean_test_pr_auc",
            "mean_test_f1",
        ]
    ].sort_values("mean_test_roc_auc", ascending=False).to_csv(
        dirs["tables_dir"] / "task6_svm_rbf_cv_results.csv", index=False
    )

    kernel_summary_df = pd.DataFrame(
        [
            {
                "kernel": "linear",
                "best_C": float(lin_grid.best_params_["svm__C"]),
                "best_gamma": "N/A",
                "class_weight": str(lin_grid.best_params_["svm__class_weight"]),
                "cv_roc_auc": float(lin_grid.best_score_),
                "oof_threshold": lin_opt_thr,
                "test_roc_auc": lin_test_optimal["roc_auc"],
                "test_pr_auc": lin_test_optimal["pr_auc"],
                "test_accuracy": lin_test_optimal["accuracy"],
                "test_precision": lin_test_optimal["precision"],
                "test_recall": lin_test_optimal["recall"],
                "test_f1": lin_test_optimal["f1"],
            },
            {
                "kernel": "rbf",
                "best_C": float(rbf_grid.best_params_["svm__C"]),
                "best_gamma": str(rbf_grid.best_params_["svm__gamma"]),
                "class_weight": str(rbf_grid.best_params_["svm__class_weight"]),
                "cv_roc_auc": float(rbf_grid.best_score_),
                "oof_threshold": rbf_opt_thr,
                "test_roc_auc": rbf_test_optimal["roc_auc"],
                "test_pr_auc": rbf_test_optimal["pr_auc"],
                "test_accuracy": rbf_test_optimal["accuracy"],
                "test_precision": rbf_test_optimal["precision"],
                "test_recall": rbf_test_optimal["recall"],
                "test_f1": rbf_test_optimal["f1"],
            },
        ]
    )
    kernel_summary_df.to_csv(dirs["tables_dir"] / "task6_svm_kernel_comparison.csv", index=False)

    # 4. Construct 2D Decision Boundary representations (with explicit projection honesty)
    pair_cols = ["Total day minutes", "Customer service calls"]
    scaler_2d = StandardScaler().fit(X_train[pair_cols])
    X_tr_2d = scaler_2d.transform(X_train[pair_cols])
    X_te_2d = scaler_2d.transform(X_test[pair_cols])

    svm_lin_2d = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=seed).fit(
        X_tr_2d, y_train
    )
    svm_rbf_2d = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed).fit(
        X_tr_2d, y_train
    )

    x_min, x_max = X_tr_2d[:, 0].min() - 0.5, X_tr_2d[:, 0].max() + 0.5
    y_min, y_max = X_tr_2d[:, 1].min() - 0.5, X_tr_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    mesh_pts = np.c_[xx.ravel(), yy.ravel()]
    Z_lin = svm_lin_2d.predict_proba(mesh_pts)[:, 1].reshape(xx.shape)
    Z_rbf = svm_rbf_2d.predict_proba(mesh_pts)[:, 1].reshape(xx.shape)

    # Panel C: 2D RBF SVM trained on top 2 Principal Components (demonstrates unsupervised 2D projection loss)
    X_tr_full_tx = best_rbf_pipe.named_steps["prep"].transform(X_train)
    X_te_full_tx = best_rbf_pipe.named_steps["prep"].transform(X_test)
    pca_full = PCA(n_components=2, random_state=seed).fit(X_tr_full_tx)
    X_tr_pca = pca_full.transform(X_tr_full_tx)
    X_te_pca = pca_full.transform(X_te_full_tx)

    svm_pca_2d = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed).fit(
        X_tr_pca, y_train
    )

    px_min = min(X_tr_pca[:, 0].min(), X_te_pca[:, 0].min()) - 0.5
    px_max = max(X_tr_pca[:, 0].max(), X_te_pca[:, 0].max()) + 0.5
    py_min = min(X_tr_pca[:, 1].min(), X_te_pca[:, 1].max()) - 0.5
    py_min = min(X_tr_pca[:, 1].min(), X_te_pca[:, 1].min()) - 0.5
    py_max = max(X_tr_pca[:, 1].max(), X_te_pca[:, 1].max()) + 0.5
    xx_p, yy_p = np.meshgrid(np.linspace(px_min, px_max, 150), np.linspace(py_min, py_max, 150))
    Z_pca_slice = svm_pca_2d.predict_proba(np.c_[xx_p.ravel(), yy_p.ravel()])[:, 1].reshape(xx_p.shape)

    boundary_data = {
        "xx": xx,
        "yy": yy,
        "Z_linear": Z_lin,
        "Z_rbf": Z_rbf,
        "X_test_2d": X_te_2d,
        "y_test": y_test.to_numpy(),
        "xx_pca": xx_p,
        "yy_pca": yy_p,
        "Z_pca_slice": Z_pca_slice,
        "X_test_pca2": X_te_pca[:, :2],
        "n_full_features": int(X_tr_full_tx.shape[1]),
        "pc1_var": float(pca_full.explained_variance_ratio_[0]),
        "pc2_var": float(pca_full.explained_variance_ratio_[1]),
    }

    # Select final pipeline strictly from CV score (RBF vs Linear)
    selected_kernel = "rbf" if rbf_grid.best_score_ >= lin_grid.best_score_ else "linear"
    selected_pipe = best_rbf_pipe if selected_kernel == "rbf" else best_lin_pipe
    selected_opt_thr = rbf_opt_thr if selected_kernel == "rbf" else lin_opt_thr

    model_path = dirs["models_dir"] / "task6_svm_churn_pipeline.joblib"
    joblib.dump(selected_pipe, model_path)

    test_proba = selected_pipe.predict_proba(X_test)[:, 1]
    preds_df = X_test[["State", "Area code", "International plan", "Total day minutes", "Customer service calls"]].copy()
    preds_df["y_true_churn"] = y_test.to_numpy()
    preds_df["pred_proba_churn"] = np.round(test_proba, 6)
    preds_df["y_pred_default_050"] = (test_proba >= 0.50).astype(int)
    preds_df["y_pred_oof_optimal"] = (test_proba >= selected_opt_thr).astype(int)
    preds_df.to_csv(dirs["predictions_dir"] / "task6_svm_churn_test_predictions.csv", index=False)

    # Figures
    fig14 = plot_svm_decision_boundaries(
        boundary_data=boundary_data,
        out_path=dirs["figures_dir"] / "fig14_svm_decision_boundaries_2d.png",
    )
    fig15 = plot_svm_kernel_comparison(
        kernel_summary_df=kernel_summary_df,
        rbf_eval=rbf_test_optimal,
        out_path=dirs["figures_dir"] / "fig15_svm_kernel_comparison_and_confusion.png",
    )

    summary: dict[str, Any] = {
        "task_id": "Task 6",
        "official_level": "Level 3 (Advanced) — Task 2",
        "title": "Support Vector Machine (SVM) for Classification",
        "dataset": "Customer Churn (churn-bigml-80 train / churn-bigml-20 test)",
        "runtime": get_runtime_metadata(),
        "selected_kernel": selected_kernel,
        "linear_svm": {
            "best_params": {
                "C": float(lin_grid.best_params_["svm__C"]),
                "class_weight": lin_grid.best_params_["svm__class_weight"],
            },
            "cv_roc_auc_mean": round(float(lin_grid.best_score_), 6),
            "oof_selected_threshold": lin_opt_thr,
            "test_default_050": {k: v for k, v in lin_test_default.items() if k != "curves"},
            "test_oof_optimal": lin_test_optimal,
        },
        "rbf_svm": {
            "best_params": {
                "C": float(rbf_grid.best_params_["svm__C"]),
                "gamma": rbf_grid.best_params_["svm__gamma"],
                "class_weight": rbf_grid.best_params_["svm__class_weight"],
            },
            "cv_roc_auc_mean": round(float(rbf_grid.best_score_), 6),
            "oof_selected_threshold": rbf_opt_thr,
            "n_support_vectors_total": int(best_rbf_pipe.named_steps["svm"].n_support_.sum()),
            "test_default_050": {k: v for k, v in rbf_test_default.items() if k != "curves"},
            "test_oof_optimal": rbf_test_optimal,
        },
        "pca_2d_explained_variance_total": round(
            float(pca_full.explained_variance_ratio_[:2].sum()), 6
        ),
        "dummy_baseline_test": {k: v for k, v in dummy_eval.items() if k != "curves"},
        "model_artifact": str(model_path.relative_to(dirs["root"])),
        "figures": [
            str(fig14.relative_to(dirs["root"])),
            str(fig15.relative_to(dirs["root"])),
        ],
    }
    save_json(summary, dirs["metrics_dir"] / "task6_svm_metrics.json")
    return summary
