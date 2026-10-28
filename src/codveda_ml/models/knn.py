"""Task 2 (Level 1 Basic): K-Nearest Neighbors Classifier on the Iris Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from codveda_ml.config import ensure_directories, get_runtime_metadata, load_config, save_json
from codveda_ml.data.loaders import get_iris_splits
from codveda_ml.evaluation.metrics import evaluate_multiclass_classifier
from codveda_ml.preprocessing.pipelines import build_iris_preprocessor
from codveda_ml.visualization.plots import (
    plot_knn_k_sweep,
    plot_multiclass_confusion_and_scatter,
)


def run_task2_knn(root: Path | None = None) -> dict[str, Any]:
    """Train, cross-validate, select, and evaluate KNN classifiers on Iris."""
    dirs = ensure_directories(root)
    cfg = load_config(dirs["root"])
    seed = int(cfg["project"]["random_seed"])
    k_grid: list[int] = [int(k) for k in cfg["tasks"]["task2_knn"]["k_grid"]]

    iris = get_iris_splits(dirs["root"], test_size=0.20, random_state=seed)
    X_train, X_test = iris["X_train"], iris["X_test"]
    y_train, y_test = iris["y_train"], iris["y_test"]
    class_names = iris["class_names"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Stratified Dummy Baseline
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(X_train, y_train)
    dummy_test_eval = evaluate_multiclass_classifier(dummy, X_test, y_test, class_names)

    # 2. Systematic CV sweep over K, Scaling (True/False), and Weighting (uniform/distance)
    cv_rows: list[dict[str, Any]] = []
    for scaled in [True, False]:
        for weights in ["uniform", "distance"]:
            for k in k_grid:
                pipe = Pipeline(
                    [
                        ("prep", build_iris_preprocessor(scale=scaled)),
                        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights)),
                    ]
                )
                scores = cross_validate(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring={"acc": "accuracy", "f1_macro": "f1_macro"},
                    return_train_score=True,
                )
                cv_rows.append(
                    {
                        "k": k,
                        "scaled": scaled,
                        "weights": weights,
                        "train_acc_mean": float(np.mean(scores["train_acc"])),
                        "cv_acc_mean": float(np.mean(scores["test_acc"])),
                        "cv_acc_std": float(np.std(scores["test_acc"], ddof=1)),
                        "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                        "cv_f1_macro_std": float(np.std(scores["test_f1_macro"], ddof=1)),
                    }
                )

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(dirs["tables_dir"] / "task2_knn_cv_sweep.csv", index=False)

    # Select best scaled configuration strictly from training CV:
    # highest cv_f1_macro_mean -> lowest cv_f1_macro_std -> prefer uniform & moderate odd K
    scaled_candidates = cv_df[cv_df["scaled"] == True].copy()
    scaled_candidates["is_uniform"] = (scaled_candidates["weights"] == "uniform").astype(int)
    scaled_candidates = scaled_candidates.sort_values(
        by=["cv_f1_macro_mean", "cv_f1_macro_std", "is_uniform", "k"],
        ascending=[False, True, False, True],
    )
    best_row = scaled_candidates.iloc[0]
    selected_k = int(best_row["k"])
    selected_weights = str(best_row["weights"])

    # Also fit unscaled counterpart at same K & weights to quantify scaling effect on test set
    final_scaled_pipe = Pipeline(
        [
            ("prep", build_iris_preprocessor(scale=True)),
            ("knn", KNeighborsClassifier(n_neighbors=selected_k, weights=selected_weights)),
        ]
    )
    final_scaled_pipe.fit(X_train, y_train)
    scaled_test_eval = evaluate_multiclass_classifier(
        final_scaled_pipe, X_test, y_test, class_names
    )

    unscaled_pipe = Pipeline(
        [
            ("prep", build_iris_preprocessor(scale=False)),
            ("knn", KNeighborsClassifier(n_neighbors=selected_k, weights=selected_weights)),
        ]
    )
    unscaled_pipe.fit(X_train, y_train)
    unscaled_test_eval = evaluate_multiclass_classifier(
        unscaled_pipe, X_test, y_test, class_names
    )

    # Duplicate-free sensitivity check (verifying duplicates do not inflate Iris result)
    dedup_df = iris["df_raw"].drop_duplicates().reset_index(drop=True)
    lbl_map = iris["label_to_idx"]
    dedup_cv_scores = cross_validate(
        Pipeline(
            [
                ("prep", build_iris_preprocessor(scale=True)),
                ("knn", KNeighborsClassifier(n_neighbors=selected_k, weights=selected_weights)),
            ]
        ),
        dedup_df[X_train.columns],
        dedup_df["species"].map(lbl_map),
        cv=cv,
        scoring="f1_macro",
    )

    # Persist model & test predictions
    model_path = dirs["models_dir"] / "task2_knn_iris_pipeline.joblib"
    joblib.dump(final_scaled_pipe, model_path)

    proba_test = final_scaled_pipe.predict_proba(X_test)
    preds_df = X_test.copy()
    preds_df["y_true_id"] = y_test.to_numpy()
    preds_df["y_true_species"] = [class_names[i] for i in y_test]
    preds_df["y_pred_id"] = final_scaled_pipe.predict(X_test)
    preds_df["y_pred_species"] = [class_names[i] for i in preds_df["y_pred_id"]]
    for idx, cname in enumerate(class_names):
        preds_df[f"prob_{cname}"] = np.round(proba_test[:, idx], 6)
    preds_df.to_csv(dirs["predictions_dir"] / "task2_knn_iris_test_predictions.csv", index=False)

    # Figures
    fig3 = plot_knn_k_sweep(
        cv_df=cv_df,
        selected_k=selected_k,
        selected_weights=selected_weights,
        out_path=dirs["figures_dir"] / "fig03_knn_k_curve_scaled_vs_unscaled.png",
    )
    fig4 = plot_multiclass_confusion_and_scatter(
        cm=scaled_test_eval["confusion_matrix"],
        class_names=class_names,
        iris_df=iris["df_raw"],
        title_prefix=f"Task 2 — Selected Scaled KNN (K = {selected_k}, {selected_weights}) on Held-Out Iris Test Split",
        subtitle_metrics=(
            f"Acc = {scaled_test_eval['accuracy']:.4f} | Macro-F1 = {scaled_test_eval['f1_macro']:.4f}"
        ),
        out_path=dirs["figures_dir"] / "fig04_knn_iris_confusion_and_feature_space.png",
    )

    summary: dict[str, Any] = {
        "task_id": "Task 2",
        "official_level": "Level 1 (Basic) — Task 3",
        "title": "K-Nearest Neighbors (KNN) Classifier",
        "dataset": "1) iris.csv",
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "stratified": True, "random_state": seed},
        "runtime": get_runtime_metadata(),
        "selected_hyperparameters": {
            "n_neighbors": selected_k,
            "weights": selected_weights,
            "scaled": True,
            "cv_f1_macro_mean": round(float(best_row["cv_f1_macro_mean"]), 6),
            "cv_f1_macro_std": round(float(best_row["cv_f1_macro_std"]), 6),
            "cv_accuracy_mean": round(float(best_row["cv_acc_mean"]), 6),
        },
        "deduplicated_147_rows_cv_f1_macro_mean": round(float(np.mean(dedup_cv_scores["test_score"])), 6),
        "dummy_baseline_test": dummy_test_eval,
        "unscaled_same_k_test": unscaled_test_eval,
        "final_scaled_knn_test": scaled_test_eval,
        "model_artifact": str(model_path.relative_to(dirs["root"])),
        "figures": [str(fig3.relative_to(dirs["root"])), str(fig4.relative_to(dirs["root"]))],
    }
    save_json(summary, dirs["metrics_dir"] / "task2_knn_metrics.json")
    return summary
