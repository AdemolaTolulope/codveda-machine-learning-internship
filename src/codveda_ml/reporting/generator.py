"""Build unified model comparison tables, task traceability matrices, and Figure 16."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from codveda_ml.config import ensure_directories, save_json
from codveda_ml.visualization.plots import plot_portfolio_benchmark


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_portfolio_summaries(root: Path | None = None) -> dict[str, Any]:
    """Read verified JSON outputs from Tasks 1–6 and emit unified comparison & traceability CSVs."""
    dirs = ensure_directories(root)
    m_dir = dirs["metrics_dir"]

    t1 = _load_json(m_dir / "task1_preprocessing_metrics.json")
    t2 = _load_json(m_dir / "task2_knn_metrics.json")
    t3 = _load_json(m_dir / "task3_logistic_regression_metrics.json")
    t4 = _load_json(m_dir / "task4_decision_tree_metrics.json")
    t5 = _load_json(m_dir / "task5_random_forest_metrics.json")
    t6 = _load_json(m_dir / "task6_svm_metrics.json")

    # 1. Unified Model Comparison Table across all baselines and tuned models
    rows: list[dict[str, Any]] = [
        # Iris Multiclass Benchmark (Test n = 30)
        {
            "task_id": "Task 2 / Task 4 Baseline",
            "level": "Level 1–2",
            "dataset": "Iris (test n=30, stratified 80/20)",
            "model_name": "DummyClassifier (Stratified)",
            "selection_procedure": "Baseline reference",
            "threshold_or_k": "N/A",
            "cv_primary_metric": None,
            "accuracy": round(t2["dummy_baseline_test"]["accuracy"], 4),
            "precision": round(t2["dummy_baseline_test"]["precision_macro"], 4),
            "recall": round(t2["dummy_baseline_test"]["recall_macro"], 4),
            "f1_primary": round(t2["dummy_baseline_test"]["f1_macro"], 4),
            "f1_macro": round(t2["dummy_baseline_test"]["f1_macro"], 4),
            "roc_auc": round(t2["dummy_baseline_test"]["roc_auc_ovr_macro"], 4),
            "pr_auc": None,
        },
        {
            "task_id": "Task 2",
            "level": "Level 1 (Basic)",
            "dataset": "Iris (test n=30, stratified 80/20)",
            "model_name": f"KNN Unscaled (K={t2['selected_hyperparameters']['n_neighbors']})",
            "selection_procedure": "Ablation (raw cm units)",
            "threshold_or_k": f"K={t2['selected_hyperparameters']['n_neighbors']}",
            "cv_primary_metric": None,
            "accuracy": round(t2["unscaled_same_k_test"]["accuracy"], 4),
            "precision": round(t2["unscaled_same_k_test"]["precision_macro"], 4),
            "recall": round(t2["unscaled_same_k_test"]["recall_macro"], 4),
            "f1_primary": round(t2["unscaled_same_k_test"]["f1_macro"], 4),
            "f1_macro": round(t2["unscaled_same_k_test"]["f1_macro"], 4),
            "roc_auc": round(t2["unscaled_same_k_test"]["roc_auc_ovr_macro"], 4),
            "pr_auc": None,
        },
        {
            "task_id": "Task 2",
            "level": "Level 1 (Basic)",
            "dataset": "Iris (test n=30, stratified 80/20)",
            "model_name": f"KNN Scaled (K={t2['selected_hyperparameters']['n_neighbors']}, {t2['selected_hyperparameters']['weights']})",
            "selection_procedure": "5-Fold Stratified CV on Train (n=120)",
            "threshold_or_k": f"K={t2['selected_hyperparameters']['n_neighbors']}",
            "cv_primary_metric": round(t2["selected_hyperparameters"]["cv_f1_macro_mean"], 4),
            "accuracy": round(t2["final_scaled_knn_test"]["accuracy"], 4),
            "precision": round(t2["final_scaled_knn_test"]["precision_macro"], 4),
            "recall": round(t2["final_scaled_knn_test"]["recall_macro"], 4),
            "f1_primary": round(t2["final_scaled_knn_test"]["f1_macro"], 4),
            "f1_macro": round(t2["final_scaled_knn_test"]["f1_macro"], 4),
            "roc_auc": round(t2["final_scaled_knn_test"]["roc_auc_ovr_macro"], 4),
            "pr_auc": None,
        },
        {
            "task_id": "Task 4",
            "level": "Level 2 (Intermediate)",
            "dataset": "Iris (test n=30, stratified 80/20)",
            "model_name": f"Decision Tree Unrestricted (Depth={t4['unrestricted_tree']['depth']}, Leaves={t4['unrestricted_tree']['n_leaves']})",
            "selection_procedure": "Unpruned baseline (ccp_alpha=0.0)",
            "threshold_or_k": "alpha=0.0",
            "cv_primary_metric": round(t4["unrestricted_tree"]["cv_f1_macro_mean"], 4),
            "accuracy": round(t4["unrestricted_tree"]["test_metrics"]["accuracy"], 4),
            "precision": round(t4["unrestricted_tree"]["test_metrics"]["precision_macro"], 4),
            "recall": round(t4["unrestricted_tree"]["test_metrics"]["recall_macro"], 4),
            "f1_primary": round(t4["unrestricted_tree"]["test_metrics"]["f1_macro"], 4),
            "f1_macro": round(t4["unrestricted_tree"]["test_metrics"]["f1_macro"], 4),
            "roc_auc": round(t4["unrestricted_tree"]["test_metrics"]["roc_auc_ovr_macro"], 4),
            "pr_auc": None,
        },
        {
            "task_id": "Task 4",
            "level": "Level 2 (Intermediate)",
            "dataset": "Iris (test n=30, stratified 80/20)",
            "model_name": f"Decision Tree Pruned (Depth={t4['selected_pruned_tree']['fitted_depth']}, Leaves={t4['selected_pruned_tree']['fitted_n_leaves']})",
            "selection_procedure": "5-Fold Stratified CV + Parsimony Pruning on Train (n=120)",
            "threshold_or_k": f"alpha={t4['selected_pruned_tree']['ccp_alpha']}",
            "cv_primary_metric": round(t4["selected_pruned_tree"]["cv_f1_macro_mean"], 4),
            "accuracy": round(t4["selected_pruned_tree"]["test_metrics"]["accuracy"], 4),
            "precision": round(t4["selected_pruned_tree"]["test_metrics"]["precision_macro"], 4),
            "recall": round(t4["selected_pruned_tree"]["test_metrics"]["recall_macro"], 4),
            "f1_primary": round(t4["selected_pruned_tree"]["test_metrics"]["f1_macro"], 4),
            "f1_macro": round(t4["selected_pruned_tree"]["test_metrics"]["f1_macro"], 4),
            "roc_auc": round(t4["selected_pruned_tree"]["test_metrics"]["roc_auc_ovr_macro"], 4),
            "pr_auc": None,
        },
        # Customer Churn Binary Benchmark (Official Held-Out Test n = 667)
        {
            "task_id": "Task 3 / 5 / 6 Baseline",
            "level": "Level 2–3",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "DummyClassifier (Most Frequent)",
            "selection_procedure": "Majority-class baseline",
            "threshold_or_k": "0.50",
            "cv_primary_metric": 0.5000,
            "accuracy": round(t3["dummy_baseline_test"]["accuracy"], 4),
            "precision": round(t3["dummy_baseline_test"]["precision"], 4),
            "recall": round(t3["dummy_baseline_test"]["recall"], 4),
            "f1_primary": round(t3["dummy_baseline_test"]["f1"], 4),
            "f1_macro": round(t3["dummy_baseline_test"]["f1_macro"], 4),
            "roc_auc": round(t3["dummy_baseline_test"]["roc_auc"], 4),
            "pr_auc": round(t3["dummy_baseline_test"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 3",
            "level": "Level 2 (Intermediate)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "Logistic Regression (Default Thr=0.50)",
            "selection_procedure": "5-Fold Stratified CV on churn-bigml-80",
            "threshold_or_k": "0.50",
            "cv_primary_metric": round(t3["best_hyperparameters"]["cv_roc_auc_mean"], 4),
            "accuracy": round(t3["tuned_logreg_test_default_050"]["accuracy"], 4),
            "precision": round(t3["tuned_logreg_test_default_050"]["precision"], 4),
            "recall": round(t3["tuned_logreg_test_default_050"]["recall"], 4),
            "f1_primary": round(t3["tuned_logreg_test_default_050"]["f1"], 4),
            "f1_macro": round(t3["tuned_logreg_test_default_050"]["f1_macro"], 4),
            "roc_auc": round(t3["tuned_logreg_test_default_050"]["roc_auc"], 4),
            "pr_auc": round(t3["tuned_logreg_test_default_050"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 3",
            "level": "Level 2 (Intermediate)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "Logistic Regression (OOF-Calibrated)",
            "selection_procedure": "5-Fold Stratified CV + OOF Threshold on churn-bigml-80",
            "threshold_or_k": f"{t3['best_hyperparameters']['oof_selected_threshold']:.2f}",
            "cv_primary_metric": round(t3["best_hyperparameters"]["cv_roc_auc_mean"], 4),
            "accuracy": round(t3["tuned_logreg_test_oof_optimal"]["accuracy"], 4),
            "precision": round(t3["tuned_logreg_test_oof_optimal"]["precision"], 4),
            "recall": round(t3["tuned_logreg_test_oof_optimal"]["recall"], 4),
            "f1_primary": round(t3["tuned_logreg_test_oof_optimal"]["f1"], 4),
            "f1_macro": round(t3["tuned_logreg_test_oof_optimal"]["f1_macro"], 4),
            "roc_auc": round(t3["tuned_logreg_test_oof_optimal"]["roc_auc"], 4),
            "pr_auc": round(t3["tuned_logreg_test_oof_optimal"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 6",
            "level": "Level 3 (Advanced)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "SVM (Linear Kernel, OOF-Calibrated)",
            "selection_procedure": "5-Fold Stratified CV + OOF Threshold on churn-bigml-80",
            "threshold_or_k": f"{t6['linear_svm']['oof_selected_threshold']:.2f}",
            "cv_primary_metric": round(t6["linear_svm"]["cv_roc_auc_mean"], 4),
            "accuracy": round(t6["linear_svm"]["test_oof_optimal"]["accuracy"], 4),
            "precision": round(t6["linear_svm"]["test_oof_optimal"]["precision"], 4),
            "recall": round(t6["linear_svm"]["test_oof_optimal"]["recall"], 4),
            "f1_primary": round(t6["linear_svm"]["test_oof_optimal"]["f1"], 4),
            "f1_macro": round(t6["linear_svm"]["test_oof_optimal"]["f1_macro"], 4),
            "roc_auc": round(t6["linear_svm"]["test_oof_optimal"]["roc_auc"], 4),
            "pr_auc": round(t6["linear_svm"]["test_oof_optimal"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 6",
            "level": "Level 3 (Advanced)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "SVM (RBF Kernel, OOF-Calibrated)",
            "selection_procedure": "5-Fold Stratified CV + OOF Threshold on churn-bigml-80",
            "threshold_or_k": f"{t6['rbf_svm']['oof_selected_threshold']:.2f}",
            "cv_primary_metric": round(t6["rbf_svm"]["cv_roc_auc_mean"], 4),
            "accuracy": round(t6["rbf_svm"]["test_oof_optimal"]["accuracy"], 4),
            "precision": round(t6["rbf_svm"]["test_oof_optimal"]["precision"], 4),
            "recall": round(t6["rbf_svm"]["test_oof_optimal"]["recall"], 4),
            "f1_primary": round(t6["rbf_svm"]["test_oof_optimal"]["f1"], 4),
            "f1_macro": round(t6["rbf_svm"]["test_oof_optimal"]["f1_macro"], 4),
            "roc_auc": round(t6["rbf_svm"]["test_oof_optimal"]["roc_auc"], 4),
            "pr_auc": round(t6["rbf_svm"]["test_oof_optimal"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 5",
            "level": "Level 3 (Advanced)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "Random Forest (Untuned Baseline)",
            "selection_procedure": "Default 100 trees, Thr=0.50",
            "threshold_or_k": "0.50",
            "cv_primary_metric": None,
            "accuracy": round(t5["untuned_rf_test_050"]["accuracy"], 4),
            "precision": round(t5["untuned_rf_test_050"]["precision"], 4),
            "recall": round(t5["untuned_rf_test_050"]["recall"], 4),
            "f1_primary": round(t5["untuned_rf_test_050"]["f1"], 4),
            "f1_macro": round(t5["untuned_rf_test_050"]["f1_macro"], 4),
            "roc_auc": round(t5["untuned_rf_test_050"]["roc_auc"], 4),
            "pr_auc": round(t5["untuned_rf_test_050"]["pr_auc"], 4),
        },
        {
            "task_id": "Task 5",
            "level": "Level 3 (Advanced)",
            "dataset": "Customer Churn (churn-bigml-20)",
            "model_name": "Random Forest (Tuned, OOF-Calibrated)",
            "selection_procedure": "5-Fold Stratified CV + OOF Threshold on churn-bigml-80",
            "threshold_or_k": f"{t5['best_hyperparameters']['oof_selected_threshold']:.2f}",
            "cv_primary_metric": round(t5["best_hyperparameters"]["cv_pr_auc_mean"], 4),
            "accuracy": round(t5["tuned_rf_test_oof_optimal"]["accuracy"], 4),
            "precision": round(t5["tuned_rf_test_oof_optimal"]["precision"], 4),
            "recall": round(t5["tuned_rf_test_oof_optimal"]["recall"], 4),
            "f1_primary": round(t5["tuned_rf_test_oof_optimal"]["f1"], 4),
            "f1_macro": round(t5["tuned_rf_test_oof_optimal"]["f1_macro"], 4),
            "roc_auc": round(t5["tuned_rf_test_oof_optimal"]["roc_auc"], 4),
            "pr_auc": round(t5["tuned_rf_test_oof_optimal"]["pr_auc"], 4),
        },
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(dirs["reports_dir"] / "model_comparison.csv", index=False)
    comparison_df.to_csv(dirs["tables_dir"] / "portfolio_model_comparison.csv", index=False)

    # 2. Generate Figure 16: Unified Portfolio Benchmark
    churn_curves = {
        "Logistic Regression": t3["tuned_logreg_test_oof_optimal"],
        "SVM (Linear)": t6["linear_svm"]["test_oof_optimal"],
        "SVM (RBF)": t6["rbf_svm"]["test_oof_optimal"],
        "Random Forest": t5["tuned_rf_test_oof_optimal"],
    }
    fig16 = plot_portfolio_benchmark(
        churn_curves=churn_curves,
        comparison_df=comparison_df,
        out_path=dirs["figures_dir"] / "fig16_portfolio_model_comparison_summary.png",
    )

    # 3. Requirements Traceability & Task-Completion Matrix
    matrix_rows = [
        {
            "requirement_id": "REQ-L1-T1-01",
            "official_level": "Level 1 (Basic)",
            "official_task": "Task 1: Data Preprocessing for Machine Learning",
            "requirement_description": "Inspect raw datasets, detect & handle missing data without leakage, encode categoricals, standardize numericals, and split train/test sets.",
            "implementation_path": "src/codveda_ml/preprocessing/pipelines.py; src/codveda_ml/preprocessing/task1_runner.py; scripts/run_preprocessing.py",
            "notebook_path": "notebooks/01_data_preprocessing.ipynb",
            "automated_test": "tests/test_data_loading.py; tests/test_preprocessing.py",
            "verified_evidence": "results/metrics/task1_preprocessing_metrics.json; reports/dataset_inventory.csv; results/figures/fig01_dataset_quality_and_missingness_audit.png; fig02_preprocessing_scaling_and_distributions.png",
            "documentation_location": "reports/technical_report.md (Sections 3–5); reports/data_dictionary.md",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-L1-T3-02",
            "official_level": "Level 1 (Basic)",
            "official_task": "Task 3: Implement K-Nearest Neighbors (KNN) Classifier",
            "requirement_description": "Train KNN on Iris with feature scaling inside pipeline, compare multiple K values & weights via 5-fold CV, evaluate accuracy, precision, recall, F1, and confusion matrix.",
            "implementation_path": "src/codveda_ml/models/knn.py; scripts/train_knn.py",
            "notebook_path": "notebooks/02_knn_iris.ipynb",
            "automated_test": "tests/test_models.py::test_knn_pipeline_and_metrics",
            "verified_evidence": "results/metrics/task2_knn_metrics.json; results/tables/task2_knn_cv_sweep.csv; results/figures/fig03_knn_k_curve_scaled_vs_unscaled.png; fig04_knn_iris_confusion_and_feature_space.png",
            "documentation_location": "reports/technical_report.md (Section 6.1 & 10.1)",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-L2-T1-03",
            "official_level": "Level 2 (Intermediate)",
            "official_task": "Task 1: Logistic Regression for Binary Classification",
            "requirement_description": "Preprocess churn-bigml-80/20 preserving official partition, remediate VIF collinearity, tune Logistic Regression, calibrate OOF threshold, compute odds ratios with 95% bootstrap CIs, plot ROC/PR/CM.",
            "implementation_path": "src/codveda_ml/models/logistic_regression.py; scripts/train_logistic_regression.py",
            "notebook_path": "notebooks/03_logistic_regression_churn.ipynb",
            "automated_test": "tests/test_models.py::test_logistic_regression_pipeline_and_odds_ratios",
            "verified_evidence": "results/metrics/task3_logistic_regression_metrics.json; results/tables/task3_logreg_odds_ratios_bootstrap.csv; results/figures/fig05..fig07",
            "documentation_location": "reports/technical_report.md (Section 6.2 & 10.2)",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-L2-T2-04",
            "official_level": "Level 2 (Intermediate)",
            "official_task": "Task 2: Decision Trees for Classification",
            "requirement_description": "Train Decision Tree on Iris, visualise tree structure, perform cost-complexity pruning & hyperparameter tuning on train split only, compare unrestricted vs pruned vs baseline.",
            "implementation_path": "src/codveda_ml/models/decision_tree.py; scripts/train_decision_tree.py",
            "notebook_path": "notebooks/04_decision_tree_iris.ipynb",
            "automated_test": "tests/test_models.py::test_decision_tree_pruning_reduces_complexity",
            "verified_evidence": "results/metrics/task4_decision_tree_metrics.json; results/tables/task4_decision_tree_ccp_path.csv; results/figures/fig08..fig10",
            "documentation_location": "reports/technical_report.md (Section 6.3 & 10.3)",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-L3-T1-05",
            "official_level": "Level 3 (Advanced)",
            "official_task": "Task 1: Build a Random Forest Classifier",
            "requirement_description": "Build Churn Random Forest pipeline, tune hyperparameters via 5-fold CV, compare MDI vs 30-repeat permutation importance, evaluate 10-seed stability, compare against Logistic Regression.",
            "implementation_path": "src/codveda_ml/models/random_forest.py; scripts/train_random_forest.py",
            "notebook_path": "notebooks/05_random_forest_churn.ipynb",
            "automated_test": "tests/test_models.py::test_random_forest_pipeline_and_importance",
            "verified_evidence": "results/metrics/task5_random_forest_metrics.json; results/tables/task5_rf_feature_importance_comparison.csv; results/figures/fig11..fig13",
            "documentation_location": "reports/technical_report.md (Section 6.4 & 10.4)",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-L3-T2-06",
            "official_level": "Level 3 (Advanced)",
            "official_task": "Task 2: Support Vector Machine (SVM) for Classification",
            "requirement_description": "Scale features inside pipeline, compare Linear vs RBF kernels via 5-fold CV on binary Churn classification, evaluate accuracy/precision/recall/F1/AUC, visualise 2D decision boundaries & explain projection limits.",
            "implementation_path": "src/codveda_ml/models/svm.py; scripts/train_svm.py",
            "notebook_path": "notebooks/06_svm_classification.ipynb",
            "automated_test": "tests/test_models.py::test_svm_linear_vs_rbf_pipeline",
            "verified_evidence": "results/metrics/task6_svm_metrics.json; results/tables/task6_svm_kernel_comparison.csv; results/figures/fig14..fig15",
            "documentation_location": "reports/technical_report.md (Section 6.5 & 10.5)",
            "status": "Complete and verified",
        },
        {
            "requirement_id": "REQ-QA-07",
            "official_level": "Cross-Cutting Engineering",
            "official_task": "Automated Testing, Reproducibility & Interactive Portfolio",
            "requirement_description": "End-to-end pytest suite (anti-leakage, schema, metric ordering, serialisation), executed Jupyter notebooks, persisted datasets & pipelines, Google Drive organisation, and submission kit.",
            "implementation_path": "tests/; scripts/generate_report.py; models/",
            "notebook_path": "notebooks/01..06 (all executed via nbconvert)",
            "automated_test": "tests/test_evaluation.py; full pytest suite",
            "verified_evidence": "reports/model_comparison.csv; results/figures/fig16_portfolio_model_comparison_summary.png",
            "documentation_location": "README.md; docs/REPRODUCIBILITY.md; docs/SUBMISSION_CHECKLIST.md",
            "status": "Complete and verified",
        },
    ]
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(dirs["reports_dir"] / "task_completion_matrix.csv", index=False)

    portfolio_meta = {
        "n_models_compared": len(comparison_df),
        "n_requirements_verified": len(matrix_df),
        "summary_figure": str(fig16.relative_to(dirs["root"])),
        "task1_stock_rows": t1["stock_pipeline"]["raw_rows"],
    }
    save_json(portfolio_meta, m_dir / "portfolio_summary.json")
    return {"comparison_df": comparison_df, "matrix_df": matrix_df, "figure": str(fig16)}
