"""Automated tests verifying persisted pipelines, reloadability, predictions, and clean training."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from codveda_ml.config import get_project_root
from codveda_ml.data.loaders import get_iris_splits, load_churn_splits
from codveda_ml.preprocessing.pipelines import build_iris_preprocessor

ROOT = get_project_root()


def test_saved_models_exist_reload_and_predict_valid_probabilities() -> None:
    models_dir = ROOT / "models"
    iris = get_iris_splits(ROOT)
    churn = load_churn_splits(ROOT)

    for iris_model_name in [
        "task2_knn_iris_pipeline.joblib",
        "task4_decision_tree_iris_pipeline.joblib",
    ]:
        m_path = models_dir / iris_model_name
        assert m_path.exists(), f"Missing {m_path}"
        pipe = joblib.load(m_path)
        preds = pipe.predict(iris["X_test"])
        proba = pipe.predict_proba(iris["X_test"])
        assert preds.shape == (30,)
        assert proba.shape == (30, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    for churn_model_name in [
        "task3_logistic_regression_churn_pipeline.joblib",
        "task5_random_forest_churn_pipeline.joblib",
        "task6_svm_churn_pipeline.joblib",
    ]:
        m_path = models_dir / churn_model_name
        assert m_path.exists(), f"Missing {m_path}"
        pipe = joblib.load(m_path)
        preds = pipe.predict(churn["X_test"].iloc[:25])
        proba = pipe.predict_proba(churn["X_test"].iloc[:25])
        assert preds.shape == (25,)
        assert proba.shape == (25, 2)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_clean_knn_training_from_scratch() -> None:
    iris = get_iris_splits(ROOT)
    pipe = Pipeline(
        [
            ("prep", build_iris_preprocessor(scale=True)),
            ("knn", KNeighborsClassifier(n_neighbors=3)),
        ]
    )
    pipe.fit(iris["X_train"], iris["y_train"])
    score = pipe.score(iris["X_test"], iris["y_test"])
    assert 0.85 <= score <= 1.0


def test_logistic_regression_odds_ratios_mathematically_consistent() -> None:
    odds_df = pd.read_csv(ROOT / "results/tables/task3_logreg_odds_ratios_bootstrap.csv")
    assert np.allclose(np.exp(odds_df["coef"]), odds_df["odds_ratio"], rtol=1e-6)
    assert (odds_df["or_ci_low"] <= odds_df["odds_ratio"] * 1.05).all()
    assert (odds_df["or_ci_high"] >= odds_df["odds_ratio"] * 0.95).all()


def test_decision_tree_pruning_strictly_reduces_tree_complexity() -> None:
    m4 = json.loads((ROOT / "results/metrics/task4_decision_tree_metrics.json").read_text())
    assert m4["selected_pruned_tree"]["fitted_n_leaves"] < m4["unrestricted_tree"]["n_leaves"]
    assert m4["selected_pruned_tree"]["fitted_depth"] < m4["unrestricted_tree"]["depth"]
    assert (
        m4["selected_pruned_tree"]["test_metrics"]["f1_macro"]
        >= m4["unrestricted_tree"]["test_metrics"]["f1_macro"]
    )


def test_random_forest_and_rbf_svm_exceed_baselines() -> None:
    m5 = json.loads((ROOT / "results/metrics/task5_random_forest_metrics.json").read_text())
    m6 = json.loads((ROOT / "results/metrics/task6_svm_metrics.json").read_text())
    assert m5["tuned_rf_test_oof_optimal"]["roc_auc"] > 0.90
    assert m5["tuned_rf_test_oof_optimal"]["f1"] > 0.80
    assert m6["rbf_svm"]["test_oof_optimal"]["roc_auc"] > m6["linear_svm"]["test_oof_optimal"]["roc_auc"]
    assert m6["rbf_svm"]["test_oof_optimal"]["f1"] > m6["linear_svm"]["test_oof_optimal"]["f1"]
