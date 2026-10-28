"""Automated tests verifying evaluation metrics, confusion matrix ordering, ROC-AUC, and report consistency."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from codveda_ml.config import get_project_root
from codveda_ml.evaluation.metrics import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    select_optimal_threshold_oof,
)

ROOT = get_project_root()


class _MockBinaryModel:
    def __init__(self, probs: np.ndarray) -> None:
        self._probs = probs

    def predict_proba(self, X: object) -> np.ndarray:
        return np.column_stack([1.0 - self._probs, self._probs])


class _MockMulticlassModel:
    def __init__(self, preds: np.ndarray) -> None:
        self._preds = preds

    def predict(self, X: object) -> np.ndarray:
        return self._preds


def test_binary_confusion_matrix_ordering_and_roc_auc() -> None:
    # Ground truth: 3 negatives (0), 2 positives (1)
    y_true = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
    mock = _MockBinaryModel(probs)

    ev = evaluate_binary_classifier(mock, np.zeros((5, 2)), y_true, threshold=0.50)
    # At thr=0.50: preds = [0, 0, 1, 0, 1] -> TN=2, FP=1, FN=1, TP=1
    assert ev["confusion_matrix"] == [[2, 1], [1, 1]]
    assert (ev["tn"], ev["fp"], ev["fn"], ev["tp"]) == (2, 1, 1, 1)
    assert np.isclose(ev["precision"], 0.5)
    assert np.isclose(ev["recall"], 0.5)
    # Pairs: (0.4 vs [0.1,0.2,0.6] -> 2/3) + (0.9 vs [0.1,0.2,0.6] -> 3/3) => 5/6
    assert np.isclose(ev["roc_auc"], 5.0 / 6.0)


def test_multiclass_confusion_matrix_ordering() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    mock = _MockMulticlassModel(y_pred)
    ev = evaluate_multiclass_classifier(mock, np.zeros((6, 2)), y_true, ["setosa", "versicolor", "virginica"])
    assert ev["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert np.isclose(ev["accuracy"], 4.0 / 6.0)


def test_oof_threshold_selector_operates_on_training_arrays_only() -> None:
    y_tr = np.array([0, 0, 0, 0, 1, 1])
    oof_p = np.array([0.05, 0.12, 0.18, 0.25, 0.38, 0.82])
    info = select_optimal_threshold_oof(y_tr, oof_p)
    assert 0.26 <= info["optimal_f1_threshold"] <= 0.38
    assert np.isclose(info["oof_f1_at_optimal"], 1.0)


def test_all_16_figures_and_summary_tables_exist_and_match_metrics() -> None:
    figs = sorted((ROOT / "results/figures").glob("fig*.png"))
    assert len(figs) == 16
    for f in figs:
        assert f.stat().st_size > 20_000, f"Figure suspiciously small: {f}"

    comp = pd.read_csv(ROOT / "reports/model_comparison.csv")
    assert len(comp) == 12

    m5 = json.loads((ROOT / "results/metrics/task5_random_forest_metrics.json").read_text())
    rf_row = comp[comp["model_name"] == "Random Forest (Tuned, OOF-Calibrated)"].iloc[0]
    assert np.isclose(rf_row["roc_auc"], round(m5["tuned_rf_test_oof_optimal"]["roc_auc"], 4))
    assert np.isclose(rf_row["f1_primary"], round(m5["tuned_rf_test_oof_optimal"]["f1"], 4))

    matrix = pd.read_csv(ROOT / "reports/task_completion_matrix.csv")
    assert len(matrix) == 7
    assert (matrix["status"] == "Complete and verified").all()
