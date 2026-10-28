#!/usr/bin/env python3
"""Master end-to-end execution script running Tasks 1–6 and summary generation sequentially."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.decision_tree import run_task4_decision_tree  # noqa: E402
from codveda_ml.models.knn import run_task2_knn  # noqa: E402
from codveda_ml.models.logistic_regression import run_task3_logistic_regression  # noqa: E402
from codveda_ml.models.random_forest import run_task5_random_forest  # noqa: E402
from codveda_ml.models.svm import run_task6_svm  # noqa: E402
from codveda_ml.preprocessing.task1_runner import run_task1_preprocessing  # noqa: E402
from codveda_ml.reporting.generator import generate_portfolio_summaries  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    steps = [
        ("Task 1 — Data Preprocessing", run_task1_preprocessing),
        ("Task 2 — KNN Classifier (Iris)", run_task2_knn),
        ("Task 3 — Logistic Regression (Churn)", run_task3_logistic_regression),
        ("Task 4 — Decision Tree & Pruning (Iris)", run_task4_decision_tree),
        ("Task 5 — Random Forest & Importance (Churn)", run_task5_random_forest),
        ("Task 6 — SVM Linear vs RBF (Churn)", run_task6_svm),
        ("Portfolio Aggregation & Benchmark Plot", generate_portfolio_summaries),
    ]
    for label, fn in steps:
        s0 = time.perf_counter()
        print(f">>> Running: {label} ...")
        fn(ROOT)
        print(f"    Completed in {time.perf_counter() - s0:.2f}s")
    print(f"\nAll 6 tasks and summary artefacts verified in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
