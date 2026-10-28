#!/usr/bin/env python3
"""Execute Level 1 Task 3 (Portfolio Task 2): K-Nearest Neighbors Classifier on Iris."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.knn import run_task2_knn  # noqa: E402


def main() -> None:
    res = run_task2_knn(ROOT)
    hp = res["selected_hyperparameters"]
    te = res["final_scaled_knn_test"]
    print(
        f"[Task 2] KNN complete | Selected K={hp['n_neighbors']} ({hp['weights']}, scaled=True) | "
        f"CV Macro-F1={hp['cv_f1_macro_mean']:.4f} | Test Acc={te['accuracy']:.4f} | Test Macro-F1={te['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
