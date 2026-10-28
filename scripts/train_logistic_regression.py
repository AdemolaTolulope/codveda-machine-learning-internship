#!/usr/bin/env python3
"""Execute Level 2 Task 1 (Portfolio Task 3): Logistic Regression on Customer Churn."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.logistic_regression import run_task3_logistic_regression  # noqa: E402


def main() -> None:
    res = run_task3_logistic_regression(ROOT)
    hp = res["best_hyperparameters"]
    te = res["tuned_logreg_test_oof_optimal"]
    print(
        f"[Task 3] Logistic Regression complete | C={hp['C']} class_weight={hp['class_weight']} "
        f"thr={hp['oof_selected_threshold']:.2f} | Test AUC={te['roc_auc']:.4f} | Test F1={te['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
