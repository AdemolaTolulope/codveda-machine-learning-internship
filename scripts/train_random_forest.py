#!/usr/bin/env python3
"""Execute Level 3 Task 1 (Portfolio Task 5): Random Forest Classifier on Customer Churn."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.random_forest import run_task5_random_forest  # noqa: E402


def main() -> None:
    res = run_task5_random_forest(ROOT)
    hp = res["best_hyperparameters"]
    te = res["tuned_rf_test_oof_optimal"]
    print(
        f"[Task 5] Random Forest complete | trees={hp['n_estimators']} max_depth={hp['max_depth']} "
        f"thr={hp['oof_selected_threshold']:.2f} | Test AUC={te['roc_auc']:.4f} | "
        f"Test PR-AUC={te['pr_auc']:.4f} | Test F1={te['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
