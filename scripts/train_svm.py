#!/usr/bin/env python3
"""Execute Level 3 Task 2 (Portfolio Task 6): Support Vector Machine on Customer Churn."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.svm import run_task6_svm  # noqa: E402


def main() -> None:
    res = run_task6_svm(ROOT)
    lin_te = res["linear_svm"]["test_oof_optimal"]
    rbf_te = res["rbf_svm"]["test_oof_optimal"]
    print(
        f"[Task 6] SVM complete | Selected={res['selected_kernel'].upper()} | "
        f"Linear AUC={lin_te['roc_auc']:.4f} (F1={lin_te['f1']:.4f}) vs "
        f"RBF AUC={rbf_te['roc_auc']:.4f} (F1={rbf_te['f1']:.4f})"
    )


if __name__ == "__main__":
    main()
