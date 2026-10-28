#!/usr/bin/env python3
"""Execute Level 1 Task 1: Data Preprocessing for Machine Learning."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.preprocessing.task1_runner import run_task1_preprocessing  # noqa: E402


def main() -> None:
    res = run_task1_preprocessing(ROOT)
    print(
        f"[Task 1] Preprocessing complete | Stock rows={res['stock_pipeline']['raw_rows']:,} | "
        f"Churn VIF: {res['churn_pipeline']['max_vif_before_collinearity_pruning']:.1f} -> "
        f"{res['churn_pipeline']['max_vif_after_collinearity_pruning']:.3f}"
    )


if __name__ == "__main__":
    main()
