#!/usr/bin/env python3
"""Execute Level 2 Task 2 (Portfolio Task 4): Decision Trees & Pruning on Iris."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.models.decision_tree import run_task4_decision_tree  # noqa: E402


def main() -> None:
    res = run_task4_decision_tree(ROOT)
    pr = res["selected_pruned_tree"]
    te = pr["test_metrics"]
    print(
        f"[Task 4] Decision Tree complete | Unrestricted leaves={res['unrestricted_tree']['n_leaves']} -> "
        f"Pruned leaves={pr['fitted_n_leaves']} (depth={pr['fitted_depth']}) | "
        f"Test Acc={te['accuracy']:.4f} | Test Macro-F1={te['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
