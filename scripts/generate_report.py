#!/usr/bin/env python3
"""Aggregate verified metrics across Tasks 1–6 into summary tables and benchmark plots."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codveda_ml.reporting.generator import generate_portfolio_summaries  # noqa: E402


def main() -> None:
    out = generate_portfolio_summaries(ROOT)
    print(
        f"[Report] Portfolio tables & Figure 16 generated | "
        f"Models compared={len(out['comparison_df'])} | "
        f"Requirements mapped={len(out['matrix_df'])}"
    )


if __name__ == "__main__":
    main()
