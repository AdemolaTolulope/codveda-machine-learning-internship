"""Reusable, leakage-free scikit-learn preprocessing pipelines and VIF diagnostics."""

from .pipelines import (
    ChurnSchemaNormalizer,
    build_churn_preprocessor,
    build_iris_preprocessor,
    build_stock_preprocessor,
    compute_vif_table,
)

__all__ = [
    "ChurnSchemaNormalizer",
    "build_churn_preprocessor",
    "build_iris_preprocessor",
    "build_stock_preprocessor",
    "compute_vif_table",
]
