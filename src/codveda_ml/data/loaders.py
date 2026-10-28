"""Verified dataset loaders, splitters, and comprehensive quality auditors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from codveda_ml.config import compute_sha256, get_project_root, load_config

IRIS_FEATURE_COLS: list[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

BOSTON_COLUMNS: list[str] = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

CHURN_CATEGORICAL_COLS: list[str] = [
    "State",
    "Area code",
    "International plan",
    "Voice mail plan",
]

CHURN_REDUNDANT_CHARGE_COLS: list[str] = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

CHURN_NUMERIC_COLS_ALL: list[str] = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

CHURN_NUMERIC_COLS_PRUNED: list[str] = [
    c for c in CHURN_NUMERIC_COLS_ALL if c not in CHURN_REDUNDANT_CHARGE_COLS
]


def _raw_dir(root: Path | None = None) -> Path:
    project_root = root or get_project_root()
    cfg = load_config(project_root)
    return project_root / cfg["paths"]["raw_dir"]


def load_iris_raw(root: Path | None = None) -> pd.DataFrame:
    """Load `1) iris.csv` and validate schema."""
    path = _raw_dir(root) / "1) iris.csv"
    if not path.exists():
        raise FileNotFoundError(f"Iris dataset not found at {path}")
    df = pd.read_csv(path)
    expected_cols = IRIS_FEATURE_COLS + ["species"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Unexpected Iris columns: {list(df.columns)}")
    return df


def get_iris_splits(
    root: Path | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return deterministic stratified 80/20 train/test split for Iris."""
    df = load_iris_raw(root)
    class_names = sorted(df["species"].unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    X = df[IRIS_FEATURE_COLS].copy()
    y = df["species"].map(label_to_idx).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return {
        "df_raw": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
        "label_to_idx": label_to_idx,
    }


def load_churn_splits(root: Path | None = None) -> dict[str, Any]:
    """
    Load the official BigML 80/20 customer churn partitions (`churn-bigml-80.csv`
    and `churn-bigml-20.csv`) while preserving their exact separation and verifying
    zero row overlap.
    """
    raw = _raw_dir(root)
    train_path = raw / "Churn Prdiction Data" / "churn-bigml-80.csv"
    test_path = raw / "Churn Prdiction Data" / "churn-bigml-20.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Churn files missing: {train_path} or {test_path}"
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if list(df_train.columns) != list(df_test.columns):
        raise ValueError("Train and test churn schemas do not match.")

    overlap = pd.merge(df_train, df_test, how="inner")
    if len(overlap) != 0:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping rows between churn splits."
        )

    feature_cols = [c for c in df_train.columns if c != "Churn"]
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train["Churn"].astype(int).copy()
    y_test = df_test["Churn"].astype(int).copy()

    return {
        "df_train": df_train,
        "df_test": df_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "train_path": train_path,
        "test_path": test_path,
    }


def load_stock_prices_raw(root: Path | None = None) -> pd.DataFrame:
    """Load `2) Stock Prices Data Set.csv` (497,472 daily S&P 500 OHLCV records)."""
    path = _raw_dir(root) / "2) Stock Prices Data Set.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stock dataset not found at {path}")
    return pd.read_csv(path)


def prepare_stock_supervised_splits(
    root: Path | None = None,
    train_end_date: str = "2016-12-31",
) -> dict[str, Any]:
    """
    Construct a chronologically partitioned, leak-free supervised task from the
    raw Stock Prices dataset:
      - Target `target_next_day_up`: 1 if `close_{t+1} > close_t` for the same symbol, else 0.
      - Terminal trading date per symbol (where `close_{t+1}` is unobserved) is excluded.
      - Genuine missing values in `open`, `high`, and `low` are preserved untouched
        so downstream scikit-learn pipelines impute them strictly from training data.
      - Chronological split: `date <= train_end_date` (2014-2016) -> Train;
        `date > train_end_date` (2017) -> Test.
    """
    df = load_stock_prices_raw(root)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    dt = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["month"] = dt.dt.month.astype(str)
    df["day_of_week"] = dt.dt.dayofweek.astype(str)
    df["zero_volume_flag"] = (df["volume"] == 0).astype(int)

    df["next_close"] = df.groupby("symbol", sort=False)["close"].shift(-1)
    supervised = df.dropna(subset=["next_close"]).copy()
    supervised["target_next_day_up"] = (
        supervised["next_close"] > supervised["close"]
    ).astype(int)

    feature_cols = [
        "symbol",
        "month",
        "day_of_week",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "zero_volume_flag",
    ]

    train_mask = supervised["date"] <= train_end_date
    test_mask = supervised["date"] > train_end_date

    train_df = supervised.loc[train_mask].reset_index(drop=True)
    test_df = supervised.loc[test_mask].reset_index(drop=True)

    if train_df["date"].max() >= test_df["date"].min():
        raise AssertionError("Temporal leakage detected in stock train/test split.")

    return {
        "df_raw": df.drop(columns=["next_close"]),
        "X_train": train_df[feature_cols].copy(),
        "X_test": test_df[feature_cols].copy(),
        "y_train": train_df["target_next_day_up"].copy(),
        "y_test": test_df["target_next_day_up"].copy(),
        "train_dates": (str(train_df["date"].min()), str(train_df["date"].max())),
        "test_dates": (str(test_df["date"].min()), str(test_df["date"].max())),
        "feature_cols": feature_cols,
    }


def load_sentiment_raw(root: Path | None = None) -> pd.DataFrame:
    """Load `3) Sentiment dataset.csv` exactly as stored on disk."""
    path = _raw_dir(root) / "3) Sentiment dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sentiment dataset not found at {path}")
    return pd.read_csv(path)


def clean_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant index columns and strip whitespace padding from string fields."""
    cleaned = df.copy()
    drop_cols = [c for c in cleaned.columns if c.startswith("Unnamed") or c == ""]
    cleaned = cleaned.drop(columns=drop_cols)
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
    return cleaned


def load_house_prices_raw(root: Path | None = None) -> pd.DataFrame:
    """Load whitespace-delimited, headerless `4) house Prediction Data Set.csv`."""
    path = _raw_dir(root) / "4) house Prediction Data Set.csv"
    if not path.exists():
        raise FileNotFoundError(f"House prediction dataset not found at {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=BOSTON_COLUMNS)


def audit_all_datasets(root: Path | None = None) -> pd.DataFrame:
    """
    Inspect all 6 supplied datasets and return a structured inventory DataFrame
    covering path, byte size, SHA-256 hash, dimensions, missingness, duplicates,
    target specification, and documented data-quality findings.
    """
    raw = _raw_dir(root)
    specs = [
        (
            "1) iris.csv",
            raw / "1) iris.csv",
            load_iris_raw(root),
            "species (3 classes: 50 setosa, 50 versicolor, 50 virginica)",
            "Task 2 (KNN), Task 4 (Decision Tree)",
            "0 missing values; 3 exact duplicate rows (2 in setosa, 1 in virginica); petal features have ~16.5x larger variance than sepal_width.",
        ),
        (
            "2) Stock Prices Data Set.csv",
            raw / "2) Stock Prices Data Set.csv",
            load_stock_prices_raw(root),
            "close / next-day directional movement (derived chronologically)",
            "Task 1 (Data Preprocessing — genuine raw NaNs & temporal anti-leakage)",
            "27 missing cells across 11 rows (open: 11, high: 8, low: 8); 4 rows with volume=0; heavy right-skew in price/volume; 505 tickers across 2014-2017.",
        ),
        (
            "3) Sentiment dataset.csv",
            raw / "3) Sentiment dataset.csv",
            load_sentiment_raw(root),
            "Sentiment (279 raw strings -> 191 after whitespace strip)",
            "Audited in Task 1 (not selected for primary supervised tasks)",
            "Two redundant index columns (Unnamed: 0.1, Unnamed: 0); trailing whitespace splits Platform ('Twitter' into 128+115) and inflates Sentiment cardinality to 191 fine-grained emotion tags across 732 rows.",
        ),
        (
            "4) house Prediction Data Set.csv",
            raw / "4) house Prediction Data Set.csv",
            load_house_prices_raw(root),
            "MEDV (continuous median home value in $1,000s, right-censored at 50.0)",
            "Audited in Task 1 (corresponds to unselected L1 Task 2 Linear Regression)",
            "Stored whitespace-delimited without header row (506x14); 0 missing values; 16 observations right-censored at MEDV=50.0; legacy B variable present.",
        ),
        (
            "Churn Prdiction Data/churn-bigml-80.csv",
            raw / "Churn Prdiction Data" / "churn-bigml-80.csv",
            pd.read_csv(raw / "Churn Prdiction Data" / "churn-bigml-80.csv"),
            "Churn (binary: 2,278 False [85.45%], 388 True [14.55%])",
            "Task 1, Task 3 (Logistic Regression), Task 5 (Random Forest), Task 6 (SVM) — Official Train Split",
            "0 missing values, 0 duplicates; moderately imbalanced (14.55% positive); Area code stored as int64 despite 3 nominal codes (408/415/510); exact linear collinearity (r>=0.999993) between all 4 *charge and *minutes pairs.",
        ),
        (
            "Churn Prdiction Data/churn-bigml-20.csv",
            raw / "Churn Prdiction Data" / "churn-bigml-20.csv",
            pd.read_csv(raw / "Churn Prdiction Data" / "churn-bigml-20.csv"),
            "Churn (binary: 572 False [85.76%], 95 True [14.24%])",
            "Task 1, Task 3 (Logistic Regression), Task 5 (Random Forest), Task 6 (SVM) — Official Held-Out Test Split",
            "0 missing values, 0 duplicates, 0 row overlap with churn-bigml-80.csv; identical 51 states and 14.24% churn prevalence; preserved strictly for final held-out evaluation.",
        ),
    ]

    records: list[dict[str, Any]] = []
    for rel_name, path, df, target_desc, task_usage, quality_notes in specs:
        records.append(
            {
                "dataset_file": rel_name,
                "size_bytes": path.stat().st_size,
                "sha256": compute_sha256(path),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "total_missing_cells": int(df.isna().sum().sum()),
                "rows_with_missing": int(df.isna().any(axis=1).sum()),
                "exact_duplicate_rows": int(df.duplicated().sum()),
                "target_specification": target_desc,
                "assigned_tasks": task_usage,
                "quality_and_leakage_notes": quality_notes,
            }
        )
    return pd.DataFrame(records)
