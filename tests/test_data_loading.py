"""Automated tests for dataset loading, schema integrity, checksums, and train/test non-overlap."""

from __future__ import annotations

import pandas as pd

from codveda_ml.config import get_project_root, load_config
from codveda_ml.data.loaders import (
    BOSTON_COLUMNS,
    IRIS_FEATURE_COLS,
    audit_all_datasets,
    clean_sentiment_dataframe,
    get_iris_splits,
    load_churn_splits,
    load_house_prices_raw,
    load_iris_raw,
    load_sentiment_raw,
    load_stock_prices_raw,
    prepare_stock_supervised_splits,
)

ROOT = get_project_root()


def test_all_six_datasets_load_and_match_checksums() -> None:
    cfg = load_config(ROOT)
    inv = audit_all_datasets(ROOT)
    assert len(inv) == 6
    sha_map = dict(zip(inv["dataset_file"], inv["sha256"]))
    assert sha_map["1) iris.csv"] == cfg["datasets"]["iris"]["sha256"]
    assert sha_map["2) Stock Prices Data Set.csv"] == cfg["datasets"]["stock_prices"]["sha256"]
    assert sha_map["3) Sentiment dataset.csv"] == cfg["datasets"]["sentiment"]["sha256"]
    assert sha_map["4) house Prediction Data Set.csv"] == cfg["datasets"]["house_prices"]["sha256"]
    assert sha_map["Churn Prdiction Data/churn-bigml-80.csv"] == cfg["datasets"]["churn"]["train_sha256"]
    assert sha_map["Churn Prdiction Data/churn-bigml-20.csv"] == cfg["datasets"]["churn"]["test_sha256"]


def test_iris_dimensions_and_stratified_split() -> None:
    df = load_iris_raw(ROOT)
    assert df.shape == (150, 5)
    assert int(df.isna().sum().sum()) == 0
    splits = get_iris_splits(ROOT, test_size=0.20, random_state=42)
    assert splits["X_train"].shape == (120, 4)
    assert splits["X_test"].shape == (30, 4)
    assert list(splits["X_train"].columns) == IRIS_FEATURE_COLS
    # Exact stratified class balance: 40/40/40 train, 10/10/10 test
    assert splits["y_train"].value_counts().to_dict() == {0: 40, 1: 40, 2: 40}
    assert splits["y_test"].value_counts().to_dict() == {0: 10, 1: 10, 2: 10}
    # Index non-overlap
    assert set(splits["X_train"].index).isdisjoint(set(splits["X_test"].index))


def test_churn_official_splits_zero_overlap_and_no_target_in_features() -> None:
    churn = load_churn_splits(ROOT)
    assert churn["X_train"].shape == (2666, 19)
    assert churn["X_test"].shape == (667, 19)
    assert "Churn" not in churn["X_train"].columns
    assert "Churn" not in churn["X_test"].columns
    overlap = pd.merge(churn["df_train"], churn["df_test"], how="inner")
    assert len(overlap) == 0


def test_stock_chronological_split_has_zero_temporal_leakage() -> None:
    raw = load_stock_prices_raw(ROOT)
    assert raw.shape == (497472, 7)
    splits = prepare_stock_supervised_splits(ROOT, train_end_date="2016-12-31")
    assert splits["train_dates"][1] < splits["test_dates"][0]
    assert "next_close" not in splits["X_train"].columns
    assert "target_next_day_up" not in splits["X_train"].columns


def test_house_and_sentiment_special_parsing() -> None:
    house = load_house_prices_raw(ROOT)
    assert house.shape == (506, 14)
    assert list(house.columns) == BOSTON_COLUMNS

    sent_raw = load_sentiment_raw(ROOT)
    sent_clean = clean_sentiment_dataframe(sent_raw)
    assert sent_raw.shape == (732, 15)
    assert sent_clean.shape == (732, 13)
    assert sent_clean["Platform"].nunique() == 3
