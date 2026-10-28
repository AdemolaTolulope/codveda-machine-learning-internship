"""Automated tests verifying missing-data imputation, categorical encoding, scaling, and anti-leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codveda_ml.config import get_project_root
from codveda_ml.data.loaders import (
    CHURN_NUMERIC_COLS_ALL,
    CHURN_NUMERIC_COLS_PRUNED,
    load_churn_splits,
    prepare_stock_supervised_splits,
)
from codveda_ml.preprocessing.pipelines import (
    build_churn_preprocessor,
    build_stock_preprocessor,
    compute_vif_table,
)

ROOT = get_project_root()


def test_transformations_fitted_strictly_on_training_data() -> None:
    churn = load_churn_splits(ROOT)
    pipe = build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)
    X_tr = pipe.fit_transform(churn["X_train"], churn["y_train"])
    X_te = pipe.transform(churn["X_test"])

    n_num = len(CHURN_NUMERIC_COLS_PRUNED)
    # Training numeric columns must have exact zero mean and unit population SD
    tr_means = X_tr[:, :n_num].mean(axis=0)
    tr_stds = X_tr[:, :n_num].std(axis=0)
    assert np.allclose(tr_means, 0.0, atol=1e-7)
    assert np.allclose(tr_stds, 1.0, atol=1e-7)

    # Test numeric columns transformed using train parameters must NOT be re-centered to exact 0
    te_means = X_te[:, :n_num].mean(axis=0)
    assert not np.allclose(te_means, 0.0, atol=1e-4)


def test_missing_value_imputation_and_unseen_categories_handled_safely() -> None:
    churn = load_churn_splits(ROOT)
    pipe = build_churn_preprocessor(drop_redundant_charges=True, scale_numeric=True)
    pipe.fit(churn["X_train"], churn["y_train"])

    # Corrupt a copy of test rows with NaNs and an unseen State / Area code
    corrupted = churn["X_test"].iloc[:5].copy()
    corrupted.loc[corrupted.index[0], "Total day minutes"] = np.nan
    corrupted.loc[corrupted.index[1], "International plan"] = np.nan
    corrupted.loc[corrupted.index[2], "State"] = "PR"  # Unseen territory code
    corrupted.loc[corrupted.index[3], "Area code"] = 999  # Unseen nominal area code

    tx = pipe.transform(corrupted)
    assert tx.shape == (5, 67)
    assert int(np.isnan(tx).sum()) == 0


def test_stock_pipeline_imputes_genuine_nans_without_leakage() -> None:
    stock = prepare_stock_supervised_splits(ROOT, train_end_date="2016-12-31")
    assert int(stock["X_train"][["open", "high", "low"]].isna().sum().sum()) > 0
    assert int(stock["X_test"][["open", "high", "low"]].isna().sum().sum()) > 0

    pipe = build_stock_preprocessor(imputer_strategy="median", scaler_type="standard", add_missing_indicator=True)
    X_tr = pipe.fit_transform(stock["X_train"], stock["y_train"])
    X_te = pipe.transform(stock["X_test"])

    assert int(np.isnan(X_tr.data).sum()) == 0
    assert int(np.isnan(X_te.data).sum()) == 0
    assert X_tr.shape[1] == X_te.shape[1]


def test_vif_collinearity_remediation_on_churn() -> None:
    churn = load_churn_splits(ROOT)
    vif_before = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_ALL])
    vif_after = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_PRUNED])
    assert vif_before["vif"].max() > 1_000_000
    assert vif_after["vif"].max() < 1.10
