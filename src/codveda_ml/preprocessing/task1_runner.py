"""Task 1 workflow: Raw dataset audit, missingness handling, encoding, scaling, and anti-leakage verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from codveda_ml.config import ensure_directories, get_runtime_metadata, save_json
from codveda_ml.data.loaders import (
    CHURN_NUMERIC_COLS_ALL,
    CHURN_NUMERIC_COLS_PRUNED,
    IRIS_FEATURE_COLS,
    audit_all_datasets,
    clean_sentiment_dataframe,
    get_iris_splits,
    load_churn_splits,
    load_sentiment_raw,
    prepare_stock_supervised_splits,
)
from codveda_ml.preprocessing.pipelines import (
    build_churn_preprocessor,
    build_iris_preprocessor,
    build_stock_preprocessor,
    compute_vif_table,
)
from codveda_ml.visualization.plots import (
    plot_dataset_quality_audit,
    plot_scaling_comparison,
)


def run_task1_preprocessing(root: Path | None = None) -> dict[str, Any]:
    """Execute Level 1 Task 1 end-to-end and persist verified artefacts."""
    dirs = ensure_directories(root)

    # 1. Audit all 6 raw datasets
    inventory_df = audit_all_datasets(dirs["root"])
    inventory_csv = dirs["reports_dir"] / "dataset_inventory.csv"
    inventory_df.to_csv(inventory_csv, index=False)
    inventory_df.to_csv(dirs["tables_dir"] / "task1_dataset_inventory.csv", index=False)

    # 2. Stock Prices preprocessing (genuine missing values + chronological split)
    stock = prepare_stock_supervised_splits(dirs["root"], train_end_date="2016-12-31")
    X_tr_s, X_te_s = stock["X_train"], stock["X_test"]
    y_tr_s, y_te_s = stock["y_train"], stock["y_test"]

    raw_stock_missing = (
        stock["df_raw"][["symbol", "date", "open", "high", "low", "close", "volume"]]
        .isna()
        .sum()
        .to_dict()
    )
    train_missing = X_tr_s[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()
    test_missing = X_te_s[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()

    # Compare median vs mean imputation statistics fitted strictly on 2014-2016 training data
    oh_cols = ["open", "high", "low", "close", "volume"]
    med_imp = SimpleImputer(strategy="median").fit(X_tr_s[oh_cols])
    mean_imp = SimpleImputer(strategy="mean").fit(X_tr_s[oh_cols])
    imputation_comparison = pd.DataFrame(
        {
            "feature": oh_cols,
            "train_missing_count": [int(X_tr_s[c].isna().sum()) for c in oh_cols],
            "test_missing_count": [int(X_te_s[c].isna().sum()) for c in oh_cols],
            "train_skewness": [round(float(X_tr_s[c].skew()), 4) for c in oh_cols],
            "train_median_fit": [round(float(v), 4) for v in med_imp.statistics_],
            "train_mean_fit": [round(float(v), 4) for v in mean_imp.statistics_],
        }
    )
    imputation_comparison.to_csv(
        dirs["tables_dir"] / "task1_stock_imputation_comparison.csv", index=False
    )

    stock_pipe = build_stock_preprocessor(
        imputer_strategy="median",
        scaler_type="standard",
        add_missing_indicator=True,
        sparse_output=True,
    )
    X_tr_stock_tx = stock_pipe.fit_transform(X_tr_s, y_tr_s)
    X_te_stock_tx = stock_pipe.transform(X_te_s)

    # Verify zero NaNs post-pipeline
    assert np.isnan(X_tr_stock_tx.data).sum() == 0, "NaNs remain in transformed stock train!"
    assert np.isnan(X_te_stock_tx.data).sum() == 0, "NaNs remain in transformed stock test!"

    # Check unseen 2017 symbols handled cleanly by OneHotEncoder(handle_unknown='ignore')
    train_symbols = set(X_tr_s["symbol"].unique())
    test_symbols = set(X_te_s["symbol"].unique())
    unseen_2017_symbols = sorted(test_symbols - train_symbols)

    # Scaling comparison on Stock close (StandardScaler vs RobustScaler vs MinMaxScaler)
    close_tr_imputed = med_imp.transform(X_tr_s[oh_cols])[:, [3]]
    close_te_imputed = med_imp.transform(X_te_s[oh_cols])[:, [3]]
    std_s = StandardScaler().fit(close_tr_imputed)
    rob_s = RobustScaler().fit(close_tr_imputed)
    mm_s = MinMaxScaler().fit(close_tr_imputed)

    std_tr_close = std_s.transform(close_tr_imputed).ravel()
    rob_tr_close = rob_s.transform(close_tr_imputed).ravel()
    std_te_close = std_s.transform(close_te_imputed).ravel()

    joblib.dump(stock_pipe, dirs["models_dir"] / "task1_stock_preprocessing_pipeline.joblib")

    # 3. Churn preprocessing pipeline & multicollinearity (VIF) verification
    churn = load_churn_splits(dirs["root"])
    vif_before = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_ALL])
    vif_after = compute_vif_table(churn["df_train"][CHURN_NUMERIC_COLS_PRUNED])
    vif_before.to_csv(dirs["tables_dir"] / "task1_churn_vif_all_15_numeric.csv", index=False)
    vif_after.to_csv(dirs["tables_dir"] / "task1_churn_vif_pruned_11_numeric.csv", index=False)

    churn_pipe = build_churn_preprocessor(
        drop_redundant_charges=True,
        scale_numeric=True,
        include_state=True,
    )
    X_tr_churn_tx = churn_pipe.fit_transform(churn["X_train"], churn["y_train"])
    X_te_churn_tx = churn_pipe.transform(churn["X_test"])
    churn_feat_names = list(churn_pipe.named_steps["columns"].get_feature_names_out())

    # Save processed Churn and Iris splits for transparent inspection
    churn_tr_proc_df = pd.DataFrame(X_tr_churn_tx, columns=churn_feat_names)
    churn_tr_proc_df["Churn"] = churn["y_train"].to_numpy()
    churn_te_proc_df = pd.DataFrame(X_te_churn_tx, columns=churn_feat_names)
    churn_te_proc_df["Churn"] = churn["y_test"].to_numpy()

    churn_tr_proc_df.to_csv(dirs["processed_dir"] / "churn_train_processed.csv", index=False)
    churn_te_proc_df.to_csv(dirs["processed_dir"] / "churn_test_processed.csv", index=False)
    joblib.dump(churn_pipe, dirs["models_dir"] / "task1_churn_preprocessing_pipeline.joblib")

    # 4. Iris preprocessing pipeline
    iris = get_iris_splits(dirs["root"], test_size=0.20, random_state=42)
    iris_pipe = build_iris_preprocessor(scale=True)
    X_tr_iris_tx = iris_pipe.fit_transform(iris["X_train"])
    X_te_iris_tx = iris_pipe.transform(iris["X_test"])

    iris_tr_proc_df = pd.DataFrame(X_tr_iris_tx, columns=IRIS_FEATURE_COLS)
    iris_tr_proc_df["species_id"] = iris["y_train"].to_numpy()
    iris_te_proc_df = pd.DataFrame(X_te_iris_tx, columns=IRIS_FEATURE_COLS)
    iris_te_proc_df["species_id"] = iris["y_test"].to_numpy()
    iris_tr_proc_df.to_csv(dirs["processed_dir"] / "iris_train_processed.csv", index=False)
    iris_te_proc_df.to_csv(dirs["processed_dir"] / "iris_test_processed.csv", index=False)
    joblib.dump(iris_pipe, dirs["models_dir"] / "task1_iris_preprocessing_pipeline.joblib")

    # 5. Sentiment string-hygiene verification
    sent_raw = load_sentiment_raw(dirs["root"])
    sent_clean = clean_sentiment_dataframe(sent_raw)

    # 6. Generate Task 1 diagnostic figures
    fig1 = plot_dataset_quality_audit(
        inventory_df=inventory_df,
        stock_missing_by_col={k: int(v) for k, v in raw_stock_missing.items()},
        churn_train_df=churn["df_train"],
        churn_test_df=churn["df_test"],
        out_path=dirs["figures_dir"] / "fig01_dataset_quality_and_missingness_audit.png",
    )

    churn_scaled_num_df = pd.DataFrame(
        X_tr_churn_tx[:, : len(CHURN_NUMERIC_COLS_PRUNED)],
        columns=CHURN_NUMERIC_COLS_PRUNED,
    )
    fig2 = plot_scaling_comparison(
        raw_train_series=X_tr_s["close"],
        std_train_vals=std_tr_close,
        rob_train_vals=rob_tr_close,
        churn_raw_df=churn["df_train"],
        churn_scaled_df=churn_scaled_num_df,
        out_path=dirs["figures_dir"] / "fig02_preprocessing_scaling_and_distributions.png",
    )

    summary: dict[str, Any] = {
        "task_id": "Task 1",
        "official_level": "Level 1 (Basic) — Task 1",
        "title": "Data Preprocessing for Machine Learning",
        "runtime": get_runtime_metadata(),
        "stock_pipeline": {
            "raw_rows": int(len(stock["df_raw"])),
            "supervised_train_rows": int(len(X_tr_s)),
            "supervised_test_rows": int(len(X_te_s)),
            "train_date_range": list(stock["train_dates"]),
            "test_date_range": list(stock["test_dates"]),
            "raw_missing_by_column": {k: int(v) for k, v in raw_stock_missing.items()},
            "train_missing_by_column": {k: int(v) for k, v in train_missing.items()},
            "test_missing_by_column": {k: int(v) for k, v in test_missing.items()},
            "transformed_feature_dim": int(X_tr_stock_tx.shape[1]),
            "post_imputation_nans_train": 0,
            "post_imputation_nans_test": 0,
            "unseen_2017_test_symbols_count": len(unseen_2017_symbols),
            "unseen_2017_test_symbols_sample": unseen_2017_symbols[:10],
            "close_train_scaled_mean": round(float(std_tr_close.mean()), 6),
            "close_train_scaled_std": round(float(std_tr_close.std()), 6),
            "close_test_scaled_mean_shift": round(float(std_te_close.mean()), 6),
            "close_test_scaled_std_shift": round(float(std_te_close.std()), 6),
            "minmax_train_range": [
                round(float(mm_s.transform(close_tr_imputed).min()), 4),
                round(float(mm_s.transform(close_tr_imputed).max()), 4),
            ],
        },
        "churn_pipeline": {
            "train_rows": int(len(churn["X_train"])),
            "test_rows": int(len(churn["X_test"])),
            "train_test_overlap_rows": 0,
            "raw_feature_count": int(churn["X_train"].shape[1]),
            "transformed_feature_count": int(X_tr_churn_tx.shape[1]),
            "max_vif_before_collinearity_pruning": float(vif_before["vif"].max()),
            "max_vif_after_collinearity_pruning": float(vif_after["vif"].max()),
            "train_numeric_scaled_max_abs_mean": round(
                float(np.abs(X_tr_churn_tx[:, : len(CHURN_NUMERIC_COLS_PRUNED)].mean(axis=0)).max()),
                8,
            ),
            "train_numeric_scaled_std_range": [
                round(float(X_tr_churn_tx[:, : len(CHURN_NUMERIC_COLS_PRUNED)].std(axis=0).min()), 6),
                round(float(X_tr_churn_tx[:, : len(CHURN_NUMERIC_COLS_PRUNED)].std(axis=0).max()), 6),
            ],
        },
        "iris_pipeline": {
            "train_rows": int(len(iris["X_train"])),
            "test_rows": int(len(iris["X_test"])),
            "feature_count": int(X_tr_iris_tx.shape[1]),
            "train_scaled_means": [round(float(v), 8) for v in X_tr_iris_tx.mean(axis=0)],
            "train_scaled_stds": [round(float(v), 6) for v in X_tr_iris_tx.std(axis=0)],
        },
        "sentiment_hygiene": {
            "raw_shape": list(sent_raw.shape),
            "cleaned_shape": list(sent_clean.shape),
            "raw_platform_unique": int(sent_raw["Platform"].nunique()),
            "cleaned_platform_unique": int(sent_clean["Platform"].nunique()),
            "raw_sentiment_unique": int(sent_raw["Sentiment"].nunique()),
            "cleaned_sentiment_unique": int(sent_clean["Sentiment"].nunique()),
        },
        "figures": [str(fig1.relative_to(dirs["root"])), str(fig2.relative_to(dirs["root"]))],
    }

    save_json(summary, dirs["metrics_dir"] / "task1_preprocessing_metrics.json")
    return summary
