# Technical Documentation — Architecture, API Contracts & QA Log

## 1. Package Architecture (`src/codveda_ml/`)

| Subpackage / Module | Responsibility |
| :--- | :--- |
| `codveda_ml.config` | Deterministic root resolution, YAML configuration parsing, SHA-256 file hashing, and runtime version capture. |
| `codveda_ml.data.loaders` | Validated dataset loaders (`load_iris_raw`, `load_churn_splits`, `load_stock_prices_raw`, `load_sentiment_raw`, `load_house_prices_raw`), chronological Stock target builder, and six-dataset inventory auditor. |
| `codveda_ml.preprocessing.pipelines` | `ChurnSchemaNormalizer` (`BaseEstimator, TransformerMixin`), `ColumnTransformer` builders for Iris, Churn, and Stock Prices, plus OLS Variance Inflation Factor (`compute_vif_table`) diagnostics. |
| `codveda_ml.evaluation.metrics` | Multiclass (`evaluate_multiclass_classifier`) and binary (`evaluate_binary_classifier`) evaluators with locked label ordering (`labels=[0, 1]`), plus leak-free out-of-fold threshold calibration (`select_optimal_threshold_oof`). |
| `codveda_ml.models.*` | Task-specific experimental runners (`knn.py`, `logistic_regression.py`, `decision_tree.py`, `random_forest.py`, `svm.py`). |
| `codveda_ml.visualization.plots` | Sixteen high-DPI Matplotlib/Seaborn visualisations with unified colorblind-aware styling. |
| `codveda_ml.reporting.generator` | Programmatic compilation of `reports/model_comparison.csv`, `reports/task_completion_matrix.csv`, and `fig16`. |

## 2. Anti-Leakage Design Contracts

1. **Zero Pre-Split Fitting:** No `SimpleImputer`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `OneHotEncoder`, or `PCA` instance is ever fitted on combined train+test data. Every transformation sits inside a `scikit-learn` `Pipeline` passed to `GridSearchCV` / `cross_validate` / `cross_val_predict`, guaranteeing fold-isolated estimation during cross-validation.
2. **Official Benchmark Preservation:** `churn-bigml-80.csv` ($2,666$ rows) and `churn-bigml-20.csv` ($667$ rows) are loaded separately and verified via inner join to share `0` overlapping rows.
3. **Chronological Time-Series Partitioning:** `2) Stock Prices Data Set.csv` is sorted within `symbol` by `date` (`mergesort`), next-day targets are computed strictly within `symbol`, and training (`<= 2016-12-31`) strictly precedes testing (`>= 2017-01-03`).
4. **Out-of-Fold Threshold Calibration:** Classification thresholds $\tau^*$ are chosen solely from 5-fold out-of-fold predictions on training data (`cross_val_predict`), never by scanning test-set F1 curves.

## 3. Quality-Assurance Defect & Resolution Log (Stage E Audit)

In accordance with transparent engineering practice, all issues encountered during implementation and testing were recorded and resolved:

| Defect ID | Stage Encountered | Symptom | Root Cause | Resolution & Verification |
| :--- | :--- | :--- | :--- | :--- |
| **QA-01** | Stage D (`train_decision_tree.py`) | `InvalidParameterError: The 'fontsize' parameter of plot_tree must be an int... Got 7.5` | `scikit-learn 1.6.1` enforces integer type validation on `plot_tree(..., fontsize=...)`. | Updated `fontsize` to `8` and `10` in `visualization/plots.py`; verified `fig08` renders cleanly. |
| **QA-02** | Stage E (`pytest` — `test_preprocessing.py`) | `ValueError: Found unknown categories ['nan'] in column 0 during transform` when synthetic `np.nan` injected into `International plan` | Calling `.astype(str)` inside `ChurnSchemaNormalizer.transform` converted `np.nan` into the literal string `'nan'` prior to `SimpleImputer`. | Refactored `ChurnSchemaNormalizer.transform` to preserve missingness masks (`pd.isna`) and map empty/`'nan'` strings back to `np.nan` before imputation; all 18 `pytest` checks passed. |
| **QA-03** | Stage E (Visual Figure Audit) | Label overlap in `fig16` Panel C and narrow conditional slice in `fig14` Panel C | Long model strings in `fig16` exceeded subplot margin; fixing 65 orthogonal PCs at median placed the 67D RBF slice inside low-risk interior. | Widened `wspace=0.42` with concise labels in `fig16` and trained an explicit 2D PCA RBF model in `fig14` Panel C to demonstrate 82.2% variance loss visually. |
