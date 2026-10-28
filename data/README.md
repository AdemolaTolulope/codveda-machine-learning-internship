# Data Directory Guide (`data/`)

Raw and processed dataset files are excluded from version control via `.gitignore` to keep the Git repository lightweight and avoid redistributing third-party datasets unnecessarily.

## Expected Directory Layout

Place the official Codveda Technologies Machine Learning Internship dataset files under `data/raw/` using the exact structure provided in the official task archive (`Data Set For Task/`):

```text
data/
├── raw/
│   ├── 1) iris.csv                                # 3,858 bytes | 150 rows x 5 cols
│   ├── 2) Stock Prices Data Set.csv               # 24,210,279 bytes | 497,472 rows x 7 cols
│   ├── 3) Sentiment dataset.csv                   # 170,776 bytes | 732 rows x 15 cols
│   ├── 4) house Prediction Data Set.csv           # 49,082 bytes | 506 rows x 14 cols (whitespace-delimited, headerless)
│   └── Churn Prdiction Data/
│       ├── churn-bigml-80.csv                     # 223,998 bytes | 2,666 rows x 20 cols (official 80% training partition)
│       └── churn-bigml-20.csv                     # 56,329 bytes | 667 rows x 20 cols (official 20% test partition)
└── processed/
    └── (Generated deterministically by `python scripts/run_preprocessing.py`)
```

## SHA-256 Checksums of Verified Raw Datasets

| File | Size (Bytes) | SHA-256 Checksum |
| :--- | ---: | :--- |
| `1) iris.csv` | 3,858 | `c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930` |
| `2) Stock Prices Data Set.csv` | 24,210,279 | `2200ea7c7b98791314cb5b4a3072b4d4166da1ed78a3e3e178339199109837d2` |
| `3) Sentiment dataset.csv` | 170,776 | `725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b` |
| `4) house Prediction Data Set.csv` | 49,082 | `baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a` |
| `Churn Prdiction Data/churn-bigml-80.csv` | 223,998 | `b8bbd0099748ba3b0f337538cfa5997b5255bdc23c35144ffabba3cafb192550` |
| `Churn Prdiction Data/churn-bigml-20.csv` | 56,329 | `c73bb8e7a47c8fced62ddef54741bbbc4005243fb40a65996360337844fd17f7` |

## Leakage-Prevention Guarantee

`data/raw/` is strictly **read-only** throughout all scripts, notebooks, and tests. Every imputation, encoding, scaling, and feature-selection step is fitted exclusively on training partitions via `scikit-learn` `Pipeline` and `ColumnTransformer` objects before transforming validation or test splits.
