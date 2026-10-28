"""Scikit-learn preprocessing pipelines with strict training-only parameter fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from codveda_ml.data.loaders import (
    CHURN_CATEGORICAL_COLS,
    CHURN_NUMERIC_COLS_ALL,
    CHURN_NUMERIC_COLS_PRUNED,
    CHURN_REDUNDANT_CHARGE_COLS,
    IRIS_FEATURE_COLS,
)


class ChurnSchemaNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize raw Churn DataFrames inside a scikit-learn Pipeline:
      - Casts `Area code` (408/415/510) to string so nominal area codes are never
        treated as continuous quantities.
      - Strips whitespace on string columns.
      - Optionally drops the 4 deterministically collinear `* charge` columns.
    """

    def __init__(self, drop_redundant_charges: bool = True) -> None:
        self.drop_redundant_charges = drop_redundant_charges

    def fit(self, X: pd.DataFrame, y: object = None) -> "ChurnSchemaNormalizer":
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in ["Area code", "State", "International plan", "Voice mail plan"]:
            if col in df.columns:
                na_mask = df[col].isna()
                # Convert non-null values cleanly (e.g., Area code 415.0 or 415 -> "415")
                if col == "Area code":
                    cleaned = df[col].map(
                        lambda v: np.nan
                        if pd.isna(v)
                        else (str(int(float(v))) if str(v).replace(".", "", 1).isdigit() else str(v).strip())
                    )
                else:
                    cleaned = df[col].astype(object).where(~na_mask, np.nan)
                    cleaned = cleaned.map(lambda v: np.nan if pd.isna(v) else str(v).strip())
                cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
                df[col] = cleaned
        if self.drop_redundant_charges:
            drop_cols = [c for c in CHURN_REDUNDANT_CHARGE_COLS if c in df.columns]
            df = df.drop(columns=drop_cols)
        return df


def build_iris_preprocessor(scale: bool = True) -> Pipeline:
    """Build preprocessing pipeline for Iris numerical features."""
    steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_churn_preprocessor(
    drop_redundant_charges: bool = True,
    scale_numeric: bool = True,
    include_state: bool = True,
    drop_first_binary: bool = True,
) -> Pipeline:
    """
    Build a full leakage-free preprocessing pipeline for Customer Churn.
    - Numerical features: median imputation + optional StandardScaler.
    - Binary categorical features (`International plan`, `Voice mail plan`):
      most-frequent imputation + single-column binary indicator (`drop='if_binary'`).
    - Multi-category nominal features (`Area code`, optionally `State`):
      most-frequent imputation + OneHotEncoder (`handle_unknown='ignore'`).
    """
    num_cols = (
        list(CHURN_NUMERIC_COLS_PRUNED)
        if drop_redundant_charges
        else list(CHURN_NUMERIC_COLS_ALL)
    )
    binary_cat_cols = ["International plan", "Voice mail plan"]
    nominal_cat_cols = ["Area code"] + (["State"] if include_state else [])

    num_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    bin_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary" if drop_first_binary else None,
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )

    nom_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("bin", bin_pipe, binary_cat_cols),
            ("nom", nom_pipe, nominal_cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            (
                "schema",
                ChurnSchemaNormalizer(
                    drop_redundant_charges=drop_redundant_charges
                ),
            ),
            ("columns", col_transformer),
        ]
    )


def build_stock_preprocessor(
    imputer_strategy: str = "median",
    scaler_type: str = "standard",
    add_missing_indicator: bool = True,
    sparse_output: bool = True,
) -> ColumnTransformer:
    """
    Build a leakage-free preprocessing ColumnTransformer for the S&P 500 Stock Prices
    dataset (`2) Stock Prices Data Set.csv`):
      - Imputes genuine missing values in `open`, `high`, `low` using training-only
        statistics (+ optional explicit missingness indicator columns).
      - Scales continuous OHLCV features using `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.
      - One-hot encodes `symbol` (handling unseen 2017 IPO/constituent tickers via
        `handle_unknown='ignore'`), `month`, and `day_of_week`.
    """
    num_cols = ["open", "high", "low", "close", "volume"]
    passthrough_cols = ["zero_volume_flag"]
    cat_cols = ["symbol", "month", "day_of_week"]

    if scaler_type == "standard":
        scaler: BaseEstimator = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    num_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=imputer_strategy,
                    add_indicator=add_missing_indicator,
                ),
            ),
            ("scaler", scaler),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                    dtype=float,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("flags", "passthrough", passthrough_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3 if sparse_output else 0.0,
    )


def compute_vif_table(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF = 1 / (1 - R_j^2)) for each column in a
    numeric feature matrix to diagnose multicollinearity before linear/logit modelling.
    """
    X = df_numeric.astype(float).to_numpy()
    cols = list(df_numeric.columns)
    rows = []
    for idx, col in enumerate(cols):
        y_j = X[:, idx]
        X_others = np.delete(X, idx, axis=1)
        reg = LinearRegression().fit(X_others, y_j)
        r2 = float(reg.score(X_others, y_j))
        vif = float("inf") if r2 >= 0.999999999 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "r_squared_with_others": round(r2, 8), "vif": round(vif, 3)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
