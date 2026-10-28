# Methodological, Dataset, and Operational Limitations

This document records verified limitations of the datasets, experimental setup, and fitted models so that no finding in this portfolio is overstated.

## 1. Finite-Sample Uncertainty on Iris (`1) iris.csv`)
- With $N = 150$ total flowers and a stratified 20% test partition ($n_{\text{test}} = 30$), a single flower corresponds to $3.33\%$ accuracy. Under exact Clopper–Pearson 95% binomial intervals, $28/30$ ($93.3\%$, `[77.9%, 99.2%]`), $29/30$ ($96.7\%$, `[82.8%, 99.9%]`), and $30/30$ ($100.0\%$, `[88.4%, 100.0%]`) overlap substantially. Model selection therefore relies strictly on 5-fold cross-validation over the 120 training observations rather than post-hoc test accuracy comparisons.
- Three exact duplicate rows exist in the supplied CSV (matching the standard UCI archive). Sensitivity testing on the 147 unique rows confirmed CV Macro-F1 shifts by less than $0.007$ (`0.9665` $\rightarrow$ `0.9596`).

## 2. Observational Scope & Non-Causality on Customer Churn (`churn-bigml-80/20.csv`)
- Modelled associations (such as $\text{OR} = 7.28$ for `International plan = Yes` and $\text{OR} = 2.05$ per SD of `Customer service calls`) capture **observational correlations**, not causal interventions. High service-call counts act as a symptom of unresolved customer friction rather than an independent cause of churn.
- The dataset contains no time-stamped event history, contract expiration dates, competitive pricing shocks, or customer revenue figures. Consequently, threshold optimisation using a $5:1$ false-negative-to-false-positive cost ratio illustrates principled decision theory rather than company-specific financial accounting.

## 3. High-Cardinality Sparse Features in Kernel Methods
- Retaining 51 one-hot `State` indicators inside the shared Churn preprocessor ensures identical feature inputs across Logistic Regression, Random Forest, and SVM, but injects 51 sparse binary coordinates with near-zero predictive signal ($\Delta\text{AUC}_{\text{perm}} = 0.00024$). While Random Forest naturally ignores uninformative dummy columns during split selection, isotropic RBF kernels include all coordinates in $\|x - x'\|^2$, attenuating local neighbourhood similarity relative to tree ensembles.

## 4. Regime Shift & Market Microstructure in Stock Prices (`2) Stock Prices Data Set.csv`)
- Daily OHLCV prices between 2014–2016 and 2017 exhibit non-stationary drift (`+0.18 SD` upward shift in standardized `close`).Task 1 uses Stock Prices strictly to demonstrate leak-free chronological partitioning, skewed missing-data imputation, and unseen-category encoding (`APTV`, `BHF`, `BHGE`, `DWDP`, `DXC`, `HLT`), not as a claim of tradable market alpha (which would additionally require corporate-action split adjustments, transaction costs, and bid–ask spread modelling).

## 5. Local Interactive Dashboard Scope
- Persisted `.joblib` pipelines (`models/`) are version-pinned (`scikit-learn 1.6.1`, `Python 3.13.16`) for reproducible local evaluation; production deployment would require scheduled probability-calibration checks and population-stability monitoring.
