# Video Demonstration Script (4–5 Minutes)

**Status:** Script complete and aligned with verified codebase, datasets, notebooks, trained `.joblib` models, and generated figures. **No video recording has been fabricated — record your screen and voiceover using OBS Studio, Loom, or Zoom following the walkthrough below.**

---

## Pre-Recording Setup (1 Minute Before Recording)

Keep three browser/editor tabs ready:
- **Tab 1:** Your published GitHub repository (`codveda-machine-learning-internship`) showing `data/`, `models/`, `notebooks/`, `results/figures/`, and `reports/technical_report.md`.
- **Tab 2:** JupyterLab or VS Code showing the six executed notebooks (`notebooks/01_data_preprocessing.ipynb` .. `06_svm_classification.ipynb`) and generated plots (`results/figures/fig01..fig16.png`).
- **Tab 3:** Terminal inside `codveda-machine-learning-internship/` ready to run `pytest -v`.

---

## Time-Coded Walkthrough & Narration Guide

### [0:00 – 0:35] 1. Project Objective & Repository Overview
- **On Screen:** GitHub repository root (`README.md`, `data/`, `models/`, `results/`, `reports/`).
- **Narration:**
  > "Hello everyone. I'm presenting my Machine Learning Internship portfolio completed with Codveda Technologies. Following the official internship guidelines, I completed two tasks per tier across Level 1 Basic, Level 2 Intermediate, and Level 3 Advanced — six tasks in total. Everything — the raw and processed datasets, the modular `codveda_ml` Python package, six executed Jupyter notebooks, persisted `.joblib` pipelines, sixteen diagnostic visualisations, and an eighteen-test `pytest` suite — is organized reproducibly in this repository."

### [0:35 – 1:25] 2. Dataset Audit & Leak-Free Preprocessing (Task 1)
- **On Screen:** `notebooks/01_data_preprocessing.ipynb` (`fig01` Dataset Audit & VIF table).
- **Narration:**
  > "Before training any model, I audited all six supplied datasets. Two critical data-quality findings stood out. First, `Stock Prices Data Set.csv` — nearly half a million daily S&P 500 records — contained 27 genuine missing values in `open`, `high`, and `low`, plus six 2017 ticker symbols unseen during 2014–2016. I built a chronological scikit-learn `ColumnTransformer` fitting median imputation, missingness indicators, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')` strictly on the 2014–2016 training window. Second, in the BigML Customer Churn dataset, all four billing-charge columns were exact linear multiples of usage minutes, inflating Variance Inflation Factors above ten million. Dropping those redundant charge columns inside a custom pipeline transformer reduced maximum VIF to 1.004 while preserving the official 2,666-row train and 667-row test split."

### [1:25 – 2:10] 3. Iris Classification: KNN vs Pruned Decision Trees (Tasks 2 & 4)
- **On Screen:** `notebooks/02_knn_iris.ipynb` (`fig03`) & `notebooks/04_decision_tree_iris.ipynb` (`fig08`).
- **Narration:**
  > "On the Iris dataset using an identical stratified 80/20 split, Task 2 tuned KNN via 5-fold training cross-validation, selecting `K = 3` with `0.9665` CV Macro-F1 and `93.33%` test accuracy (`0.9933` OVR ROC-AUC). Comparing scaled versus unscaled KNN revealed why equalizing variance across homogeneous centimetre measurements slightly increases sensitivity to noisy `sepal_width`. In Task 4, an unrestricted Decision Tree grew to 8 leaves and depth 5, overfitting sepal noise. Applying minimal cost-complexity pruning (`ccp_alpha = 0.0081`) halved the tree to just 4 leaves using only petal length and petal width, raising held-out test accuracy to `96.67%`."

### [2:10 – 3:35] 4. Customer Churn Progression: Logistic Regression -> SVM -> Random Forest (Tasks 3, 6 & 5)
- **On Screen:** `notebooks/03`, `06`, and `05` (`fig06` Odds Ratios, `fig14` SVM 2D Boundaries, `fig11` RF MDI vs Permutation, `fig16` Unified ROC/PR summary).
- **Narration:**
  > "Tasks 3, 5, and 6 form a controlled progression on the imbalanced Customer Churn benchmark (`14.24%` test churn, where a majority-class dummy classifier gets `85.76%` accuracy yet `0.00` F1).
  > In Task 3, regularized Logistic Regression provided clear directional odds ratios with 1,000-draw bootstrap confidence intervals — customers on the International Plan had 7.28 times higher modelled churn odds, and each standard deviation of customer service calls doubled churn odds. However, both Logistic Regression and Task 6's Linear SVM plateaued around `0.826–0.827` ROC-AUC and `0.47–0.48` F1.
  > Task 6's 2D decision-boundary plot explains why: customers churn either at high daytime minutes OR when customer service calls reach 4 or more — a non-linear OR-gate interaction that no single straight hyperplane can separate. Switching to an RBF kernel SVM immediately raised held-out ROC-AUC to `0.9130` and F1 to `0.6970`.
  > Finally, in Task 5, a 300-tree Random Forest with out-of-fold threshold calibration (`0.44`) captured those sharp threshold interactions best, reaching `95.65%` accuracy, `90.24%` precision, `77.89%` recall, `0.8362` F1, `0.9258` ROC-AUC, and `0.8828` PR-AUC, with tight stability across 10 random seeds. Comparing Gini importance against 30-repeat permutation importance also proved that the 51 `State` dummy columns had inflated Gini credit (`4.0%`) despite zero generalisable permutation lift."

### [3:35 – 4:15] 5. Persisted Models, Automated `pytest` Verification & Wrap-Up
- **On Screen:** Terminal running `pytest -v` (`18 passed`) and `reports/model_comparison.csv`.
- **Narration:**
  > "Finally, running `pytest -v` executes 18 automated checks verifying SHA-256 dataset checksums, zero train/test leakage, persisted `.joblib` pipeline predictions, and metric consistency across all saved JSON and CSV reports. All datasets, trained models, executed notebooks, and documentation are available in my GitHub repository linked in the post. Thank you to Codveda Technologies for this learning experience!"
