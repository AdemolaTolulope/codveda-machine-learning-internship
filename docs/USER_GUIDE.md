# User Guide — Running, Inspecting & Verifying the Portfolio

## 1. Prerequisites

- Python `3.10+` (developed and verified under Python `3.13.16`).
- All official Codveda CSV datasets (`data/raw/`), processed train/test splits (`data/processed/`), and trained `scikit-learn` `.joblib` pipelines (`models/`) are included directly in this repository.

## 2. Environment Setup

```bash
cd codveda-machine-learning-internship
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Running Individual Tasks or Full Pipeline

Execute the complete six-task pipeline in one command:

```bash
python scripts/run_all.py
```

Or run any task independently:

```bash
python scripts/run_preprocessing.py          # Task 1 (Level 1 Basic)
python scripts/train_knn.py                  # Task 2 (Level 1 Basic)
python scripts/train_logistic_regression.py  # Task 3 (Level 2 Intermediate)
python scripts/train_decision_tree.py        # Task 4 (Level 2 Intermediate)
python scripts/train_random_forest.py        # Task 5 (Level 3 Advanced)
python scripts/train_svm.py                  # Task 6 (Level 3 Advanced)
python scripts/generate_report.py            # Regenerate model_comparison.csv & Figure 16
```

## 4. Running Automated Quality-Assurance Tests

```bash
pytest
```

All 18 tests verify dataset SHA-256 hashes, partition non-overlap, training-only transformation fitting, `NaN`/unseen-category resilience, model reloadability, probability simplex sums, confusion-matrix ordering, and report consistency.

## 5. Exploring Executed Jupyter Notebooks & Persisted Models

Open `notebooks/01_data_preprocessing.ipynb` through `notebooks/06_svm_classification.ipynb` in JupyterLab or VS Code:

```bash
jupyter lab notebooks/
```

To load any persisted trained pipeline directly in Python:

```python
import joblib
rf_pipeline = joblib.load("models/task5_random_forest_churn_pipeline.joblib")
```
