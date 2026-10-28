# Reproducibility Specification

## 1. Verified Execution Environment

| Component | Verified Version |
| :--- | :--- |
| Execution Date | `2026-10-28` |
| Operating System | Linux (`x86_64`) |
| Python | `3.13.16` |
| `numpy` | `2.3.5` |
| `pandas` | `2.2.3` |
| `scikit-learn` | `1.6.1` |
| `scipy` | `1.17.1` |
| `matplotlib` | `3.10.9` |
| `seaborn` | `0.13.2` |
| `joblib` | `1.5.3` |
| `pytest` | `9.0.3` |
| `nbconvert` / `nbformat` | `7.17.2` / `5.11.1` |

## 2. Deterministic Seeds

- Primary global seed (`config/project_config.yaml`): `42`
  - Stratified Iris 80/20 split (`train_test_split(..., random_state=42, stratify=y)`)
  - All 5-fold cross-validators (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
  - Task 3 stratified bootstrap (`np.random.default_rng(42)`, `1,000` iterations)
  - Task 5 permutation importance (`n_repeats=30, random_state=42`)
- Task 5 multi-seed stability audit: `[11, 22, 33, 42, 55, 66, 77, 88, 99, 101]`

## 3. Verified Raw Dataset SHA-256 Checksums

```text
c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930  data/raw/1) iris.csv
2200ea7c7b98791314cb5b4a3072b4d4166da1ed78a3e3e178339199109837d2  data/raw/2) Stock Prices Data Set.csv
725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b  data/raw/3) Sentiment dataset.csv
baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a  data/raw/4) house Prediction Data Set.csv
b8bbd0099748ba3b0f337538cfa5997b5255bdc23c35144ffabba3cafb192550  data/raw/Churn Prdiction Data/churn-bigml-80.csv
c73bb8e7a47c8fced62ddef54741bbbc4005243fb40a65996360337844fd17f7  data/raw/Churn Prdiction Data/churn-bigml-20.csv
```

## 4. Clean-Room Verification Workflow

```bash
# Verify dataset checksums, preprocessing anti-leakage, model reloadability, and metric consistency
pytest -v

# Re-execute all 6 tasks end-to-end
python scripts/run_all.py

# Re-execute all 6 Jupyter notebooks in place
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
