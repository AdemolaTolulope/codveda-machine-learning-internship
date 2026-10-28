# Persisted Model Artefacts (`models/`)

Serialised `scikit-learn` pipelines (`*.joblib`) are written here deterministically by the training scripts in `scripts/` and excluded from Git commits via `.gitignore`.

To regenerate all persisted pipelines from a clean checkout (once `data/raw/` is populated):

```bash
python scripts/run_preprocessing.py
python scripts/train_knn.py
python scripts/train_logistic_regression.py
python scripts/train_decision_tree.py
python scripts/train_random_forest.py
python scripts/train_svm.py
```

Generated artefacts:
- `task1_stock_preprocessing_pipeline.joblib`
- `task1_churn_preprocessing_pipeline.joblib`
- `task1_iris_preprocessing_pipeline.joblib`
- `task2_knn_iris_pipeline.joblib`
- `task3_logistic_regression_churn_pipeline.joblib`
- `task4_decision_tree_iris_pipeline.joblib`
- `task5_random_forest_churn_pipeline.joblib`
- `task6_svm_churn_pipeline.joblib`
