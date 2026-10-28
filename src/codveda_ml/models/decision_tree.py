"""Task 4 (Level 2 Intermediate): Decision Trees & Cost-Complexity Pruning on Iris."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from codveda_ml.config import ensure_directories, get_runtime_metadata, load_config, save_json
from codveda_ml.data.loaders import IRIS_FEATURE_COLS, get_iris_splits
from codveda_ml.evaluation.metrics import evaluate_multiclass_classifier
from codveda_ml.visualization.plots import (
    plot_ccp_pruning_curve,
    plot_decision_trees_comparison,
    plot_multiclass_confusion_and_scatter,
)


def run_task4_decision_tree(root: Path | None = None) -> dict[str, Any]:
    """Train unrestricted and pruned Decision Trees on Iris using training-only CV."""
    dirs = ensure_directories(root)
    cfg = load_config(dirs["root"])
    seed = int(cfg["project"]["random_seed"])

    iris = get_iris_splits(dirs["root"], test_size=0.20, random_state=seed)
    X_train, X_test = iris["X_train"], iris["X_test"]
    y_train, y_test = iris["y_train"], iris["y_test"]
    class_names = iris["class_names"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. Stratified Dummy Baseline
    dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(X_train, y_train)
    dummy_eval = evaluate_multiclass_classifier(dummy, X_test, y_test, class_names)

    # 2. Unrestricted Decision Tree (max_depth=None, ccp_alpha=0.0)
    unrestricted_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("tree", DecisionTreeClassifier(criterion="gini", max_depth=None, ccp_alpha=0.0, random_state=seed)),
        ]
    )
    unrestricted_pipe.fit(X_train, y_train)
    unres_cv = cross_validate(unrestricted_pipe, X_train, y_train, cv=cv, scoring="f1_macro", return_train_score=True)
    unres_test_eval = evaluate_multiclass_classifier(unrestricted_pipe, X_test, y_test, class_names)
    unres_clf: DecisionTreeClassifier = unrestricted_pipe.named_steps["tree"]

    # 3. Minimal Cost-Complexity Pruning Path on Training Data ONLY
    X_tr_imp = SimpleImputer(strategy="median").fit_transform(X_train)
    base_tree = DecisionTreeClassifier(criterion="gini", random_state=seed)
    path_info = base_tree.cost_complexity_pruning_path(X_tr_imp, y_train)
    raw_alphas = np.unique(np.round(path_info.ccp_alphas, 6))
    # Exclude trivial single-root collapse alpha at extreme upper end if it predicts 1 class
    ccp_rows: list[dict[str, Any]] = []
    for alpha in raw_alphas:
        pipe_a = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("tree", DecisionTreeClassifier(criterion="gini", ccp_alpha=float(alpha), random_state=seed)),
            ]
        )
        pipe_a.fit(X_train, y_train)
        t_clf = pipe_a.named_steps["tree"]
        if t_clf.get_n_leaves() < 3:
            continue
        cv_sc = cross_validate(pipe_a, X_train, y_train, cv=cv, scoring="f1_macro", return_train_score=True)
        train_pred = pipe_a.predict(X_train)
        ccp_rows.append(
            {
                "ccp_alpha": float(alpha),
                "depth": int(t_clf.get_depth()),
                "n_leaves": int(t_clf.get_n_leaves()),
                "train_f1_macro": float(f1_score(y_train, train_pred, average="macro")),
                "cv_f1_macro_mean": float(np.mean(cv_sc["test_score"])),
                "cv_f1_macro_std": float(np.std(cv_sc["test_score"], ddof=1)),
            }
        )
    ccp_df = pd.DataFrame(ccp_rows)
    ccp_df.to_csv(dirs["tables_dir"] / "task4_decision_tree_ccp_path.csv", index=False)

    # 4. Joint Hyperparameter & Pruning Grid Search on Training Set ONLY
    # Parsimony rule: among configurations within 0.005 of top CV Macro-F1, prefer fewer leaves & lower depth
    grid_rows: list[dict[str, Any]] = []
    candidate_alphas = sorted(set([0.0] + ccp_df["ccp_alpha"].tolist()))
    for crit in ["gini", "entropy"]:
        for mdepth in [2, 3, 4, 5, None]:
            for mleaf in [1, 2, 3, 5]:
                for alpha in candidate_alphas:
                    pipe_g = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "tree",
                                DecisionTreeClassifier(
                                    criterion=crit,
                                    max_depth=mdepth,
                                    min_samples_leaf=mleaf,
                                    ccp_alpha=float(alpha),
                                    random_state=seed,
                                ),
                            ),
                        ]
                    )
                    pipe_g.fit(X_train, y_train)
                    t_g = pipe_g.named_steps["tree"]
                    if t_g.get_n_leaves() < 3:
                        continue
                    sc = cross_validate(pipe_g, X_train, y_train, cv=cv, scoring={"acc": "accuracy", "f1": "f1_macro"})
                    grid_rows.append(
                        {
                            "criterion": crit,
                            "max_depth_param": -1 if mdepth is None else mdepth,
                            "min_samples_leaf": mleaf,
                            "ccp_alpha": float(alpha),
                            "fitted_depth": int(t_g.get_depth()),
                            "fitted_n_leaves": int(t_g.get_n_leaves()),
                            "cv_f1_macro_mean": float(np.mean(sc["test_f1"])),
                            "cv_f1_macro_std": float(np.std(sc["test_f1"], ddof=1)),
                            "cv_acc_mean": float(np.mean(sc["test_acc"])),
                        }
                    )

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(dirs["tables_dir"] / "task4_decision_tree_cv_grid.csv", index=False)

    peak_cv_f1 = float(grid_df["cv_f1_macro_mean"].max())
    near_optimal = grid_df[grid_df["cv_f1_macro_mean"] >= peak_cv_f1 - 0.005].copy()
    near_optimal = near_optimal.sort_values(
        by=["fitted_n_leaves", "fitted_depth", "cv_f1_macro_mean", "ccp_alpha"],
        ascending=[True, True, False, False],
    )
    selected = near_optimal.iloc[0]
    sel_depth = None if int(selected["max_depth_param"]) == -1 else int(selected["max_depth_param"])
    sel_alpha = float(selected["ccp_alpha"])
    sel_crit = str(selected["criterion"])
    sel_mleaf = int(selected["min_samples_leaf"])

    # Pick representative ccp_alpha from pure CCP path for Figure 9 annotation
    best_ccp_only = ccp_df[ccp_df["cv_f1_macro_mean"] >= ccp_df["cv_f1_macro_mean"].max() - 0.005].sort_values(
        ["n_leaves", "ccp_alpha"], ascending=[True, False]
    ).iloc[0]

    pruned_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "tree",
                DecisionTreeClassifier(
                    criterion=sel_crit,
                    max_depth=sel_depth,
                    min_samples_leaf=sel_mleaf,
                    ccp_alpha=sel_alpha,
                    random_state=seed,
                ),
            ),
        ]
    )
    pruned_pipe.fit(X_train, y_train)
    pruned_clf: DecisionTreeClassifier = pruned_pipe.named_steps["tree"]
    pruned_test_eval = evaluate_multiclass_classifier(pruned_pipe, X_test, y_test, class_names)
    tree_rules_text = export_text(pruned_clf, feature_names=IRIS_FEATURE_COLS)

    # Persist pipeline & predictions
    model_path = dirs["models_dir"] / "task4_decision_tree_iris_pipeline.joblib"
    joblib.dump(pruned_pipe, model_path)

    proba_test = pruned_pipe.predict_proba(X_test)
    preds_df = X_test.copy()
    preds_df["y_true_species"] = [class_names[i] for i in y_test]
    preds_df["y_pred_unrestricted"] = [class_names[i] for i in unrestricted_pipe.predict(X_test)]
    preds_df["y_pred_pruned"] = [class_names[i] for i in pruned_pipe.predict(X_test)]
    for idx, cname in enumerate(class_names):
        preds_df[f"prob_{cname}"] = np.round(proba_test[:, idx], 6)
    preds_df.to_csv(dirs["predictions_dir"] / "task4_decision_tree_iris_test_predictions.csv", index=False)

    # Figures
    fig8 = plot_decision_trees_comparison(
        unrestricted_clf=unres_clf,
        pruned_clf=pruned_clf,
        feature_names=IRIS_FEATURE_COLS,
        class_names=class_names,
        out_path=dirs["figures_dir"] / "fig08_decision_tree_unrestricted_vs_pruned.png",
    )
    fig9 = plot_ccp_pruning_curve(
        ccp_df=ccp_df,
        selected_alpha=float(best_ccp_only["ccp_alpha"]),
        out_path=dirs["figures_dir"] / "fig09_decision_tree_ccp_pruning_curve.png",
    )
    fig10 = plot_multiclass_confusion_and_scatter(
        cm=pruned_test_eval["confusion_matrix"],
        class_names=class_names,
        iris_df=iris["df_raw"],
        title_prefix=(
            f"Task 4 — Pruned Decision Tree (Depth = {pruned_clf.get_depth()}, "
            f"Leaves = {pruned_clf.get_n_leaves()}) on Held-Out Iris Test Split"
        ),
        subtitle_metrics=(
            f"Acc = {pruned_test_eval['accuracy']:.4f} | Macro-F1 = {pruned_test_eval['f1_macro']:.4f}"
        ),
        out_path=dirs["figures_dir"] / "fig10_decision_tree_confusion_matrix.png",
    )

    summary: dict[str, Any] = {
        "task_id": "Task 4",
        "official_level": "Level 2 (Intermediate) — Task 2",
        "title": "Decision Trees for Classification",
        "dataset": "1) iris.csv",
        "runtime": get_runtime_metadata(),
        "unrestricted_tree": {
            "depth": int(unres_clf.get_depth()),
            "n_leaves": int(unres_clf.get_n_leaves()),
            "train_f1_macro": round(float(np.mean(unres_cv["train_score"])), 6),
            "cv_f1_macro_mean": round(float(np.mean(unres_cv["test_score"])), 6),
            "cv_f1_macro_std": round(float(np.std(unres_cv["test_score"], ddof=1)), 6),
            "test_metrics": unres_test_eval,
        },
        "selected_pruned_tree": {
            "criterion": sel_crit,
            "max_depth": sel_depth,
            "min_samples_leaf": sel_mleaf,
            "ccp_alpha": sel_alpha,
            "ccp_path_optimal_alpha": float(best_ccp_only["ccp_alpha"]),
            "fitted_depth": int(pruned_clf.get_depth()),
            "fitted_n_leaves": int(pruned_clf.get_n_leaves()),
            "cv_f1_macro_mean": round(float(selected["cv_f1_macro_mean"]), 6),
            "cv_f1_macro_std": round(float(selected["cv_f1_macro_std"]), 6),
            "feature_importances": {
                col: round(float(imp), 6)
                for col, imp in zip(IRIS_FEATURE_COLS, pruned_clf.feature_importances_)
            },
            "decision_rules_text": tree_rules_text,
            "test_metrics": pruned_test_eval,
        },
        "dummy_baseline_test": dummy_eval,
        "model_artifact": str(model_path.relative_to(dirs["root"])),
        "figures": [
            str(fig8.relative_to(dirs["root"])),
            str(fig9.relative_to(dirs["root"])),
            str(fig10.relative_to(dirs["root"])),
        ],
    }
    save_json(summary, dirs["metrics_dir"] / "task4_decision_tree_metrics.json")
    return summary
