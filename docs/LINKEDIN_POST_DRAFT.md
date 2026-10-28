# LinkedIn Project Completion Post Draft

**Instructions before posting:**
1. Your public GitHub repository URL (`https://github.com/AdemolaTolulope/codveda-machine-learning-internship`) is already inserted and verified below!
2. Attach your recorded 4–5 minute walkthrough video (see `docs/VIDEO_DEMONSTRATION_SCRIPT.md`) or attach key summary visuals (`fig16_portfolio_model_comparison_summary.png`, `fig14_svm_decision_boundaries_2d.png`, `fig08_decision_tree_unrestricted_vs_pruned.png`).
3. Tag **@Codveda** directly in the LinkedIn composer so the mention resolves to Codveda Technologies' official LinkedIn page.

---

## Draft Post (Authentic, Technically Grounded)

Over the past month, I completed my Machine Learning Internship portfolio with **@Codveda** (Codveda Technologies), building and validating six end-to-end workflows across Basic, Intermediate, and Advanced tiers inside a single reproducible Python package (`codveda_ml`) tested with `pytest`.

Rather than treating each task as a standalone notebook, I focused on strict anti-leakage engineering, honest baseline comparisons, and investigating *why* different model families succeed or fail on the same data:

🔹 **Level 1 (Basic) — Leakage-Free Preprocessing & KNN Classification**
• Audited 6 raw datasets, building a chronological `ColumnTransformer` pipeline for 497,472 S&P 500 daily OHLCV records to impute genuine missing price entries using training-only medians while handling unseen 2017 ticker symbols safely.
• Detected exact linear billing collinearity (`minutes` vs `charges`, `r > 0.99999`) in the BigML Telecom Churn dataset that inflated VIF above 10,000,000 — pruning redundant charge columns brought max VIF down to `1.004`.
• Tuned KNN (`K = 3`) on Iris via 5-fold stratified CV (`0.9665` CV Macro-F1, `93.33%` test accuracy, `0.9933` ROC-AUC) and analysed how isotropic scaling interacts with noisy low-variance features.

🔹 **Level 2 (Intermediate) — Logistic Regression & Decision-Tree Pruning**
• Applied minimal cost-complexity pruning (`ccp_alpha = 0.0081`) on Iris Decision Trees, cutting structural complexity from 8 leaves to 4 human-readable petal rules and improving held-out test accuracy from `93.33%` to `96.67%`.
• Modelled binary Customer Churn (`2,666` train / `667` held-out test rows, `14.24%` churn) with regularized Logistic Regression, calibrating probability thresholds strictly on out-of-fold training predictions and estimating 1,000-resample bootstrap 95% confidence intervals on odds ratios (`International Plan = Yes` OR = `7.28`).

🔹 **Level 3 (Advanced) — SVM Kernel Geometry & Random Forest Ensembles**
• Discovered that both Logistic Regression (`0.8274` AUC, `0.4658` F1) and Linear SVM (`0.8258` AUC, `0.4800` F1) hit the exact same ceiling on Telecom Churn because churn follows non-linear threshold interactions (high daytime usage *or* `>= 4` customer-service calls).
• Switching to an RBF-kernel SVM lifted held-out ROC-AUC to `0.9130` (`0.6970` F1), and a tuned 300-tree Random Forest reached **`0.9258` ROC-AUC, `0.8828` PR-AUC, `90.24%` precision, and `0.8362` F1** (`0.9268 ± 0.0021` AUC across 10 random seeds).
• Contrasting Gini importance against 30-repeat permutation importance demonstrated empirically that 51 one-hot `State` indicators receive `4.0%` in-sample Gini credit despite near-zero (`0.0002`) generalisable AUC lift.

 *(Disclosure: AI engineering tooling was used to assist with repository scaffolding, plotting automation, and test harness verification; all experimental configurations, statistical interpretations, and results were executed and verified against the raw datasets.)*

📂 **GitHub Repository (Datasets, Trained Models, Executed Notebooks & Technical Report):** https://github.com/AdemolaTolulope/codveda-machine-learning-internship  
🎥 **Walkthrough Video:** Attached below!

Thank you to the team at **@Codveda** for structuring a practical curriculum spanning foundational preprocessing through ensemble diagnostics.

#CodvedaJourney #CodvedaExperience #FutureWithCodveda #MachineLearning #DataScience #ScikitLearn #Python #MLOps
