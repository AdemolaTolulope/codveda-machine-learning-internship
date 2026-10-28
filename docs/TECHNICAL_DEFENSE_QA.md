# Technical Defence Preparation — Evaluator Q&A Guide

Every answer below is grounded directly in the actual code, equations, and verified metrics of this repository so you can explain and defend every engineering decision accurately.

---

### Q1. Data Preprocessing: *Why did you use `2) Stock Prices Data Set.csv` for missing-data imputation instead of `iris.csv` or `churn-bigml-80.csv`, and why median over mean?*
**Answer:**  
When I audited all six raw CSVs supplied in the Codveda archive, `iris.csv`, `churn-bigml-80.csv`, `churn-bigml-20.csv`, `Sentiment dataset.csv`, and `house Prediction Data Set.csv` all contained **zero** missing values (`0 NaN`s). Rather than fabricating synthetic `NaN`s, I used `2) Stock Prices Data Set.csv` ($497,472$ rows), which contained **27 genuine missing values** across 11 trading days (`open`: 11, `high`: 8, `low`: 8). Because stock prices and trading volumes are heavily right-skewed (`close` skewness $= 6.08$, `volume` skewness $= 16.92$), the training mean (`$81.82`) is pulled well above the typical stock (`$61.97` median). Fitting `SimpleImputer(strategy="median", add_indicator=True)` strictly on the 2014–2016 training partition prevented outlier distortion while preserving binary missingness flags for zero-volume trading halts.

---

### Q2. Train/Test Leakage: *What specific forms of data leakage could occur in these tasks, and how does your code guarantee none occurred?*
**Answer:**  
Three distinct leakage traps existed:
1. **Transformation leakage:** Fitting scalers or imputers on the full dataset before splitting leaks test distribution moments. All transformations live inside `scikit-learn` `Pipeline` / `ColumnTransformer` objects (`src/codveda_ml/preprocessing/pipelines.py`), verified by `tests/test_preprocessing.py` (which confirms training numeric columns have exact mean `0.0000` and SD `1.0000`, whereas transformed test columns retain natural shift).
2. **Temporal leakage:** Randomly splitting `Stock Prices` would allow a model to learn 2016 market crashes from 2016 adjacent days in the test set. I enforced a strict chronological cutoff (`2014–2016` train vs `2017` test) and computed `next_close` strictly within each ticker.
3. **Threshold-tuning leakage:** Choosing a classification probability cutoff by maximising F1 on `churn-bigml-20` leaks test labels. Instead, `select_optimal_threshold_oof` (`src/codveda_ml/evaluation/metrics.py`) selects $\tau^*$ solely from 5-fold out-of-fold (`cross_val_predict`) probabilities on `churn-bigml-80`.

---

### Q3. Feature Scaling: *Why did unscaled KNN score `1.0000` on the 30-row Iris test split while `StandardScaler` KNN scored `0.9333`, whereas scaling was essential for SVM on Churn?*
**Answer:**  
In Customer Churn, predictors have wildly different units (`Account length` spans `1–243` days, `Total day minutes` spans `0–351` minutes, while binary plan indicators span `0–1`). Without `StandardScaler`, Euclidean distances in KNN or RBF SVM would be completely dominated by `Total day minutes` and `Account length`.  
In Iris, however, all four features are already measured in centimetres, and biological species separation resides almost entirely in `petal_length` ($\sigma^2 = 3.11\text{ cm}^2$) and `petal_width` ($\sigma^2 = 0.58\text{ cm}^2$), while `sepal_width` ($\sigma^2 = 0.19\text{ cm}^2$) is noisy and overlaps heavily across *versicolor* and *virginica*. Applying `StandardScaler` forces `sepal_width` to have the exact same variance ($1.0$) as `petal_length`, causing two borderline *virginica* test flowers with unusual `sepal_width` to fall closer to *versicolor* neighbours. Crucially, we kept the scaled pipeline selected on training CV rather than peeking at the test set to swap models retroactively.

---

### Q4. KNN & Choice of $K$: *How does $K$ control the bias–variance trade-off, and how did you pick $K = 3$?*
**Answer:**  
At $K = 1$, KNN achieves $100\%$ training accuracy (`0` training error) because every point is its own nearest neighbour, creating high-variance Voronoi boundaries sensitive to label or measurement noise. Very large $K$ (e.g., $K = 31$ on a 96-sample CV training fold, where each class has only 32 points) oversmooths boundaries across class regions, increasing bias. Across our 5-fold stratified CV grid on the 120 Iris training flowers, `K = 3` (`uniform` weights, scaled) achieved the highest mean CV Macro-F1 (`0.9665 ± 0.0187`) while using an odd integer to reduce two-class tie probabilities.

---

### Q5. Logistic Regression, Collinearity & Odds Ratios: *Why did you drop the four `* charge` columns before fitting Logistic Regression, and how do you interpret an Odds Ratio of `7.28`?*
**Answer:**  
In `churn-bigml-80.csv`, `Total day charge = 0.17 × Total day minutes` ($r = 0.99999995$), and the same exact formula holds for evening (`0.085`), night (`0.045`), and international (`0.27`) charges. Keeping all 15 numeric columns produced a Variance Inflation Factor of **$10,410,725.8$**, which splits linear weights arbitrarily between `minutes` and `charge` and invalidates ceteris-paribus ("holding other variables constant") odds-ratio interpretation (since you cannot increase `Total day minutes` by 1 SD while holding `Total day charge` constant!). Dropping the four derived charge columns reduced max VIF to **`1.004`**.  
For `International Plan = Yes`, $\hat{\beta} = 1.9852$ gives $\text{OR} = e^{1.9852} = 7.28$ ($95\%$ bootstrap CI `[5.51, 9.79]`): holding usage minutes, service calls, voice-mail plan, area code, and state constant, customers enrolled in the international plan have **7.28 times higher modelled odds** ($\frac{p}{1-p}$) of churning than unenrolled customers. Note that odds ($p/(1-p)$) are not probabilities ($p$), and this is an observational association, not proof of causation.

---

### Q6. Decision-Tree Pruning: *How does minimal cost-complexity pruning (`ccp_alpha`) work, and what changed on Iris?*
**Answer:**  
Minimal cost-complexity pruning minimises $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is total leaf impurity weighted by node sample fraction, $|T|$ is the number of terminal leaves, and $\alpha \ge 0$ (`ccp_alpha`) is the complexity penalty per additional leaf. As $\alpha$ increases from `0`, weakest internal subtrees whose impurity reduction per split falls below $\alpha$ are collapsed sequentially. On Iris, the unpruned tree (`alpha = 0`) grew to `8` leaves (`Depth = 5`) by splitting single noisy flowers on `sepal_length` and `sepal_width`, scoring `0.9496` CV Macro-F1 and `93.33%` test accuracy. Pruning at `ccp_alpha = 0.008114` eliminated every sepal split, leaving `4` clean petal leaves (`Depth = 3`) and improving CV Macro-F1 to `0.9663` and held-out test accuracy to **`96.67%` (`29/30`)**.

---

### Q7. Random Forest & Feature Importance: *Why did `State` look moderately important in Gini MDI (`4.02%`) yet drop to `0.0002` under Permutation Importance?*
**Answer:**  
Mean Decrease in Impurity (MDI) sums training-set Gini reductions across all splits in the forest. Because `State` expands into 51 binary columns (and continuous variables like `Account length` offer dozens of candidate cutpoints), deep trees (`max_depth = 16`) occasionally use `State` dummies to reduce residual impurity on small bootstrap subsets, inflating in-sample MDI (`4.02%` summed across 51 columns). Permutation importance measures how much **held-out ROC-AUC degrades** when `State` is randomly shuffled 30 times (`+0.00024 ± 0.0008` — indistinguishable from zero), exposing `State`'s in-sample overfitting bias. Conversely, `Total day minutes` (`0.212` AUC drop), `Customer service calls` (`0.134`), and `International plan` (`0.075`) proved genuinely predictive under both measures.

---

### Q8. SVM Kernels & 2D Projections: *Why did RBF SVM (`0.9130` AUC, `0.6970` F1) dramatically beat Linear SVM (`0.8258` AUC, `0.4800` F1), and what is the limitation of your 2D decision-boundary plot?*
**Answer:**  
Linear SVM and Logistic Regression both fit a single flat hyperplane $w^\top x + b = 0$ in the 67-dimensional feature space, which explains why they achieved almost identical AUC (`0.8258` vs `0.8274`) and F1 (`0.4800` vs `0.4658`). Telecom churn violates linear separability because customers churn in disjoint regimes — e.g., dissatisfied customers with `>= 4` customer-service calls even at low daytime minutes, *or* heavy daytime users without voice mail. The RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ maps inputs into an infinite-dimensional Hilbert space, forming closed curved contours around each churn pocket (`Test AUC = 0.9130`).  
However, visualising decision boundaries in 2D requires either training a reduced 2-feature model (`Total day minutes` vs `Customer service calls`) or projecting 67 dimensions onto PC1–PC2 (`17.8%` total variance). Because `82.2%` of feature variance is discarded in 2D PCA, points that are well separated along `International plan` or `Total eve minutes` appear superimposed in 2D.

---

### Q9. Model Selection & Trade-Offs: *Between Logistic Regression, RBF SVM, and Tuned Random Forest, which model would you recommend for deployment and why?*
**Answer:**  
For operational churn prioritisation, **Tuned Random Forest** (`Test ROC-AUC = 0.9258`, `PR-AUC = 0.8828`, `Precision = 90.24%`, `Recall = 77.89%`, `F1 = 0.8362`, `Brier = 0.0412`) dominates both Logistic Regression (`F1 = 0.4658`) and RBF SVM (`F1 = 0.6970`), catching 74 of 95 churners on `churn-bigml-20` while triggering only 8 false alarms (`90.2%` precision). Tree splits naturally match tariff-plan thresholds (`day minutes > ~264`, `service calls >= 4`) while ignoring the 51 uninformative `State` dummies that dilute RBF Euclidean distances. I would pair Random Forest for customer scoring with Logistic Regression's bootstrap odds ratios and permutation-importance summaries for stakeholder reporting.

---

### Q10. Evaluation Metrics: *Why is `85.76%` accuracy useless on Customer Churn, and why report both ROC-AUC and PR-AUC?*
**Answer:**  
Because `85.76%` of `churn-bigml-20` accounts do not churn, `DummyClassifier(strategy="most_frequent")` achieves `85.76%` accuracy while catching **zero** churners (`Recall = 0.00, F1 = 0.00`). Furthermore, ROC-AUC plots True Positive Rate against False Positive Rate ($\text{FP} / (\text{FP} + \text{TN})$); when $\text{TN} = 572$ is large, 75 false positives produce a modest FPR of `13.1%` (`ROC-AUC = 0.827`), even though precision ($\text{TP} / (\text{TP}+\text{FP})$) drops below `42%` (`PR-AUC = 0.427`). Reporting **PR-AUC** (`0.8828` for Random Forest vs `0.4271` for Logistic Regression) directly exposes false-positive cost on the minority class.

---

### Q11. Limitations & Generalisation: *Does `0.9258` ROC-AUC guarantee the Random Forest will achieve `0.9258` in live production next quarter?*
**Answer:**  
No. Cross-validation and `churn-bigml-20` measure generalisation only under the **i.i.d. assumption** (same tariff plans, same customer demographics, same time period). In live telecom operations, competitor promotions, pricing changes, or macroeconomic shifts induce **covariate and concept drift** (similar to the `+0.18 SD` price shift observed between 2014–2016 and 2017 in Task 1's Stock dataset). Production deployment would require scheduled probability calibration checks (Brier score), population stability monitoring on top features (`Total day minutes`, `Customer service calls`), and randomised holdout control groups to verify that retention interventions actually reduce churn.
