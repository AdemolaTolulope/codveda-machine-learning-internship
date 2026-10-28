# Data Dictionary — Official Codveda Machine Learning Archive

All schemas, types, units, ranges, and data-quality observations below were verified directly against the supplied raw files in `data/raw/`.

---

## 1. `1) iris.csv` (`150 rows × 5 columns`, `3,858 bytes`)

| Column | Raw Type | Units / Domain | Summary / Categories | Role & Notes |
| :--- | :--- | :--- | :--- | :--- |
| `sepal_length` | `float64` | cm (`4.30 – 7.90`) | Mean `5.843`, SD `0.828` | Continuous predictor |
| `sepal_width` | `float64` | cm (`2.00 – 4.40`) | Mean `3.054`, SD `0.434` | Continuous predictor (low variance, high class overlap) |
| `petal_length` | `float64` | cm (`1.00 – 6.90`) | Mean `3.759`, SD `1.764` | Primary discriminative predictor |
| `petal_width` | `float64` | cm (`0.10 – 2.50`) | Mean `1.199`, SD `0.763` | Secondary discriminative predictor |
| `species` | `object` | Categorical (3 levels) | `setosa` (50), `versicolor` (50), `virginica` (50) | Multiclass target (Tasks 2 & 4) |

---

## 2. `Churn Prdiction Data/churn-bigml-80.csv` (`2,666 × 20`) & `churn-bigml-20.csv` (`667 × 20`)

| Column | Raw Type | Domain / Range (Train) | Role & Preprocessing Treatment |
| :--- | :--- | :--- | :--- |
| `State` | `object` | 51 two-letter US codes (`50` states + `DC`) | Nominal categorical $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')` |
| `Account length` | `int64` | `1 – 243` days/weeks active | Numeric predictor $\rightarrow$ median imputer + `StandardScaler` |
| `Area code` | `int64` | `{408, 415, 510}` | **Nominal categorical** (cast to `str` inside `ChurnSchemaNormalizer`) |
| `International plan` | `object` | `No` (2,396), `Yes` (270) | Binary indicator (`1 = Yes`, `0 = No`) |
| `Voice mail plan` | `object` | `No` (1,933), `Yes` (733) | Binary indicator (`1 = Yes`, `0 = No`); strictly coupled to `Number vmail messages > 0` |
| `Number vmail messages` | `int64` | `0 – 50` messages | Numeric predictor |
| `Total day minutes` | `float64` | `0.0 – 350.8` minutes | Numeric predictor |
| `Total day calls` | `int64` | `0 – 160` calls | Numeric predictor |
| `Total day charge` | `float64` | `$0.00 – $59.64` | **Dropped** (`0.17 × Total day minutes`, $r = 0.99999995$, $\text{VIF} > 10^7$) |
| `Total eve minutes` | `float64` | `0.0 – 363.7` minutes | Numeric predictor |
| `Total eve calls` | `int64` | `0 – 170` calls | Numeric predictor |
| `Total eve charge` | `float64` | `$0.00 – $30.91` | **Dropped** (`0.085 × Total eve minutes`, $r = 0.99999978$) |
| `Total night minutes` | `float64` | `43.7 – 395.0` minutes | Numeric predictor |
| `Total night calls` | `int64` | `33 – 166` calls | Numeric predictor |
| `Total night charge` | `float64` | `$1.97 – $17.77` | **Dropped** (`0.045 × Total night minutes`, $r = 0.99999923$) |
| `Total intl minutes` | `float64` | `0.0 – 20.0` minutes | Numeric predictor |
| `Total intl calls` | `int64` | `0 – 20` calls | Numeric predictor |
| `Total intl charge` | `float64` | `$0.00 – $5.40` | **Dropped** (`0.27 × Total intl minutes`, $r = 0.99999265$) |
| `Customer service calls` | `int64` | `0 – 9` calls | Numeric predictor (strong non-linear churn trigger at $\ge 4$ calls) |
| `Churn` | `bool` | `False` (85.45% train / 85.76% test), `True` (14.55% / 14.24%) | Binary target (`1 = Churned`, `0 = Retained`) |

---

## 3. `2) Stock Prices Data Set.csv` (`497,472 rows × 7 columns`, `24,210,279 bytes`)

| Column | Raw Type | Missing Cells | Domain / Summary | Notes |
| :--- | :--- | ---: | :--- | :--- |
| `symbol` | `object` | 0 | 505 S&P 500 tickers (`AAL` .. `ZTS`) | 6 tickers appear only in 2017 (`APTV, BHF, BHGE, DWDP, DXC, HLT`) |
| `date` | `object` | 0 | `2014-01-02` to `2017-12-29` (1,007 trading days) | Used strictly for chronological ordering & `2016-12-31` split |
| `open` | `float64` | **11** | `$1.62 – $2,044.00` USD | Imputed via training median + missingness indicator |
| `high` | `float64` | **8** | `$1.69 – $2,067.99` USD | Imputed via training median + missingness indicator |
| `low` | `float64` | **8** | `$1.50 – $2,035.11` USD | Imputed via training median + missingness indicator |
| `close` | `float64` | 0 | `$1.59 – $2,049.00` USD | Contemporaneous close; next-day `close_{t+1} > close_t` defines supervised label |
| `volume` | `int64` | 0 | `0 – 618,237,563` shares | 4 missing-OHLC records also exhibit `volume = 0` |

---

## 4. Audited Auxiliary Archive Datasets (`3) Sentiment dataset.csv` & `4) house Prediction Data Set.csv`)

- **`3) Sentiment dataset.csv` (`732 × 15`):** Social-media posts with `Unnamed: 0.1` and `Unnamed: 0` index columns, `Text`, `Sentiment` (191 stripped emotion tags), `Timestamp`, `User`, `Platform` (`Instagram`, `Twitter`, `Facebook` after stripping whitespace), `Hashtags`, `Retweets`, `Likes`, `Country`, `Year`, `Month`, `Day`, `Hour`.
- **`4) house Prediction Data Set.csv` (`506 × 14`):** Classic Boston Housing dataset stored whitespace-delimited without a header row (`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV`), with 16 right-censored observations at `MEDV = 50.0`.
