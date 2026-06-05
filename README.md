# Engineering Report: Bank Customer Churn Predictive Modeling

This report documents the end-to-end development, evaluation, and engineering pipeline for a high-performance customer churn prediction system. The primary goal is to identify at-risk customers with high precision and recall, allowing for targeted retention strategies before capital flight occurs.

---

## 1. Project Overview & Objective
Customer retention is a critical driver of profitability in retail banking. Acquiring new customers costs significantly more than retaining existing ones. This project establishes a robust machine learning workflow to ingest raw customer account metrics, isolate behavioral signals, and flag individuals with a high probability of churning (`churn_flag`). 

The pipeline handles high-cardinality categorical attributes, enforces strict data isolation to prevent target leakage, and benchmarks ensemble frameworks to deliver a production-ready model.

---

## 2. Exploratory Data Analysis (EDA) Highlights
Before transformation, data distributions and correlations were analyzed to understand underlying structural patterns:
* **Target Distribution:** The target variable `churn_flag` exhibits a severe class imbalance, with non-churned customers making up the vast majority of the dataset. This dictated the use of stratified splits and evaluation metrics tied to imbalanced data (F1-Score, ROC AUC) rather than raw accuracy.
* **Feature Relationships:** Correlation analysis showed that variables like `numcomplaints` and `outstanding_loans` possess strong positive correlations with churn behavior, whereas high `credit_score` and established `customer_tenure` act as negative indicators.
* **Categorical Imbalance:** Feature segments such as `occupation` showed high-cardinality nominal distributions, requiring dedicated encoding strategies to preserve signal integrity without inducing severe matrix sparsity.

---

## 3. Data Preprocessing & Feature Engineering Pipeline

To transform the raw `botswana_bank_customer_churn.csv` dataset into a clean, model-ready tensor matrix, the following sequential ETL pipeline was developed:

* **Schema Normalization:** Stripped whitespaces, lowercased all feature names, and replaced spaces with underscores (`_`) to ensure standardized, programmatic formatting.
* **Feature Selection & Dropping:** * Erased high-cardinality identifiers that lack predictive power: `rownumber`, `customerid`, `surname`, and `first_name`.
  * Removed features with more than 50% missing values (`churn_reason` and `churn_date`) to prevent data imputation noise.
  * Extracted and removed complex text strings such as `address` and `contact_information`.
* **Feature Extraction:** Transformed the raw string timestamp `date_of_birth` into a computed, stable integer feature (`age`) based on the current system runtime year.
* **Categorical Vectorization:** Applied One-Hot Encoding to categorical columns (`gender`, `marital_status`, `occupation`, `education_level`, `customer_segment`, `preferred_communication_channel`). Due to the 639 distinct values within the `occupation` feature, this expanded the structural layout from 17 features to a 657-dimensional sparse matrix.
* **Data Isolation:** Enforced a strict 80/20 train/test split utilizing a stratified distribution method based on the `churn_flag` target array. This guarantees that both partitions hold identical target class proportions.
* **Feature Scaling:** Initialized an `sklearn.preprocessing.StandardScaler` instance. The scaler computed mean and variance attributes *solely* from the training subset (`X_train`) and projected these parameters onto the testing subset (`X_test`) to ensure zero forward data leakage.

---

## 4. Model Training & Evaluation

Two distinct machine learning architectures were trained on the processed 657-feature space. Both models were evaluated using the withheld test partition (`X_test`).

### Model 1: Random Forest Classifier
An ensemble of 100 decision trees was initialized via `RandomForestClassifier`. The architecture leveraged bootstrap aggregation to build diverse estimators.
* **Accuracy:** 97.79%
* **Precision:** 98.16%
* **Recall:** 83.47%
* **F1-Score:** 90.22%
* **ROC AUC Score:** 99.77%

*Analysis:* While the baseline Random Forest showed an exceptional capability to separate classes globally (99.77% ROC AUC) and limited its false positive rate (98.16% Precision), it exhibited a visible performance drop in its **Recall** score (83.47%). This indicates a higher rate of false negatives—missing roughly 16.5% of truly churning customers.

### Model 2: XGBoost Classifier
A gradient-boosted decision tree framework was deployed using `XGBClassifier`. The configuration optimized a binary logistic loss objective function (`objective='binary:logistic'`) and penalized classification errors iteratively.
* **Accuracy:** 99.50%
* **Precision:** 98.22%
* **Recall:** 97.66%
* **F1-Score:** 97.94%
* **ROC AUC Score:** 99.98%

*Analysis:* By sequentially correcting structural residual errors, XGBoost effectively resolved the false-negative bottleneck observed in the Random Forest model. It successfully brought **Recall** up to 97.66% while maintaining a stellar precision boundary of 98.22%.

---

## 5. Comparative Evaluation Summary

| Model Framework | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest Baseline | 97.79% | 98.16% | 83.47% | 90.22% | 99.77% |
| **XGBoost Classifier (Champion)** | **99.50%** | **98.22%** | **97.66%** | **97.94%** | **99.98%** |

### Core Engineering Takeaways
1. **Metric Prioritization:** In a financial churn architecture, **Recall** is the primary business metric. Missing an at-risk customer (False Negative) is significantly more expensive to the institution than misidentifying a loyal customer (False Positive).
2. **Framework Performance:** XGBoost outperformed the Random Forest configuration across every monitored dimension, closing the critical metric gap and achieving an outstanding **F1-Score of 97.94%**. It is selected as the production-ready champion architecture.

---

## 6. Model Serialization & Production Roadmap

* **Artifact Exportation:** The trained champion XGBoost instance has been compiled, frozen, and serialized to disk as a reusable binary asset file named `best_churn_prediction_model.joblib`.
* **Inference Pipeline Refactoring:** Next optimization steps require refactoring the manual text processing and one-hot encoding code blocks into a native, single object workflow using an `sklearn.pipeline.Pipeline`. This encapsulates the scaling parameters, OHE mapping indices, and the `.joblib` model artifact into a unified executable object, mitigating feature alignment issues during production calls.
* **Telemetry & Drift Auditing:** Upon exposing the model via an API endpoint, automated logging routines will track incoming data payload distributions. This telemetry will specifically monitor the high-cardinality `occupation` arrays for structural covariate drift, prompting automated retraining alerts when input feature spaces diverge from training boundaries.
