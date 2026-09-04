# 🎓 Project Presentation Cheat Sheet

Use this quick guide when presenting your project to your instructor or reviewing your work.

---

### 1. The 30-Second Elevator Pitch
> *"For my data curation and machine learning capstone, I built **GlucoGuard AI**, an end-to-end diabetes prediction system. Rather than just fitting a baseline classifier, I focused heavily on **data curation** — auditing the dataset for medically impossible zero values in insulin, glucose, and blood pressure, imputing them using multivariate KNN, engineering clinical interaction features, and deploying a soft-voting ensemble model with an interactive Streamlit clinical dashboard."*

---

### 2. Key Data Curation Concepts to Highlight
- **The Problem:** The Pima Indians Diabetes dataset encodes missing observations as `0`. Having 0 glucose, 0 blood pressure, or 0 BMI is biologically impossible for a living human.
  - *Insulin had 48.7% missing/zero values.*
  - *Skinfold thickness had 29.6% missing/zero values.*
  - *Blood pressure had 4.6% missing/zero values.*
- **The Solution:** We converted `0`s to `NaN` and applied **KNN Imputation ($k=5$)**, which predicts missing values using the nearest clinical neighbors rather than distorting variance with simple column means.
- **Scaling:** Used `RobustScaler` (median & interquartile range) because clinical data contains natural outliers.

---

### 3. Feature Engineering You Created
1. **`BMI_Age_Risk` ($\text{BMI} \times \text{Age} / 100$):** Captures how excess weight compounds with age-related metabolic slowdown.
2. **`Glucose_BMI_Risk` ($\text{Glucose} \times \text{BMI} / 100$):** Synergistic interaction between elevated blood sugar and adiposity.
3. **`Insulin_Glucose_Ratio` ($\text{Insulin} / \text{Glucose}$):** Mathematical proxy for insulin resistance and pancreatic workload.

---

### 4. Models Benchmarked & Results
| Model | Test Accuracy | F1 Score | Test ROC-AUC | 5-Fold CV ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (L2)** | 70.13% | 0.5490 | 0.8159 | 0.8420 |
| **Random Forest (180 trees)** | 68.83% | 0.5200 | 0.8080 | 0.8289 |
| **Gradient Boosting** | 74.03% | 0.6154 | 0.8235 | 0.8167 |
| **Soft Voting Ensemble (Final)** | **72.73%** | **0.5882** | **0.8209** | **0.8294** |

---

### 5. Questions Your Instructor Might Ask & How to Answer

**Q1: Why use ROC-AUC instead of only Accuracy?**
> *"Because clinical datasets are often class-imbalanced (here ~65% non-diabetic vs 35% diabetic). A model could achieve 65% accuracy by simply predicting non-diabetic every time. ROC-AUC evaluates the model's true diagnostic ability across all decision thresholds."*

**Q2: Why use a Soft Voting Ensemble?**
> *"Soft voting averages the predicted probability distributions of Gradient Boosting, Random Forest, and Logistic Regression, reducing individual model variance and producing well-calibrated risk probabilities."*

**Q3: How does the web app connect to the machine learning code?**
> *"The complete preprocessing pipeline (zero imputer + feature engineering + scaler + ensemble model) is serialized into a single `.joblib` artifact. The Streamlit app passes raw form inputs directly into the pipeline's `predict_proba()` method for real-time inference."*
