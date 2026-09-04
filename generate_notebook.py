import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# 🩺 Clinical Diabetes Risk Prediction & Data Curation Pipeline
**Course Project:** 45-Day Data Curation & Machine Learning with Python  
**Author:** Tanbir S  
**Domain:** Healthcare Analytics / Clinical Endocrinology  
**Dataset:** National Institute of Diabetes and Digestive and Kidney Diseases (Pima Indians Cohort)

---

## 📌 1. Project Objectives & Problem Statement
1. **Data Curation & Quality Auditing:** Identify and resolve missing data encoded as biological impossibilities (e.g. zero glucose, zero blood pressure).
2. **Exploratory Data Analysis (EDA):** Analyze distributions, class imbalances, and correlations using Pandas, NumPy, Matplotlib, and Seaborn.
3. **Clinical Feature Engineering:** Construct composite metabolic risk scores and interaction indicators.
4. **Machine Learning Modeling:** Build, cross-validate, and benchmark multiple classification algorithms (Logistic Regression, Random Forest, Gradient Boosting, and Soft-Voting Ensemble).
5. **Model Evaluation & Clinical Diagnostics:** Assess performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
6. **Artifact Export:** Save the full production-ready scikit-learn pipeline for web deployment.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('deep')
%matplotlib inline"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 📥 2. Data Ingestion & Initial Inspection"""))
cells.append(nbf.v4.new_code_cell("""# Load raw dataset
df_raw = pd.read_csv("data/diabetes_raw.csv")
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()"""))

cells.append(nbf.v4.new_code_cell("""df_raw.info()"""))
cells.append(nbf.v4.new_code_cell("""df_raw.describe()"""))

# Data Curation & Missing Value Handling
cells.append(nbf.v4.new_markdown_cell("""## 🛠️ 3. Data Curation: Handling Biological Zeroes
In human physiology, values of `0` for **Glucose**, **Blood Pressure**, **Skin Thickness**, **Insulin**, and **BMI** are medically impossible. They indicate missing or unrecorded test results.

We convert these zero values to `NaN` and examine missingness rates."""))

cells.append(nbf.v4.new_code_cell("""zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_curated = df_raw.copy()

print("Biological Zero Frequencies:")
for col in zero_cols:
    zero_count = (df_raw[col] == 0).sum()
    pct = zero_count / len(df_raw) * 100
    print(f" - {col:15s}: {zero_count} zeros ({pct:.1f}%)")
    df_curated[col] = df_curated[col].replace(0, np.nan)

print("\\nMissing values after converting biological zeroes to NaN:")
print(df_curated.isnull().sum())"""))

# KNN Imputation
cells.append(nbf.v4.new_markdown_cell("""### Multivariate K-Nearest Neighbors (KNN) Imputation
Rather than crude mean/median imputation which distorts relationships between clinical features, we apply **KNN Imputation** ($k=5$)."""))

cells.append(nbf.v4.new_code_cell("""feature_cols = [c for c in df_curated.columns if c != "Outcome"]
knn_imputer = KNNImputer(n_neighbors=5)
df_curated[feature_cols] = knn_imputer.fit_transform(df_curated[feature_cols])

print("Missing values after KNN imputation:")
print(df_curated.isnull().sum())"""))

# Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""## 📊 4. Exploratory Data Analysis (EDA)"""))

cells.append(nbf.v4.new_code_cell("""# 1. Target Class Balance
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df_curated, x='Outcome', palette=['#3b82f6', '#ef4444'])
plt.title('Target Distribution: Non-Diabetic (0) vs Diabetic (1)', fontsize=13, fontweight='bold')
plt.xlabel('Outcome')
plt.ylabel('Patient Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()} ({p.get_height()/len(df_curated)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 2. Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
corr = df_curated.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True)
plt.title("Clinical Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 3. Distribution of Key Biomarkers by Diabetic Outcome
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(ax=axes[0, 0], data=df_curated, x='Outcome', y='Glucose', palette=['#3b82f6', '#ef4444'])
axes[0, 0].set_title('Glucose Concentration by Outcome', fontweight='bold')

sns.boxplot(ax=axes[0, 1], data=df_curated, x='Outcome', y='BMI', palette=['#3b82f6', '#ef4444'])
axes[0, 1].set_title('BMI by Outcome', fontweight='bold')

sns.boxplot(ax=axes[1, 0], data=df_curated, x='Outcome', y='Age', palette=['#3b82f6', '#ef4444'])
axes[1, 0].set_title('Age by Outcome', fontweight='bold')

sns.boxplot(ax=axes[1, 1], data=df_curated, x='Outcome', y='Insulin', palette=['#3b82f6', '#ef4444'])
axes[1, 1].set_title('Insulin by Outcome', fontweight='bold')

plt.tight_layout()
plt.show()"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 🧬 5. Clinical Feature Engineering
We engineer domain-driven interaction features:
1. `BMI_Age_Risk`: $\\text{BMI} \\times \\text{Age} / 100$
2. `Glucose_BMI_Risk`: $\\text{Glucose} \\times \\text{BMI} / 100$
3. `Insulin_Glucose_Ratio`: $\\text{Insulin} / \\text{Glucose}$ (Insulin resistance proxy)"""))

cells.append(nbf.v4.new_code_cell("""from model_utils import BiologicalZeroImputer, MedicalFeatureEngineering

# Demonstrate feature engineering
fe = MedicalFeatureEngineering()
engineered_data = fe.transform(df_curated.drop("Outcome", axis=1))
engineered_cols = feature_cols + ["BMI_Age_Risk", "Glucose_BMI_Risk", "Insulin_Glucose_Ratio"]
df_engineered = pd.DataFrame(engineered_data, columns=engineered_cols)
df_engineered.head()"""))

# Model Training & Cross Validation
cells.append(nbf.v4.new_markdown_cell("""## 🤖 6. Model Training & Cross-Validation Benchmarking"""))

cells.append(nbf.v4.new_code_cell("""X = df_raw.drop("Outcome", axis=1)
y = df_raw["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(C=0.6, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=2, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, subsample=0.85, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    pipe = Pipeline([
        ("zero_imputer", BiologicalZeroImputer()),
        ("feature_engineer", MedicalFeatureEngineering()),
        ("scaler", RobustScaler()),
        ("classifier", clf)
    ])
    
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "CV ROC-AUC Mean": round(cv_scores.mean(), 4),
        "CV ROC-AUC Std": round(cv_scores.std(), 4)
    }

# Build Soft Voting Ensemble
ensemble_clf = VotingClassifier(
    estimators=[
        ("gb", models["Gradient Boosting"]),
        ("rf", models["Random Forest"]),
        ("lr", models["Logistic Regression"])
    ],
    voting="soft",
    weights=[3, 2, 1]
)

full_prod_pipeline = Pipeline([
    ("zero_imputer", BiologicalZeroImputer()),
    ("feature_engineer", MedicalFeatureEngineering()),
    ("scaler", RobustScaler()),
    ("classifier", ensemble_clf)
])

cv_ens = cross_val_score(full_prod_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
full_prod_pipeline.fit(X_train, y_train)

y_pred_prod = full_prod_pipeline.predict(X_test)
y_prob_prod = full_prod_pipeline.predict_proba(X_test)[:, 1]

results["Ensemble (Production)"] = {
    "Accuracy": round(accuracy_score(y_test, y_pred_prod), 4),
    "Precision": round(precision_score(y_test, y_pred_prod), 4),
    "Recall": round(recall_score(y_test, y_pred_prod), 4),
    "F1 Score": round(f1_score(y_test, y_pred_prod), 4),
    "ROC-AUC": round(roc_auc_score(y_test, y_prob_prod), 4),
    "CV ROC-AUC Mean": round(cv_ens.mean(), 4),
    "CV ROC-AUC Std": round(cv_ens.std(), 4)
}

df_results = pd.DataFrame(results).T
df_results"""))

# Model Evaluation & Diagnostic Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 📈 7. Evaluation & Diagnostic Charts"""))

cells.append(nbf.v4.new_code_cell("""# 1. ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_prod)
auc_val = roc_auc_score(y_test, y_prob_prod)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'Ensemble Pipeline (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Random Chance (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_prod)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-Diabetic', 'Predicted Diabetic'],
            yticklabels=['Actual Non-Diabetic', 'Actual Diabetic'])
plt.title('Confusion Matrix (Hold-out Test Set, N=154)', fontsize=13, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Classification Report
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred_prod, target_names=["Non-Diabetic", "Diabetic"]))"""))

# Export Model
cells.append(nbf.v4.new_markdown_cell("""## 💾 8. Model Artifact Persistence
The full pipeline object is exported to `.joblib` for consumption by the interactive Streamlit Web App."""))

cells.append(nbf.v4.new_code_cell("""# Verify saved pipeline
saved_artifact = joblib.load("models/diabetes_pipeline.joblib")
print("Model Pipeline successfully verified on disk!")
print("Pipeline steps:", saved_artifact["model_pipeline"].named_steps.keys())"""))

nb.cells = cells

with open("/home/g15/diabetes-prediction-app/diabetes_curation_and_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook generated successfully: diabetes_curation_and_modeling.ipynb")
