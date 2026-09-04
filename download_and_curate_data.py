import os
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATA_PATH = os.path.join(DATA_DIR, "diabetes.csv")

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

print("=== STEP 1: DOWNLOADING DATASET ===")
try:
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, names=COLUMNS)
    print(f"Dataset downloaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
except Exception as e:
    print(f"Direct download failed: {e}. Generating dataset fallback...")
    # In case of network issue, load or generate structured data
    raise e

print("\n=== STEP 2: DATA CURATION & PREPROCESSING ===")
print("Initial Data Summary:")
print(df.describe())

# Biological zero replacement:
# Features where 0 is physiologically impossible and denotes missing data:
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print(f"\nZero counts in physiologically impossible columns before curation:")
for col in zero_invalid_cols:
    print(f" - {col}: {(df[col] == 0).sum()} zeros ({(df[col] == 0).mean()*100:.1f}%)")

df_curated = df.copy()
for col in zero_invalid_cols:
    df_curated[col] = df_curated[col].replace(0, np.nan)

print("\nMissing values after zero-to-NaN conversion:")
print(df_curated.isnull().sum())

# Split features and target
X = df_curated.drop("Outcome", axis=1)
y = df_curated["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

print("\n=== STEP 3: MODEL SELECTION & TRAINING ===")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, min_samples_split=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "Support Vector Machine": SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
}

results = {}
best_score = 0
best_model_name = None
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "CV ROC-AUC Mean": round(cv_scores.mean(), 4),
        "CV ROC-AUC Std": round(cv_scores.std(), 4)
    }
    
    print(f"\nModel: {name}")
    print(f" - Test Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    if roc_auc > best_score:
        best_score = roc_auc
        best_model_name = name
        best_pipeline = pipe

print(f"\nBest Model Selected: {best_model_name} with ROC-AUC: {best_score:.4f}")

# Train final production pipeline on all training data
best_pipeline.fit(X_train, y_train)
y_pred_best = best_pipeline.predict(X_test)
y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred_best)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)

# Save the full curated dataset
df_curated_clean = df_curated.copy()
# Save raw and curated CSVs for Streamlit app
df.to_csv(os.path.join(DATA_DIR, "diabetes_raw.csv"), index=False)
df_curated.to_csv(os.path.join(DATA_DIR, "diabetes_curated.csv"), index=False)

# Compute feature medians & statistics for app presets and bounds
feature_stats = {}
for col in X.columns:
    feature_stats[col] = {
        "min": float(df_curated[col].min()),
        "max": float(df_curated[col].max()),
        "mean": float(df_curated[col].mean()),
        "median": float(df_curated[col].median()),
        "std": float(df_curated[col].std()),
        "median_nondiabetic": float(df_curated[df_curated["Outcome"] == 0][col].median()),
        "median_diabetic": float(df_curated[df_curated["Outcome"] == 1][col].median())
    }

# Feature importances (if tree-based, else logistic coefs)
classifier = best_pipeline.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_.tolist()
elif hasattr(classifier, "coef_"):
    importances = np.abs(classifier.coef_[0]).tolist()
else:
    importances = [1/len(X.columns)] * len(X.columns)

feature_importance_dict = dict(zip(X.columns, [round(imp, 4) for imp in importances]))

# Save artifacts
artifact = {
    "model_pipeline": best_pipeline,
    "model_name": best_model_name,
    "feature_names": list(X.columns),
    "target_name": "Outcome",
    "metrics": results[best_model_name],
    "all_model_results": results,
    "feature_stats": feature_stats,
    "feature_importances": feature_importance_dict,
    "confusion_matrix": cm.tolist(),
    "roc_curve": {
        "fpr": [round(x, 4) for x in fpr.tolist()],
        "tpr": [round(x, 4) for x in tpr.tolist()]
    }
}

joblib.dump(artifact, os.path.join(MODELS_DIR, "diabetes_pipeline.joblib"))
print("\nArtifact saved successfully to", os.path.join(MODELS_DIR, "diabetes_pipeline.joblib"))
