import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_raw.csv")
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(DATA_PATH)

zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_curated = df.copy()
for col in zero_invalid_cols:
    df_curated[col] = df_curated[col].replace(0, np.nan)

# Let's test basic imputation before feature engineering
imputer = KNNImputer(n_neighbors=7)
imputed_data = imputer.fit_transform(df_curated.drop("Outcome", axis=1))
df_imputed = pd.DataFrame(imputed_data, columns=df_curated.drop("Outcome", axis=1).columns)
df_imputed["Outcome"] = df_curated["Outcome"].values

# Add domain-specific clinical features
df_imputed["BMI_Age_Risk"] = (df_imputed["BMI"] * df_imputed["Age"]) / 100.0
df_imputed["Glucose_BMI_Risk"] = (df_imputed["Glucose"] * df_imputed["BMI"]) / 100.0
df_imputed["Insulin_Glucose_Ratio"] = df_imputed["Insulin"] / (df_imputed["Glucose"] + 1e-5)

X = df_imputed.drop("Outcome", axis=1)
y = df_imputed["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=42),
}

for name, m in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(m, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    m.fit(X_train_scaled, y_train)
    preds = m.predict(X_test_scaled)
    probs = m.predict_proba(X_test_scaled)[:, 1]
    print(f"{name:20s} | CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f} | Test Acc: {accuracy_score(y_test, preds)*100:.2f}% | Test ROC-AUC: {roc_auc_score(y_test, probs):.4f} | Test F1: {f1_score(y_test, preds):.4f}")

# Voting Classifier
ensemble = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"])
    ],
    voting="soft",
    weights=[1, 2, 2]
)
scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
ensemble.fit(X_train_scaled, y_train)
preds = ensemble.predict(X_test_scaled)
probs = ensemble.predict_proba(X_test_scaled)[:, 1]
print(f"{'Ensemble (Soft)':20s} | CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f} | Test Acc: {accuracy_score(y_test, preds)*100:.2f}% | Test ROC-AUC: {roc_auc_score(y_test, probs):.4f} | Test F1: {f1_score(y_test, preds):.4f}")
