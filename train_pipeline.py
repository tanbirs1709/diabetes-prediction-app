import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib

from model_utils import BiologicalZeroImputer, MedicalFeatureEngineering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_export():
    raw_path = os.path.join(DATA_DIR, "diabetes_raw.csv")
    if not os.path.exists(raw_path):
        import urllib.request
        DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        COLUMNS = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
        urllib.request.urlretrieve(DATA_URL, raw_path)
        df = pd.read_csv(raw_path, names=COLUMNS)
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)

    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(C=0.6, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, subsample=0.85, random_state=42)
    }

    # Evaluation comparison
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

    # Production Ensemble Model
    gb_model = models["Gradient Boosting"]
    rf_model = models["Random Forest"]
    lr_model = models["Logistic Regression"]
    
    ensemble_clf = VotingClassifier(
        estimators=[
            ("gb", gb_model),
            ("rf", rf_model),
            ("lr", lr_model)
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
    
    prod_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred_prod), 4),
        "Precision": round(precision_score(y_test, y_pred_prod), 4),
        "Recall": round(recall_score(y_test, y_pred_prod), 4),
        "F1 Score": round(f1_score(y_test, y_pred_prod), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob_prod), 4),
        "CV ROC-AUC Mean": round(cv_ens.mean(), 4),
        "CV ROC-AUC Std": round(cv_ens.std(), 4)
    }
    results["Ensemble (Production)"] = prod_metrics

    # Confusion matrix and ROC
    cm = confusion_matrix(y_test, y_pred_prod)
    fpr, tpr, _ = roc_curve(y_test, y_prob_prod)

    # Feature Importance analysis
    gb_model.fit(
        full_prod_pipeline.named_steps["scaler"].transform(
            full_prod_pipeline.named_steps["feature_engineer"].transform(
                full_prod_pipeline.named_steps["zero_imputer"].transform(X_train)
            )
        ),
        y_train
    )
    
    all_feature_names = feature_cols + ["BMI_Age_Risk", "Glucose_BMI_Risk", "Insulin_Glucose_Ratio"]
    feature_importances = dict(zip(all_feature_names, [round(float(v), 4) for v in gb_model.feature_importances_]))

    # Curate dataset view with realistic clean values for EDA
    df_curated_view = df.copy()
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        df_curated_view[col] = df_curated_view[col].replace(0, np.nan)
    
    knn = KNNImputer(n_neighbors=5)
    df_curated_view[feature_cols] = knn.fit_transform(df_curated_view[feature_cols])
    df_curated_view.to_csv(os.path.join(DATA_DIR, "diabetes_curated.csv"), index=False)

    # Compute descriptive statistics & reference ranges
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            "min": float(df_curated_view[col].min()),
            "max": float(df_curated_view[col].max()),
            "mean": float(df_curated_view[col].mean()),
            "median": float(df_curated_view[col].median()),
            "std": float(df_curated_view[col].std()),
            "median_nondiabetic": float(df_curated_view[df_curated_view["Outcome"] == 0][col].median()),
            "median_diabetic": float(df_curated_view[df_curated_view["Outcome"] == 1][col].median()),
            "normal_ref_min": 0 if col == "Pregnancies" else (70 if col == "Glucose" else (60 if col == "BloodPressure" else (10 if col == "SkinThickness" else (16 if col == "Insulin" else (18.5 if col == "BMI" else (0.08 if col == "DiabetesPedigreeFunction" else 18)))))),
            "normal_ref_max": 2 if col == "Pregnancies" else (99 if col == "Glucose" else (80 if col == "BloodPressure" else (25 if col == "SkinThickness" else (166 if col == "Insulin" else (24.9 if col == "BMI" else (0.50 if col == "DiabetesPedigreeFunction" else 45)))))),
            "unit": "count" if col == "Pregnancies" else ("mg/dL" if col == "Glucose" else ("mm Hg" if col == "BloodPressure" else ("mm" if col == "SkinThickness" else ("µU/mL" if col == "Insulin" else ("kg/m²" if col == "BMI" else ("score" if col == "DiabetesPedigreeFunction" else "years"))))))
        }

    artifact = {
        "model_pipeline": full_prod_pipeline,
        "model_name": "Gradient Boosting & Random Forest Ensemble",
        "feature_names": feature_cols,
        "engineered_feature_names": all_feature_names,
        "target_name": "Outcome",
        "metrics": prod_metrics,
        "all_model_results": results,
        "feature_stats": feature_stats,
        "feature_importances": feature_importances,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr.tolist()],
            "tpr": [round(float(x), 4) for x in tpr.tolist()]
        }
    }

    joblib.dump(artifact, os.path.join(MODELS_DIR, "diabetes_pipeline.joblib"))
    print("Training and Artifact generation complete.")

if __name__ == "__main__":
    train_and_export()
