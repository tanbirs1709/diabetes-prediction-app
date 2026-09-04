import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

class BiologicalZeroImputer(BaseEstimator, TransformerMixin):
    """
    Imputes biologically impossible zero values (Glucose, BloodPressure, 
    SkinThickness, Insulin, BMI) using K-Nearest Neighbors.
    """
    def __init__(self, zero_cols_idx=[1, 2, 3, 4, 5], n_neighbors=5):
        self.zero_cols_idx = zero_cols_idx
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        
    def fit(self, X, y=None):
        X_arr = np.array(X, dtype=float, copy=True)
        for idx in self.zero_cols_idx:
            zeros_mask = (X_arr[:, idx] == 0)
            X_arr[zeros_mask, idx] = np.nan
        self.knn_imputer.fit(X_arr)
        return self
        
    def transform(self, X):
        X_arr = np.array(X, dtype=float, copy=True)
        for idx in self.zero_cols_idx:
            zeros_mask = (X_arr[:, idx] == 0)
            X_arr[zeros_mask, idx] = np.nan
        return self.knn_imputer.transform(X_arr)


class MedicalFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Engineers clinically relevant interaction features:
    1. BMI_Age_Risk: Joint impact of BMI and age on metabolic deceleration
    2. Glucose_BMI_Risk: Synergy of high blood glucose with excess adiposity
    3. Insulin_Glucose_Ratio: Proxy for insulin resistance / pancreatic beta-cell workload
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ])
        else:
            X_df = X.copy()
            
        X_df["BMI_Age_Risk"] = (X_df["BMI"] * X_df["Age"]) / 100.0
        X_df["Glucose_BMI_Risk"] = (X_df["Glucose"] * X_df["BMI"]) / 100.0
        X_df["Insulin_Glucose_Ratio"] = X_df["Insulin"] / (X_df["Glucose"] + 1e-5)
        
        return X_df.values
