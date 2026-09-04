# 🩺 GlucoGuard AI — Clinical Diabetes Risk Assessment & Data Curation Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical machine learning application and interactive diagnostic dashboard built for diabetes risk stratification, data curation, multivariate biological imputation, and automated screening.

---

## 📌 Executive Summary

Early detection of Type-2 diabetes can prevent severe microvascular and macrovascular complications. However, clinical datasets frequently suffer from structural zero-encoding (where missing diagnostic values like Insulin or Skin Thickness are logged as `0`), skewed feature distributions, and class imbalance.

**GlucoGuard AI** addresses these challenges through:
1. **Multivariate Biological Zero Imputation (KNN, $k=5$)** preserving inter-feature covariance.
2. **Clinical Domain Feature Engineering** (Insulin Resistance Proxies, Glucose-BMI Interaction Risk, Age-BMI indices).
3. **Calibrated Soft-Voting Ensemble** combining Logistic Regression, Random Forest, and Gradient Boosting.
4. **Interactive 4-Tab Web Application** featuring live gauge meters, clinical recommendations, EDA exploration, and batch patient CSV screening.

---

## 🏗️ Architecture & Pipeline Flow

```
[ Raw Clinical Data (PIMA) ]
           │
           ▼
[ 1. Data Curation & Quality Audit ]
   • Biological Zero-to-NaN Conversion (Glucose, BP, Skin, Insulin, BMI)
   • Multivariate 5-Nearest Neighbors Imputation
           │
           ▼
[ 2. Domain Feature Engineering ]
   • Glucose × BMI Risk Index
   • BMI × Age Index
   • Insulin / Glucose Resistance Proxy
           │
           ▼
[ 3. Robust Scaling & Model Pipeline ]
   • Outlier-resistant feature scaling (RobustScaler)
   • Multi-model 5-Fold Stratified Cross-Validation
   • Soft Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting)
           │
           ▼
[ 4. Interactive Streamlit Web Platform ]
   ├── 🩺 Diagnostic Assessment Form & Risk Gauge
   ├── 📊 Interactive Exploratory Data Analysis (EDA)
   ├── 🔬 Model Benchmarking & ROC-AUC Curves
   └── 📁 Batch Patient CSV Screening & Export
```

---

## 📊 Benchmark & Performance Metrics

Evaluated using **5-Fold Stratified Cross-Validation** and a holdout test split (20%):

| Model Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test ROC-AUC | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2, C=1.0)** | 84.47% (±1.65%) | 70.13% | 81.50% | 0.549 |
| **Random Forest (150 trees, max_depth=6)** | 83.74% (±1.99%) | 70.13% | 80.94% | 0.540 |
| **Gradient Boosting (lr=0.05, max_depth=3)** | 82.22% (±2.56%) | 74.68% | 82.48% | 0.621 |
| **🏆 Production Ensemble (Soft Voting)** | **83.91% (±2.03%)** | **72.08%** | **82.07%** | **0.574** |

---

## 🌟 Interactive Dashboard Features

1. **Patient Assessment Form:**
   - Single-patient biometric input with clinical archetype presets (Healthy Athlete, Borderline Prediabetic, High-Risk Patient).
   - Real-time Plotly Gauge Meter color-coded by clinical thresholds (Low < 35%, Moderate 35-65%, High > 65%).
   - Normal vs. Elevated Biomarker Reference Grid.
   - Evidence-based personalized lifestyle & clinical action recommendations.

2. **Interactive EDA Suite:**
   - Distribution histograms with kernel density estimates (KDE).
   - Comparative boxplots segmented by outcome.
   - Feature correlation heatmaps.
   - Interactive 3D feature scatter space.

3. **Model Transparency Tab:**
   - Feature importance rankings (MDI / Coefficients).
   - Confusion matrix and ROC-AUC curve visualizations.

4. **Batch Screening Engine:**
   - Upload any multi-patient CSV.
   - Run parallel inference and receive enriched risk scores, category flags, and instant CSV export.

---

## 📁 Repository Structure

```
diabetes-prediction-app/
├── app.py                                # Full interactive Streamlit Web Application
├── model_utils.py                        # Custom scikit-learn transformers (imputation & feature engineering)
├── train_pipeline.py                     # Pipeline training and artifact serialization script
├── test_models.py                        # Standalone benchmark test runner
├── download_and_curate_data.py           # Data curation and raw/curated dataset generator
├── generate_notebook.py                  # Automated headless notebook generation script
├── diabetes_curation_and_modeling.ipynb  # Pre-executed Jupyter Notebook with full EDA & math
├── presentation_notes.md                 # Clinical & technical Q&A reference guide
├── requirements.txt                      # Production dependencies
├── data/
│   ├── diabetes_raw.csv                  # Raw PIMA Indian Diabetes dataset
│   ├── diabetes_curated.csv              # Curated dataset after KNN biological imputation
│   └── diabetes.csv                      # Baseline dataset
└── models/
    └── diabetes_pipeline.joblib          # Serialized production ensemble model artifact
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR-USERNAME>/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### 2. Create and activate a virtual environment
```bash
# Using standard venv
python3 -m venv .venv
source .venv/bin/activate

# Or using uv (blazing fast)
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
Open **`http://127.0.0.1:8501`** in your browser.

---

## 🧪 Retraining & Pipeline Verification

To retrain the model pipeline and regenerate the exported `.joblib` artifact:
```bash
python train_pipeline.py
```

To run the automated model evaluation benchmarks:
```bash
python test_models.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
