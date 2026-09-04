import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Import custom transformers so unpickling works seamlessly
from model_utils import BiologicalZeroImputer, MedicalFeatureEngineering

# Page configuration
st.set_page_config(
    page_title="GlucoGuard AI - Diabetes Risk & Data Curation Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        color: #b0bec5;
    }
    
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(255, 255, 255, 0.15);
        color: #eceff1;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    
    .risk-banner-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
    }
    
    .risk-banner-med {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
    }
    
    .risk-banner-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
    }

    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifact and data
@st.cache_resource
def load_model_and_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "diabetes_pipeline.joblib")
    raw_data_path = os.path.join(base_dir, "data", "diabetes_raw.csv")
    curated_data_path = os.path.join(base_dir, "data", "diabetes_curated.csv")
    train_script = os.path.join(base_dir, "train_pipeline.py")
    
    if not os.path.exists(model_path):
        import subprocess
        subprocess.run([sys.executable, train_script], check=True)
        
    artifact = joblib.load(model_path)
    df_raw = pd.read_csv(raw_data_path)
    df_curated = pd.read_csv(curated_data_path)
    return artifact, df_raw, df_curated

artifact, df_raw, df_curated = load_model_and_data()
pipeline = artifact["model_pipeline"]
metrics = artifact["metrics"]
stats = artifact["feature_stats"]

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🩺 GlucoGuard AI</h1>
    <p>End-to-End Clinical Diabetes Risk Prediction & Data Curation Platform</p>
    <div style="margin-top: 12px;">
        <span class="badge-tag">🔬 KNN Biological Imputation</span>
        <span class="badge-tag">⚡ Soft Voting Ensemble</span>
        <span class="badge-tag">📈 ROC-AUC 82.1%</span>
        <span class="badge-tag">🧬 Clinical Feature Engineering</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Preset Profiles & App Info
st.sidebar.header("📋 Clinical Profile Presets")
preset = st.sidebar.selectbox(
    "Choose a pre-configured patient profile:",
    [
        "Custom Input (Manual)",
        "Profile A: Healthy Young Adult (Low Risk)",
        "Profile B: Middle-Aged Borderline (Moderate Risk)",
        "Profile C: High-Risk Metabolic Patient (High Risk)"
    ]
)

# Preset Values
presets = {
    "Profile A: Healthy Young Adult (Low Risk)": {
        "Pregnancies": 0,
        "Glucose": 88.0,
        "BloodPressure": 68.0,
        "SkinThickness": 18.0,
        "Insulin": 65.0,
        "BMI": 21.5,
        "DiabetesPedigreeFunction": 0.22,
        "Age": 24
    },
    "Profile B: Middle-Aged Borderline (Moderate Risk)": {
        "Pregnancies": 2,
        "Glucose": 128.0,
        "BloodPressure": 82.0,
        "SkinThickness": 28.0,
        "Insulin": 135.0,
        "BMI": 28.4,
        "DiabetesPedigreeFunction": 0.48,
        "Age": 42
    },
    "Profile C: High-Risk Metabolic Patient (High Risk)": {
        "Pregnancies": 6,
        "Glucose": 175.0,
        "BloodPressure": 92.0,
        "SkinThickness": 39.0,
        "Insulin": 240.0,
        "BMI": 38.6,
        "DiabetesPedigreeFunction": 0.85,
        "Age": 54
    }
}

defaults = presets.get(preset, {
    "Pregnancies": 1,
    "Glucose": 115.0,
    "BloodPressure": 74.0,
    "SkinThickness": 23.0,
    "Insulin": 95.0,
    "BMI": 26.2,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 33
})

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About the Project")
st.sidebar.markdown("""
- **Domain:** Clinical Endocrinology & Predictive Diagnostics
- **Target:** 2-Hour Diabetes Mellitus Onset (Binary)
- **Curation:** Physiologically impossible zeros treated via KNN Imputation
- **Architecture:** Gradient Boosting + Random Forest + Logistic Regression Soft Voting
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Risk Assessment",
    "📊 Exploratory Data Analysis",
    "🧠 Model Architecture & Metrics",
    "📁 Batch Patient Screening"
])

# ==========================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==========================================
with tab1:
    st.subheader("Interactive Clinical Assessment Form")
    st.markdown("Enter the patient's diagnostic and physical measurements below:")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧪 Biochemical & Blood Indicators")
        
        glucose = st.slider(
            "Plasma Glucose Concentration (2h oral glucose tolerance test)",
            min_value=40.0,
            max_value=220.0,
            value=float(defaults["Glucose"]),
            step=1.0,
            help="Normal fasting is < 100 mg/dL, 2-hr oral glucose test normal is < 140 mg/dL."
        )
        st.caption(f"📍 Standard Normal Range: 70 – 99 mg/dL (Fasting) | < 140 mg/dL (Postprandial)")
        
        insulin = st.slider(
            "2-Hour Serum Insulin (µU/mL)",
            min_value=10.0,
            max_value=600.0,
            value=float(defaults["Insulin"]),
            step=5.0,
            help="Normal fasting range: 16 - 166 µU/mL."
        )
        st.caption(f"📍 Standard Reference Range: 16 – 166 µU/mL")
        
        blood_pressure = st.slider(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=40.0,
            max_value=130.0,
            value=float(defaults["BloodPressure"]),
            step=1.0,
            help="Diastolic blood pressure resting measurement."
        )
        st.caption(f"📍 Standard Optimal Range: 60 – 80 mm Hg")

        skin_thickness = st.slider(
            "Triceps Skin Fold Thickness (mm)",
            min_value=5.0,
            max_value=70.0,
            value=float(defaults["SkinThickness"]),
            step=1.0,
            help="Subcutaneous fat caliper measurement."
        )
        st.caption(f"📍 Reference Range: 10 – 30 mm")

    with col2:
        st.markdown("#### 🧬 Anthropometric & Demographic Indicators")
        
        bmi = st.slider(
            "Body Mass Index (BMI in kg/m²)",
            min_value=14.0,
            max_value=55.0,
            value=float(defaults["BMI"]),
            step=0.1,
            help="Weight (kg) divided by Height squared (m²)."
        )
        st.caption(f"📍 WHO Categories: 18.5 – 24.9 (Normal) | 25 – 29.9 (Overweight) | ≥ 30 (Obese)")
        
        age = st.slider(
            "Patient Age (Years)",
            min_value=18,
            max_value=90,
            value=int(defaults["Age"]),
            step=1
        )
        st.caption(f"📍 Risk increases with age, particularly after 45 years")

        pregnancies = st.slider(
            "Number of Pregnancies",
            min_value=0,
            max_value=17,
            value=int(defaults["Pregnancies"]),
            step=1,
            help="Total number of times pregnant."
        )
        st.caption(f"📍 Indicator for potential gestational diabetes history")

        dpf = st.slider(
            "Diabetes Pedigree Function (Genetic Likelihood)",
            min_value=0.05,
            max_value=2.50,
            value=float(defaults["DiabetesPedigreeFunction"]),
            step=0.01,
            help="Genetic risk score derived from family history of diabetes."
        )
        st.caption(f"📍 Dataset Average: ~0.47 | Values > 0.8 indicate strong familial genetic predisposition")

    st.markdown("---")
    
    # Prediction Calculation
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])
    
    risk_prob = float(pipeline.predict_proba(input_data)[0][1])
    risk_pct = round(risk_prob * 100, 1)
    prediction = int(pipeline.predict(input_data)[0])

    # Result Section
    res_col1, res_col2 = st.columns([1.2, 1.8])
    
    with res_col1:
        st.markdown("### 📊 Diagnostic Risk Meter")
        
        # Plotly Gauge Chart
        gauge_color = "#10b981" if risk_pct < 35 else ("#f59e0b" if risk_pct < 65 else "#ef4444")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number={'suffix': "%", 'font': {'size': 44, 'color': gauge_color, 'family': 'Inter'}},
            title={'text': "Calculated Diabetes Probability", 'font': {'size': 16, 'color': '#334155'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.8,
                    'value': 65
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("### 🏥 Diagnostic Evaluation")
        
        if risk_pct < 35:
            st.markdown(f"""
            <div class="risk-banner-low">
                <h2 style="margin:0; font-size:1.6rem;">🟢 LOW RISK OF DIABETES</h2>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">The AI model estimates a <b>{risk_pct}%</b> probability of diabetes onset. Clinical biomarkers are within favorable boundaries.</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_pct < 65:
            st.markdown(f"""
            <div class="risk-banner-med">
                <h2 style="margin:0; font-size:1.6rem;">🟡 MODERATE / PRE-DIABETIC RISK</h2>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">The AI model estimates a <b>{risk_pct}%</b> probability. Several indicators (e.g. glucose, BMI, or blood pressure) are elevated.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-high">
                <h2 style="margin:0; font-size:1.6rem;">🔴 HIGH RISK OF DIABETES</h2>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">The AI model estimates a <b>{risk_pct}%</b> probability of diabetes mellitus. Clinical review is strongly advised.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Biomarker Status Breakdown")
        
        # Status Grid
        status_items = [
            ("Glucose", f"{glucose} mg/dL", "Normal (< 100)" if glucose < 100 else ("Borderline (100-125)" if glucose <= 125 else "Elevated (≥ 126)"), "#10b981" if glucose < 100 else ("#f59e0b" if glucose <= 125 else "#ef4444")),
            ("BMI", f"{bmi} kg/m²", "Normal (18.5-24.9)" if bmi < 25 else ("Overweight (25-29.9)" if bmi < 30 else "Obese (≥ 30)"), "#10b981" if bmi < 25 else ("#f59e0b" if bmi < 30 else "#ef4444")),
            ("Blood Pressure", f"{blood_pressure} mm Hg", "Optimal (< 80)" if blood_pressure < 80 else ("Pre-hypertension (80-89)" if blood_pressure < 90 else "Hypertensive (≥ 90)"), "#10b981" if blood_pressure < 80 else ("#f59e0b" if blood_pressure < 90 else "#ef4444")),
            ("Insulin", f"{insulin} µU/mL", "Normal Range" if insulin <= 166 else "Elevated (Insulin Resistance)", "#10b981" if insulin <= 166 else "#ef4444")
        ]
        
        cols_grid = st.columns(4)
        for i, (label, val, stat_text, stat_color) in enumerate(status_items):
            with cols_grid[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#64748b; font-weight:600;">{label.upper()}</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#1e293b; margin:4px 0;">{val}</div>
                    <div style="font-size:0.75rem; font-weight:600; color:{stat_color};">{stat_text}</div>
                </div>
                """, unsafe_allow_html=True)

    # Actionable Clinical Recommendations
    st.markdown("### 💡 Tailored Action Plan & Clinical Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("##### 🥗 Nutrition & Physical Activity Targets")
        recs = []
        if glucose >= 100:
            recs.append("• **Glycemic Control:** Prioritize low-GI carbohydrates, eliminate refined sugars, and increase dietary fiber to > 30g/day.")
        else:
            recs.append("• **Balanced Glycemia:** Maintain current dietary habits rich in leafy greens, whole grains, and lean proteins.")
            
        if bmi >= 25:
            recs.append(f"• **Weight Management:** Target a 5–7% reduction in body weight to significantly improve insulin sensitivity (Current BMI: {bmi}).")
        else:
            recs.append("• **Healthy Adiposity:** Maintain current healthy body mass index with regular physical activity.")
            
        if blood_pressure >= 80:
            recs.append(f"• **Cardiovascular Care:** Moderate daily sodium intake (< 2,300 mg/day) and incorporate aerobic exercises (Current BP: {blood_pressure} mm Hg).")
            
        recs.append("• **Activity Goal:** 150 minutes of moderate-intensity aerobic exercise per week plus 2 sessions of resistance training.")
        
        for r in recs:
            st.markdown(r)
            
    with rec_col2:
        st.markdown("##### 🩺 Recommended Diagnostic Follow-ups")
        tests = []
        if risk_pct >= 35 or glucose >= 100:
            tests.append("• **HbA1c Blood Test:** Comprehensive 3-month glycemic control assessment (Target: < 5.7%).")
            tests.append("• **Fasting Lipid Profile:** Total Cholesterol, HDL, LDL, and Triglyceride ratio.")
            tests.append("• **Comprehensive Metabolic Panel (CMP):** Renal and liver function monitoring.")
        else:
            tests.append("• **Routine Annual Screening:** Standard preventative metabolic blood panel once every 12 months.")
            tests.append("• **Home Vitals Monitoring:** Periodic blood pressure and BMI checks.")
            
        if dpf > 0.6:
            tests.append("• **Familial History Review:** Discuss family metabolic predisposition with a primary care physician.")
            
        for t in tests:
            st.markdown(t)

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.subheader("Exploratory Data Analysis (PIMA Indians Diabetes Cohort)")
    st.markdown("Explore data distributions, class balances, and feature correlations after curation.")
    
    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Cohort Size", f"{len(df_raw)} patients")
    with m2:
        pos_count = (df_raw['Outcome'] == 1).sum()
        st.metric("Diabetic Cases", f"{pos_count} ({pos_count/len(df_raw)*100:.1f}%)")
    with m3:
        neg_count = (df_raw['Outcome'] == 0).sum()
        st.metric("Non-Diabetic Cases", f"{neg_count} ({neg_count/len(df_raw)*100:.1f}%)")
    with m4:
        st.metric("Curated Features", "8 Primary + 3 Engineered")

    st.markdown("---")
    
    eda_sub1, eda_sub2 = st.columns([1, 1])
    
    with eda_sub1:
        st.markdown("#### 🔬 Feature Distribution by Outcome")
        feature_choice = st.selectbox(
            "Select Clinical Feature to Analyze:",
            [c for c in df_curated.columns if c != "Outcome"],
            index=1
        )
        
        plot_type = st.radio("Visualization Style:", ["Box Plot", "Violin Plot", "Histogram / KDE"], horizontal=True)
        
        df_plot = df_curated.copy()
        df_plot["Outcome_Label"] = df_plot["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
        
        if plot_type == "Box Plot":
            fig_dist = px.box(
                df_plot,
                x="Outcome_Label",
                y=feature_choice,
                color="Outcome_Label",
                color_discrete_map={"Non-Diabetic": "#0ea5e9", "Diabetic": "#ef4444"},
                points="all",
                title=f"Distribution of {feature_choice} by Diabetic Outcome"
            )
        elif plot_type == "Violin Plot":
            fig_dist = px.violin(
                df_plot,
                x="Outcome_Label",
                y=feature_choice,
                color="Outcome_Label",
                color_discrete_map={"Non-Diabetic": "#0ea5e9", "Diabetic": "#ef4444"},
                box=True,
                points="all",
                title=f"Violin Distribution of {feature_choice}"
            )
        else:
            fig_dist = px.histogram(
                df_plot,
                x=feature_choice,
                color="Outcome_Label",
                barmode="overlay",
                marginal="box",
                color_discrete_map={"Non-Diabetic": "#0ea5e9", "Diabetic": "#ef4444"},
                title=f"Histogram of {feature_choice}"
            )
            
        fig_dist.update_layout(height=380, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

    with eda_sub2:
        st.markdown("#### 📈 Interactive Correlation Matrix")
        corr_matrix = df_curated.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Coefficients Matrix"
        )
        fig_corr.update_layout(height=420, margin=dict(t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 Multi-dimensional Interaction: Glucose vs. BMI vs. Age")
    
    df_curated_labelled = df_curated.copy()
    df_curated_labelled["Outcome_Label"] = df_curated_labelled["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
    
    fig_scatter = px.scatter(
        df_curated_labelled,
        x="Glucose",
        y="BMI",
        size="Age",
        color="Outcome_Label",
        color_discrete_map={"Non-Diabetic": "#3b82f6", "Diabetic": "#ef4444"},
        hover_data=["Age", "BloodPressure", "Insulin"],
        title="Interactive Patient Clustering: Glucose vs BMI (Bubble size represents Age)"
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("📄 View Curated Dataset Table"):
        st.dataframe(df_curated, use_container_width=True)
        csv_data = df_curated.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Curated CSV Dataset", csv_data, "curated_diabetes_dataset.csv", "text/csv")

# ==========================================
# TAB 3: MODEL PERFORMANCE & ARCHITECTURE
# ==========================================
with tab3:
    st.subheader("Machine Learning Pipeline & Benchmark Metrics")
    st.markdown("Evaluation metrics tested across stratified 5-fold cross-validation and an independent hold-out test set (20%).")

    all_models = artifact["all_model_results"]
    df_bench = pd.DataFrame(all_models).T
    df_bench = df_bench[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV ROC-AUC Mean"]]
    
    st.markdown("#### 🏆 Model Benchmark Comparison")
    st.dataframe(
        df_bench.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "CV ROC-AUC Mean": "{:.4f}"
        }).highlight_max(axis=0, color="#dcfce7"),
        use_container_width=True
    )

    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.markdown("#### 📉 ROC-AUC Curve")
        fpr = artifact["roc_curve"]["fpr"]
        tpr = artifact["roc_curve"]["tpr"]
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Ensemble ROC (AUC = {metrics["ROC-AUC"]:.3f})',
            line=dict(color='#2563eb', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Chance Baseline (AUC = 0.50)',
            line=dict(color='#94a3b8', dash='dash')
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=380,
            margin=dict(t=30, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_eval2:
        st.markdown("#### 🎯 Confusion Matrix (Test Set: N=154)")
        cm_data = artifact["confusion_matrix"]
        
        cm_labels = ["Non-Diabetic (0)", "Diabetic (1)"]
        fig_cm = px.imshow(
            cm_data,
            x=cm_labels,
            y=cm_labels,
            text_auto=True,
            color_continuous_scale="Teal",
            labels=dict(x="Predicted Label", y="True Label")
        )
        fig_cm.update_layout(height=380, margin=dict(t=30, b=30, l=40, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌟 Global Feature Importance (Gradient Boosting Sub-Estimator)")
    
    feat_imp = artifact["feature_importances"]
    df_feat_imp = pd.DataFrame({
        "Feature": list(feat_imp.keys()),
        "Importance": list(feat_imp.values())
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        df_feat_imp,
        x="Importance",
        y="Feature",
        orientation='h',
        color="Importance",
        color_continuous_scale="Purples",
        title="Predictive Weight of Clinical & Engineered Features"
    )
    fig_imp.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🛠️ Data Curation & Preprocessing Methodology")
    st.markdown("""
    1. **Biological Zero Value Imputation:** In clinical datasets, values of `0` in Glucose, Blood Pressure, Skinfold Thickness, Insulin, and BMI represent unrecorded measurements rather than true zeroes. We convert zero values in these columns to `NaN` and impute them using a multivariate **K-Nearest Neighbors (KNN)** imputer ($k=5$).
    2. **Clinical Feature Engineering:**
       - **BMI-Age Metabolic Risk Factor:** $\\text{BMI} \\times \\text{Age} / 100$
       - **Glucose-BMI Interaction:** $\\text{Glucose} \\times \\text{BMI} / 100$
       - **Insulin-to-Glucose Ratio:** $\\text{Insulin} / \\text{Glucose}$ (Insulin resistance proxy)
    3. **Robust Feature Scaling:** Uses median and interquartile ranges via `RobustScaler` to ensure robustness against medical outliers.
    4. **Ensemble Architecture:** Soft-voting combination of **Gradient Boosting (weight=3)**, **Random Forest (weight=2)**, and **L2-Regularized Logistic Regression (weight=1)**.
    """)

# ==========================================
# TAB 4: BATCH PATIENT SCREENING
# ==========================================
with tab4:
    st.subheader("Batch Patient Screening (CSV Upload)")
    st.markdown("Upload a CSV file containing multiple patient records to perform automated screening in bulk.")
    
    # Download sample template
    sample_df = df_raw.head(5).drop("Outcome", axis=1)
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=sample_csv,
        file_name="sample_patient_batch.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} patient records!")
            
            # Verify required columns
            required_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                # Perform predictions
                batch_probs = pipeline.predict_proba(batch_df[required_cols])[:, 1]
                batch_preds = pipeline.predict(batch_df[required_cols])
                
                results_df = batch_df.copy()
                results_df["Diabetes_Risk_Probability"] = [f"{p*100:.1f}%" for p in batch_probs]
                results_df["Risk_Score_Numeric"] = batch_probs
                results_df["Predicted_Class"] = ["Positive (Diabetic)" if p == 1 else "Negative (Non-Diabetic)" for p in batch_preds]
                results_df["Risk_Category"] = [
                    "Low Risk" if p < 0.35 else ("Moderate Risk" if p < 0.65 else "High Risk") for p in batch_probs
                ]
                
                # Summary Cards
                b1, b2, b3 = st.columns(3)
                with b1:
                    low_c = (results_df["Risk_Category"] == "Low Risk").sum()
                    st.metric("Low Risk Patients", f"{low_c} ({low_c/len(results_df)*100:.1f}%)")
                with b2:
                    med_c = (results_df["Risk_Category"] == "Moderate Risk").sum()
                    st.metric("Moderate Risk Patients", f"{med_c} ({med_c/len(results_df)*100:.1f}%)")
                with b3:
                    high_c = (results_df["Risk_Category"] == "High Risk").sum()
                    st.metric("High Risk Patients", f"{high_c} ({high_c/len(results_df)*100:.1f}%)")
                    
                st.markdown("#### 📋 Batch Diagnostic Results")
                
                # Color code display
                st.dataframe(results_df.drop("Risk_Score_Numeric", axis=1), use_container_width=True)
                
                # Download results button
                res_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Diagnostic Screening Report (CSV)",
                    data=res_csv,
                    file_name="diabetes_batch_screening_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
