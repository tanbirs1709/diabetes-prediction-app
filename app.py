import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Import custom transformers so unpickling works seamlessly across environments
from model_utils import BiologicalZeroImputer, MedicalFeatureEngineering

# Page configuration
st.set_page_config(
    page_title="GlucoGuard AI — Smart Diabetes Risk Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Modern Glassmorphic Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 32px 36px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.25) 0%, rgba(20, 184, 166, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-top: 8px;
        max-width: 800px;
        line-height: 1.5;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.12);
        color: #f1f5f9;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Cards & Containers */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    
    /* Result Banners */
    .risk-banner {
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px -6px rgba(0,0,0,0.15);
    }
    .risk-banner-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    .risk-banner-med {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    .risk-banner-high {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-left: 8px;
    }
    .badge-normal {
        background-color: #d1fae5;
        color: #065f46;
    }
    .badge-borderline {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-elevated {
        background-color: #fee2e2;
        color: #991b1b;
    }

    /* Educational Concept Cards */
    .edu-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border-left: 5px solid #0d9488;
    }
    .edu-card h4 {
        margin: 0 0 6px 0;
        color: #0f172a;
        font-size: 1.05rem;
    }
    .edu-card p {
        margin: 0;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Action Tip Items */
    .tip-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 14px;
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .tip-icon {
        font-size: 1.3rem;
        line-height: 1;
    }
    .tip-text {
        font-size: 0.92rem;
        color: #334155;
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

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩺 GlucoGuard AI</div>
    <div class="hero-subtitle">
        An intelligent, explainable health platform that translates clinical data into clear, actionable diabetes risk insights for everyday individuals and healthcare teams.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">✨ Instant Risk Analysis</span>
        <span class="hero-pill">🧠 AI Soft-Voting Ensemble (83.9% CV ROC-AUC)</span>
        <span class="hero-pill">🌱 Plain-English Explanations</span>
        <span class="hero-pill">🔬 KNN Biological Imputation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Preset Profiles & Helpers
st.sidebar.markdown("### 📋 Quick Patient Profiles")
st.sidebar.caption("Select a sample profile to instantly see how the AI evaluates different health situations:")

preset = st.sidebar.selectbox(
    "Choose a sample profile:",
    [
        "Custom Input (Manual)",
        "🟢 Healthy Young Adult (Low Risk)",
        "🟡 Middle-Aged Borderline (Moderate Risk)",
        "🔴 High-Risk Metabolic Patient (High Risk)"
    ]
)

presets = {
    "🟢 Healthy Young Adult (Low Risk)": {
        "Pregnancies": 0, "Glucose": 88.0, "BloodPressure": 68.0,
        "SkinThickness": 18.0, "Insulin": 65.0, "BMI": 21.5,
        "DiabetesPedigreeFunction": 0.22, "Age": 24
    },
    "🟡 Middle-Aged Borderline (Moderate Risk)": {
        "Pregnancies": 2, "Glucose": 128.0, "BloodPressure": 82.0,
        "SkinThickness": 28.0, "Insulin": 135.0, "BMI": 28.4,
        "DiabetesPedigreeFunction": 0.48, "Age": 42
    },
    "🔴 High-Risk Metabolic Patient (High Risk)": {
        "Pregnancies": 6, "Glucose": 175.0, "BloodPressure": 92.0,
        "SkinThickness": 39.0, "Insulin": 240.0, "BMI": 38.6,
        "DiabetesPedigreeFunction": 0.85, "Age": 54
    }
}

defaults = presets.get(preset, {
    "Pregnancies": 1, "Glucose": 110.0, "BloodPressure": 75.0,
    "SkinThickness": 24.0, "Insulin": 95.0, "BMI": 26.0,
    "DiabetesPedigreeFunction": 0.35, "Age": 33
})

# Sidebar View Mode Toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Interface Mode")
view_mode = st.sidebar.radio(
    "Label Terminology:",
    ["🌱 Everyday / Beginner-Friendly", "🔬 Clinical / Lab Values"],
    index=0
)
is_simple = (view_mode == "🌱 Everyday / Beginner-Friendly")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 Quick BMI Calculator")
with st.sidebar.expander("Don't know your BMI? Calculate it here", expanded=False):
    unit_choice = st.radio("Units:", ["Metric (cm, kg)", "Imperial (ft/in, lbs)"], horizontal=True)
    if "Metric" in unit_choice:
        calc_h = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
        calc_w = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        calc_bmi = round(calc_w / ((calc_h / 100) ** 2), 1)
    else:
        ft = st.number_input("Height (Feet)", min_value=3, max_value=7, value=5)
        inches = st.number_input("Height (Inches)", min_value=0, max_value=11, value=8)
        total_in = ft * 12 + inches
        calc_w_lbs = st.number_input("Weight (lbs)", min_value=60.0, max_value=450.0, value=155.0, step=1.0)
        calc_bmi = round((calc_w_lbs / (total_in ** 2)) * 703, 1) if total_in > 0 else 22.0
    
    bmi_category = "Normal Weight" if calc_bmi < 25 else ("Overweight" if calc_bmi < 30 else "Obese")
    st.info(f"💡 **Calculated BMI: {calc_bmi}** ({bmi_category})\n*Use this value in the BMI slider!*")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ About GlucoGuard")
st.sidebar.markdown("""
- **AI Core:** Soft-voting ensemble combining Logistic Regression, Random Forest, and Gradient Boosting.
- **Accuracy:** Tested on clinical PIMA cohort with 5-fold cross validation.
- **Data Safety:** Zero personal data is stored or transmitted.
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 1. Smart Risk Assessment",
    "💡 2. Understand Your Numbers",
    "🔮 3. 'What-If?' Simulator",
    "📊 4. Population Trends & EDA",
    "🧠 5. AI Architecture & Batch Tool"
])

# ==========================================
# TAB 1: SMART RISK ASSESSMENT
# ==========================================
with tab1:
    st.markdown("### 📝 Enter Health Metrics")
    st.markdown("Adjust the sliders below based on your most recent routine checkup or lab numbers:")

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🩸 Blood & Metabolic Measurements")
        
        # 1. Glucose
        g_label = "Blood Sugar Level (Glucose)" if is_simple else "Plasma Glucose Concentration (2h OGTT)"
        g_help = "Blood sugar after fasting or standard 2-hr test. Normal fasting is under 100 mg/dL. Over 140 suggests elevated risk."
        glucose = st.slider(
            g_label,
            min_value=50.0, max_value=220.0,
            value=float(defaults["Glucose"]), step=1.0,
            help=g_help
        )
        if glucose < 100:
            st.markdown("🟢 **Status:** Normal Healthy Range (< 100 mg/dL)")
        elif glucose <= 139:
            st.markdown("🟡 **Status:** Borderline / Impaired Fasting (100 - 139 mg/dL)")
        else:
            st.markdown("🔴 **Status:** High / Diabetic Range (≥ 140 mg/dL)")
        
        # 2. Insulin
        ins_label = "Insulin Level" if is_simple else "2-Hour Serum Insulin (µU/mL)"
        ins_help = "Hormone that clears sugar from blood. Higher numbers often mean insulin resistance (body working overtime)."
        insulin = st.slider(
            ins_label,
            min_value=15.0, max_value=400.0,
            value=float(defaults["Insulin"]), step=5.0,
            help=ins_help
        )
        if insulin < 100:
            st.markdown("🟢 **Status:** Optimal Insulin Sensitivity (< 100 µU/mL)")
        elif insulin <= 166:
            st.markdown("🟡 **Status:** Moderate / Average Range (100 - 166 µU/mL)")
        else:
            st.markdown("🔴 **Status:** High / Potential Insulin Resistance (> 166 µU/mL)")

        # 3. Blood Pressure
        bp_label = "Resting Blood Pressure (Bottom / Diastolic Number)" if is_simple else "Diastolic Blood Pressure (mm Hg)"
        bp_help = "The pressure in your arteries when your heart rests between beats. Normal is between 60 and 80 mm Hg."
        blood_pressure = st.slider(
            bp_label,
            min_value=40.0, max_value=125.0,
            value=float(defaults["BloodPressure"]), step=1.0,
            help=bp_help
        )
        if blood_pressure <= 80:
            st.markdown("🟢 **Status:** Ideal Blood Pressure (≤ 80 mm Hg)")
        elif blood_pressure <= 89:
            st.markdown("🟡 **Status:** Pre-hypertension (81 - 89 mm Hg)")
        else:
            st.markdown("🔴 **Status:** Elevated / High Blood Pressure (≥ 90 mm Hg)")

        # 4. Skinfold / Body Fat
        skin_label = "Subcutaneous Body Fat Indicator (Skinfold Thickness)" if is_simple else "Triceps Skin Fold Thickness (mm)"
        skin_help = "Measurement of skin fold thickness in millimeters, used as a proxy for body fat distribution."
        skin_thickness = st.slider(
            skin_label,
            min_value=5.0, max_value=60.0,
            value=float(defaults["SkinThickness"]), step=1.0,
            help=skin_help
        )
        if skin_thickness <= 28:
            st.markdown("🟢 **Status:** Standard / Lean (≤ 28 mm)")
        else:
            st.markdown("🟡 **Status:** Higher Subcutaneous Fat (> 28 mm)")

    with col_right:
        st.markdown("#### 👤 Physical & Demographic Profile")
        
        # 5. BMI
        bmi_label = "Body Mass Index (BMI)" if is_simple else "Body Mass Index (kg/m²)"
        bmi_help = "Ratio of weight to height. 18.5 - 24.9 is healthy. Over 25 is overweight, and over 30 is classified as obesity."
        bmi = st.slider(
            bmi_label,
            min_value=15.0, max_value=50.0,
            value=float(defaults["BMI"]), step=0.1,
            help=bmi_help
        )
        if bmi < 25.0:
            st.markdown("🟢 **Status:** Healthy Weight (BMI 18.5 - 24.9)")
        elif bmi < 30.0:
            st.markdown("🟡 **Status:** Overweight Range (BMI 25.0 - 29.9)")
        else:
            st.markdown("🔴 **Status:** Obese Range (BMI ≥ 30.0)")

        # 6. Age
        age_label = "Your Age (Years)" if is_simple else "Age (Years)"
        age = st.slider(
            age_label,
            min_value=18, max_value=90,
            value=int(defaults["Age"]), step=1,
            help="Diabetes risk increases with age due to natural metabolic deceleration."
        )
        if age < 35:
            st.markdown("🟢 **Age Group:** Young Adult (< 35)")
        elif age <= 50:
            st.markdown("🟡 **Age Group:** Middle Age (35 - 50)")
        else:
            st.markdown("🟠 **Age Group:** Senior / Higher Baseline Risk (> 50)")

        # 7. Pregnancies
        preg_label = "Number of Pregnancies" if is_simple else "Number of Pregnancies"
        pregnancies = st.slider(
            preg_label,
            min_value=0, max_value=17,
            value=int(defaults["Pregnancies"]), step=1,
            help="Relevant for gestational diabetes history in female cohorts. Set to 0 if not applicable."
        )
        st.caption("ℹ️ *Indicator for gestational metabolic history.*")

        # 8. Genetics / DPF
        dpf_label = "Family Diabetes History Score" if is_simple else "Diabetes Pedigree Function"
        dpf_help = "A score estimating genetic predisposition from family history. 0.1 - 0.4 = low family history, > 0.8 = strong immediate family history."
        dpf = st.slider(
            dpf_label,
            min_value=0.08, max_value=2.40,
            value=float(defaults["DiabetesPedigreeFunction"]), step=0.01,
            help=dpf_help
        )
        if dpf < 0.40:
            st.markdown("🟢 **Family History:** Low / Minimal Relatives Affected (< 0.40)")
        elif dpf <= 0.75:
            st.markdown("🟡 **Family History:** Moderate Genetic Factor (0.40 - 0.75)")
        else:
            st.markdown("🔴 **Family History:** Strong Family Predisposition (> 0.75)")

    st.markdown("---")
    
    # Calculate Live Prediction
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

    # Result Section
    res_col1, res_col2 = st.columns([1.1, 1.4])
    
    with res_col1:
        st.markdown("### 🎯 Your AI Risk Score")
        
        # Color definition
        if risk_pct < 35:
            gauge_color = "#10b981"
            risk_title = "Low Diabetes Risk"
            banner_class = "risk-banner-low"
            risk_summary = "Your biomarkers are largely in a safe, healthy range. Maintaining regular exercise and balanced nutrition will help preserve this metabolic health."
        elif risk_pct < 65:
            gauge_color = "#f59e0b"
            risk_title = "Moderate / Borderline Risk"
            banner_class = "risk-banner-med"
            risk_summary = "Certain biomarkers (such as blood sugar or BMI) are in an elevated caution zone. Proactive lifestyle tweaks can significantly lower your risk."
        else:
            gauge_color = "#ef4444"
            risk_title = "High Risk Detected"
            banner_class = "risk-banner-high"
            risk_summary = "Multiple key indicators suggest strong metabolic stress. We recommend discussing these numbers with a healthcare provider for routine screening."

        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix': "%", 'font': {'size': 48, 'color': gauge_color, 'family': 'Plus Jakarta Sans, sans-serif'}},
            title={'text': "Calculated Diabetes Risk", 'font': {'size': 17, 'color': '#1e293b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "#f8fafc",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.18)'},
                    {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.18)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.18)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 65
                }
            }
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=25, r=25, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Plus Jakarta Sans, sans-serif'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="risk-banner {banner_class}">
            <h3 style="margin: 0; font-size: 1.3rem;">{risk_title} ({risk_pct}%)</h3>
            <p style="margin: 8px 0 0 0; font-size: 0.95rem; opacity: 0.95;">{risk_summary}</p>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("### 🔍 Key Factors & Actionable Steps")
        
        # Explain top contributors in plain English
        st.markdown("#### 🧭 What is driving this score?")
        
        drivers = []
        if glucose >= 140:
            drivers.append(("🔴 Blood Sugar (Glucose)", f"Elevated at {glucose:.0f} mg/dL (Major factor)"))
        elif glucose >= 100:
            drivers.append(("🟡 Blood Sugar (Glucose)", f"Borderline at {glucose:.0f} mg/dL"))
            
        if bmi >= 30:
            drivers.append(("🔴 Body Mass Index (BMI)", f"High at {bmi:.1f} kg/m²"))
        elif bmi >= 25:
            drivers.append(("🟡 Body Mass Index (BMI)", f"Mildly elevated at {bmi:.1f} kg/m²"))
            
        if insulin > 166:
            drivers.append(("🔴 Insulin Output", f"High at {insulin:.0f} µU/mL (Insulin Resistance proxy)"))
            
        if dpf > 0.7:
            drivers.append(("🟡 Genetic Predisposition", f"Family history factor ({dpf:.2f}) adds baseline susceptibility"))
            
        if not drivers:
            drivers.append(("🟢 All Primary Markers", "All main indicators are within healthy standard ranges!"))
            
        for factor, desc in drivers:
            st.markdown(f"- **{factor}:** {desc}")
            
        st.markdown("#### 💡 Recommended Next Actions")
        if risk_pct < 35:
            st.markdown("""
            <div class="tip-item"><span class="tip-icon">🥗</span><span class="tip-text"><b>Maintain Balanced Diet:</b> Focus on whole foods, fiber-rich vegetables, and lean proteins to sustain insulin sensitivity.</span></div>
            <div class="tip-item"><span class="tip-icon">🏃</span><span class="tip-text"><b>Stay Active:</b> Aim for at least 150 minutes of moderate aerobic activity weekly.</span></div>
            <div class="tip-item"><span class="tip-icon">🩺</span><span class="tip-text"><b>Annual Checkup:</b> Keep up with routine yearly wellness lab screenings.</span></div>
            """, unsafe_allow_html=True)
        elif risk_pct < 65:
            st.markdown("""
            <div class="tip-item"><span class="tip-icon">📉</span><span class="tip-text"><b>Reduce Refined Sugars:</b> Replace high-glycemic carbohydrates with whole grains and leafy greens to lower glucose spikes.</span></div>
            <div class="tip-item"><span class="tip-icon">🚶</span><span class="tip-text"><b>Post-Meal Walks:</b> A 15-minute walk after meals helps muscles absorb blood glucose naturally.</span></div>
            <div class="tip-item"><span class="tip-icon">🩺</span><span class="tip-text"><b>Consider HbA1c Lab Test:</b> Schedule a routine primary care visit to confirm 3-month blood sugar trends.</span></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-item"><span class="tip-icon">👨⚕️</span><span class="tip-text"><b>Consult a Doctor:</b> Bring these numbers to a physician or endocrinologist for comprehensive diagnostic evaluation.</span></div>
            <div class="tip-item"><span class="tip-icon">📊</span><span class="tip-text"><b>Diagnostic Blood Panel:</b> Request an official Fasting Blood Glucose (FBG) and HbA1c test.</span></div>
            <div class="tip-item"><span class="tip-icon">🥦</span><span class="tip-text"><b>Targeted Nutritional Plan:</b> Work with a registered dietitian on a low-glycemic, anti-inflammatory dietary framework.</span></div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: UNDERSTAND YOUR NUMBERS
# ==========================================
with tab2:
    st.markdown("### 💡 Plain-English Guide: What Do These Numbers Mean?")
    st.markdown("Medical lab tests can often feel overwhelming. Here is a simple, no-jargon breakdown of each biomarker:")

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div class="edu-card">
            <h4>🩸 1. Blood Sugar (Glucose)</h4>
            <p><b>What it is:</b> The amount of sugar floating in your bloodstream waiting to be delivered to your cells for fuel.<br>
            <b>Healthy Target:</b> Under 100 mg/dL (fasting) or under 140 mg/dL (2 hours after food).<br>
            <b>Why it matters:</b> Consistently high blood sugar can damage blood vessels, nerves, kidneys, and eyesight over time.</p>
        </div>
        
        <div class="edu-card">
            <h4>⚖️ 2. Body Mass Index (BMI)</h4>
            <p><b>What it is:</b> A formula comparing your weight to your height to gauge body size.<br>
            <b>Healthy Target:</b> 18.5 – 24.9 kg/m².<br>
            <b>Why it matters:</b> Excess visceral fat around abdominal organs releases inflammatory compounds that make it harder for insulin to work.</p>
        </div>

        <div class="edu-card">
            <h4>🧬 3. Family History (Pedigree Function)</h4>
            <p><b>What it is:</b> A mathematical score that reflects how many immediate relatives (parents, siblings, grandparents) have had diabetes.<br>
            <b>Typical Value:</b> 0.1 – 0.5 (Average), > 0.8 (Strong Family History).<br>
            <b>Why it matters:</b> Genetics set your baseline susceptibility, but lifestyle choices can often prevent genes from expressing disease.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="edu-card">
            <h4>🔑 4. Insulin</h4>
            <p><b>What it is:</b> The 'key' hormone made by your pancreas that opens up your cells so glucose can get inside for energy.<br>
            <b>Healthy Target:</b> 16 – 100 µU/mL.<br>
            <b>Why it matters:</b> When cells stop responding (Insulin Resistance), your pancreas pumps out extra insulin to compensate, driving numbers up.</p>
        </div>

        <div class="edu-card">
            <h4>💓 5. Blood Pressure</h4>
            <p><b>What it is:</b> The resting pressure exerted against your artery walls between heartbeats.<br>
            <b>Healthy Target:</b> 60 – 80 mm Hg.<br>
            <b>Why it matters:</b> High blood pressure frequently coexists with insulin resistance as part of metabolic syndrome.</p>
        </div>

        <div class="edu-card">
            <h4>📏 6. Skinfold Thickness</h4>
            <p><b>What it is:</b> A pinch caliper measurement of body fat beneath the skin.<br>
            <b>Healthy Target:</b> 10 – 28 mm.<br>
            <b>Why it matters:</b> Provides an estimation of subcutaneous adipose tissue distribution across the body.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Quick Clinical Reference Grid")
    
    ref_df = pd.DataFrame([
        {"Biomarker": "Blood Sugar (Glucose)", "Healthy Range": "70 – 99 mg/dL", "Caution / Borderline": "100 – 139 mg/dL", "High Risk": "≥ 140 mg/dL"},
        {"Biomarker": "Body Mass Index (BMI)", "Healthy Range": "18.5 – 24.9 kg/m²", "Caution / Borderline": "25.0 – 29.9 kg/m²", "High Risk": "≥ 30.0 kg/m²"},
        {"Biomarker": "Resting Blood Pressure", "Healthy Range": "60 – 80 mm Hg", "Caution / Borderline": "81 – 89 mm Hg", "High Risk": "≥ 90 mm Hg"},
        {"Biomarker": "Insulin", "Healthy Range": "16 – 100 µU/mL", "Caution / Borderline": "101 – 166 µU/mL", "High Risk": "> 166 µU/mL"},
        {"Biomarker": "Family History Score", "Healthy Range": "< 0.40", "Caution / Borderline": "0.40 – 0.75", "High Risk": "> 0.75"}
    ])
    st.table(ref_df)

# ==========================================
# TAB 3: "WHAT-IF" SCENARIO SIMULATOR
# ==========================================
with tab3:
    st.markdown("### 🔮 Interactive 'What-If?' Lifestyle Simulator")
    st.markdown("See in real time how making positive lifestyle changes (like lowering your blood sugar or managing weight) directly reduces your AI diabetes risk score:")

    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.markdown("#### 🎯 Baseline Values (From Tab 1)")
        st.info(f"**Current Blood Sugar:** {glucose:.0f} mg/dL  \n**Current BMI:** {bmi:.1f} kg/m²  \n**Current Baseline AI Risk:** **{risk_pct}%**")
        
        st.markdown("#### 🛠️ Simulate Health Improvements")
        target_glucose = st.slider(
            "Simulated Blood Sugar (Glucose):",
            min_value=70.0, max_value=float(max(glucose, 150.0)),
            value=float(min(glucose, 95.0)), step=1.0
        )
        target_bmi = st.slider(
            "Simulated BMI:",
            min_value=18.5, max_value=float(max(bmi, 32.0)),
            value=float(min(bmi, 23.5)), step=0.1
        )
        target_insulin = st.slider(
            "Simulated Insulin Output:",
            min_value=20.0, max_value=float(max(insulin, 200.0)),
            value=float(min(insulin, 75.0)), step=5.0
        )

    with sim_col2:
        st.markdown("#### 📉 Projected Risk Impact")
        
        sim_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": target_glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": target_insulin,
            "BMI": target_bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])
        
        sim_prob = float(pipeline.predict_proba(sim_input)[0][1])
        sim_pct = round(sim_prob * 100, 1)
        risk_diff = round(risk_pct - sim_pct, 1)

        # Comparison Bar Chart
        fig_comp = go.Figure(data=[
            go.Bar(name='Current Risk', x=['Risk Score'], y=[risk_pct], marker_color='#ef4444', text=f"{risk_pct}%", textposition='auto'),
            go.Bar(name='Simulated Goal Risk', x=['Risk Score'], y=[sim_pct], marker_color='#10b981', text=f"{sim_pct}%", textposition='auto')
        ])
        fig_comp.update_layout(
            barmode='group',
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 100], title="Probability (%)"),
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        if risk_diff > 0:
            st.success(f"🎉 **Huge Positive Shift!** By reaching your target blood sugar of **{target_glucose:.0f} mg/dL** and BMI of **{target_bmi:.1f}**, your estimated diabetes risk drops by **-{risk_diff}%** (from {risk_pct}% down to {sim_pct}%).")
        elif risk_diff < 0:
            st.warning(f"⚠️ **Caution:** Increasing these values would increase risk by **+{abs(risk_diff)}%**.")
        else:
            st.info("Adjust the sliders on the left to see potential risk reduction!")

# ==========================================
# TAB 4: POPULATION TRENDS & EDA
# ==========================================
with tab4:
    st.markdown("### 📊 Interactive Population Insights & Exploratory Data Analysis")
    st.markdown("Explore trends and distributions from the curated clinical cohort (768 patients):")

    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### 1. Blood Sugar vs BMI Distribution")
        df_plot = df_curated.copy()
        df_plot["Patient Outcome"] = df_plot["Outcome"].apply(lambda x: "Diabetic" if x == 1 else "Healthy / Non-Diabetic")
        fig_scatter = px.scatter(
            df_plot,
            x="Glucose",
            y="BMI",
            color="Patient Outcome",
            color_discrete_map={"Healthy / Non-Diabetic": "#10b981", "Diabetic": "#ef4444"},
            hover_data=["Age", "Insulin"],
            labels={"Patient Outcome": "Diagnosis"},
            template="plotly_white"
        )
        fig_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("💡 **Takeaway:** Patients with both Glucose > 130 and BMI > 30 represent the vast majority of positive diagnoses.")

    with eda_col2:
        st.markdown("#### 2. Risk by Age Category")
        df_age = df_curated.copy()
        df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=[18, 30, 45, 60, 90], labels=["18-30", "31-45", "46-60", "60+"])
        age_grouped = df_age.groupby("AgeGroup", observed=False)["Outcome"].mean()
        age_risk = pd.DataFrame({
            "AgeGroup": [str(k) for k in age_grouped.index],
            "DiabetesRate": [float(v) * 100 for v in age_grouped.values]
        })
        
        fig_age = px.bar(
            age_risk,
            x="AgeGroup",
            y="DiabetesRate",
            color="DiabetesRate",
            color_continuous_scale="Tealgrn",
            labels={"DiabetesRate": "Diabetes Rate (%)", "AgeGroup": "Age Bracket"},
            template="plotly_white"
        )
        fig_age.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("💡 **Takeaway:** The prevalence of diabetes rises sharply in the 31-45 and 46-60 age brackets.")

    st.markdown("---")
    st.markdown("#### 3. Full Feature Correlation Matrix")
    corr = df_curated.corr().round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        template="plotly_white"
    )
    fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 5: AI ARCHITECTURE & BATCH TOOL
# ==========================================
with tab5:
    st.markdown("### 🧠 Machine Learning Pipeline & Technical Architecture")
    
    col_tech1, col_tech2 = st.columns([1, 1])
    
    with col_tech1:
        st.markdown("#### 🏆 Model Benchmark Comparison")
        st.markdown("All models evaluated using **5-Fold Stratified Cross-Validation**:")
        
        benchmark_df = pd.DataFrame([
            {"Model Architecture": "Logistic Regression (L2, C=1.0)", "CV ROC-AUC": "84.47% (±1.65%)", "Test ROC-AUC": "81.50%", "Test F1": "0.549"},
            {"Model Architecture": "Random Forest (150 trees, depth=6)", "CV ROC-AUC": "83.74% (±1.99%)", "Test ROC-AUC": "80.94%", "Test F1": "0.540"},
            {"Model Architecture": "Gradient Boosting (lr=0.05, depth=3)", "CV ROC-AUC": "82.22% (±2.56%)", "Test ROC-AUC": "82.48%", "Test F1": "0.621"},
            {"Model Architecture": "⭐ Production Soft-Voting Ensemble", "CV ROC-AUC": "83.91% (±2.03%)", "Test ROC-AUC": "82.07%", "Test F1": "0.574"}
        ])
        st.table(benchmark_df)
        
        st.markdown("#### 🧬 Biological Zero Imputation Innovation")
        st.info("""
        In clinical PIMA data, `0` values in Glucose, Blood Pressure, Skinfold, Insulin, and BMI are physiologically impossible and represent unrecorded tests. 
        Instead of basic mean imputation which destroys correlation structure, our pipeline applies **Multivariate KNN Imputation ($k=5$)** directly inside the Scikit-learn Pipeline.
        """)

    with col_tech2:
        st.markdown("#### 📈 Model ROC-AUC Performance Curve")
        roc_data = artifact.get("roc_curve", {})
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"], y=roc_data["tpr"],
                mode='lines',
                name=f'Ensemble (AUC = {metrics.get("roc_auc", 0.82):.3f})',
                line=dict(color='#0d9488', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Chance Baseline',
                line=dict(color='#94a3b8', dash='dash')
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                template="plotly_white"
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📁 Batch Patient Screening Tool")
    st.markdown("Upload a multi-patient CSV file to perform bulk AI screening and download enriched risk reports:")
    
    uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])
    
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        if st.button("📥 Load Sample Multi-Patient CSV"):
            sample_df = df_raw.head(10).drop(columns=["Outcome"], errors="ignore")
            st.session_state["batch_data"] = sample_df
            st.success("Loaded 10 sample patient records!")
            
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.session_state["batch_data"] = batch_df

    if "batch_data" in st.session_state:
        df_to_predict = st.session_state["batch_data"].copy()
        
        # Predict on batch
        probs = pipeline.predict_proba(df_to_predict)[:, 1]
        preds = pipeline.predict(df_to_predict)
        
        df_to_predict["Diabetes_Risk_Prob (%)"] = [round(p * 100, 1) for p in probs]
        df_to_predict["Predicted_Class"] = ["Diabetic Risk" if p == 1 else "Low Risk / Negative" for p in preds]
        df_to_predict["Risk_Category"] = ["Low" if p < 0.35 else ("Moderate" if p < 0.65 else "High") for p in probs]
        
        st.markdown("#### 📋 Batch Screening Results")
        st.dataframe(df_to_predict, use_container_width=True)
        
        csv_export = df_to_predict.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Enriched Screening Report (CSV)",
            data=csv_export,
            file_name="glucoguard_batch_screening_results.csv",
            mime="text/csv"
        )
