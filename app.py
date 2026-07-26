"""
GlucoGuide — AI-Based Diabetes Prediction and Personalized Risk Analysis
=========================================================================

A production-ready Streamlit application that loads a pre-trained XGBoost
multiclass classifier and its accompanying preprocessing pipeline to predict
diabetes status (No Diabetes / Pre-Diabetes / Type 1 / Type 2 / Gestational)
from patient clinical data, and presents a personalized, visually rich risk
report.

Run with:
    streamlit run app.py

Author : GlucoGuide Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

APP_TITLE = "GlucoGuide"
APP_ICON = "🩺"
MODEL_VERSION = "v1.0.0 (XGBoost Multiclass)"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
HERO_IMAGE_PATH = BASE_DIR / "assets" / "hero-illustration.jpg"

# Diabetes classes that should be treated as "elevated risk" for UI coloring.
# Everything else that is not "No Diabetes" is considered elevated risk.
LOW_RISK_CLASS = "No Diabetes"
MODERATE_RISK_CLASS = "Pre-Diabetes"

# Color palette — professional healthcare blue / cyan / white
COLOR_PRIMARY = "#0F6DBF"
COLOR_PRIMARY_DARK = "#0A4E8A"
COLOR_ACCENT = "#00C2CB"
COLOR_SUCCESS = "#1FAA59"
COLOR_WARNING = "#F2994A"
COLOR_DANGER = "#EB5757"
COLOR_BG_LIGHT = "#F4F9FC"

RISK_COLOR_MAP = {
    LOW_RISK_CLASS: COLOR_SUCCESS,
    MODERATE_RISK_CLASS: COLOR_WARNING,
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PatientInput:
    """Structured container for all clinical inputs collected from the user."""

    age: int
    gender: str
    bmi: float
    height_cm: float
    weight_kg: float
    hba1c: float
    glucose_fasting: float
    glucose_postprandial: float
    cholesterol_total: float
    systolic_bp: float
    family_history_diabetes: int
    physical_activity_minutes_per_week: float
    insulin_level: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the patient input into a single-row DataFrame matching the
        exact column order the preprocessor was fitted on."""
        return pd.DataFrame(
            [
                {
                    "age": self.age,
                    "gender": self.gender,
                    "bmi": self.bmi,
                    "hba1c": self.hba1c,
                    "glucose_fasting": self.glucose_fasting,
                    "glucose_postprandial": self.glucose_postprandial,
                    "cholesterol_total": self.cholesterol_total,
                    "systolic_bp": self.systolic_bp,
                    "family_history_diabetes": self.family_history_diabetes,
                    "physical_activity_minutes_per_week": self.physical_activity_minutes_per_week,
                    "insulin_level": self.insulin_level,
                }
            ]
        )


@dataclass
class PredictionResult:
    """Holds the outcome of a model inference call."""

    predicted_class: str
    confidence: float  # probability of the predicted class, 0-1
    class_probabilities: dict[str, float]

    @property
    def is_low_risk(self) -> bool:
        return self.predicted_class == LOW_RISK_CLASS

    @property
    def is_moderate_risk(self) -> bool:
        return self.predicted_class == MODERATE_RISK_CLASS

    @property
    def risk_color(self) -> str:
        return RISK_COLOR_MAP.get(self.predicted_class, COLOR_DANGER)

    @property
    def risk_label(self) -> str:
        if self.is_low_risk:
            return "Low Risk"
        if self.is_moderate_risk:
            return "Moderate Risk"
        return "High Risk"


# ==============================================================================
# RESOURCE LOADING (cached, with graceful error handling)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Optional[object], Optional[object], Optional[object], Optional[str]]:
    """Load the trained model, preprocessor, and label encoder from disk.

    Returns
    -------
    (model, preprocessor, label_encoder, error_message)
    error_message is None if every artifact loaded successfully.
    """
    missing = [
        str(p.name)
        for p in (MODEL_PATH, PREPROCESSOR_PATH, LABEL_ENCODER_PATH)
        if not p.exists()
    ]
    if missing:
        return None, None, None, f"Missing required file(s): {', '.join(missing)}"

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, preprocessor, label_encoder, None
    except Exception as exc:  # noqa: BLE001 — surface any loading failure to the UI
        return None, None, None, f"Failed to load model artifacts: {exc}"


@st.cache_data(show_spinner=False)
def get_image_base64(path: Path) -> Optional[str]:
    """Read an image file from disk and return it as a base64-encoded string,
    so it can be embedded directly inside styled HTML (e.g. for drop-shadow
    filters that st.image cannot apply). Returns None if the file is missing.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def run_prediction(patient: PatientInput) -> PredictionResult:
    """Run the full inference pipeline: preprocess → predict → decode.

    Raises
    ------
    RuntimeError if artifacts are unavailable or inference fails.
    """
    model, preprocessor, label_encoder, error = load_artifacts()
    if error is not None:
        raise RuntimeError(error)

    try:
        raw_df = patient.to_dataframe()
        transformed = preprocessor.transform(raw_df)
        probabilities = model.predict_proba(transformed)[0]
        predicted_index = int(probabilities.argmax())
        predicted_class = str(label_encoder.inverse_transform([predicted_index])[0])
        class_probabilities = {
            str(label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(probabilities)
        }
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=float(probabilities[predicted_index]),
            class_probabilities=class_probabilities,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Prediction failed due to invalid input or model error: {exc}") from exc


# ==============================================================================
# STYLING
# ==============================================================================

def inject_custom_css() -> None:
    """Inject all custom CSS for the premium healthcare SaaS look and feel."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* ---------- Global App Background ---------- */
        .stApp {{
            background: linear-gradient(160deg, {COLOR_BG_LIGHT} 0%, #E7F3FB 45%, #E1F7F8 100%);
        }}

        /* Hide default Streamlit chrome for a cleaner SaaS feel */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent !important;}}

        /* ---------- Hero Section ---------- */
        .hero-container {{
            background: linear-gradient(120deg, {COLOR_PRIMARY_DARK} 0%, {COLOR_PRIMARY} 55%, {COLOR_ACCENT} 100%);
            border-radius: 28px;
            padding: 3rem 3rem;
            color: white;
            box-shadow: 0 20px 45px rgba(15, 109, 191, 0.25);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-container::before {{
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: rgba(255,255,255,0.08);
        }}
        .hero-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            letter-spacing: -1px;
        }}
        .hero-subtitle {{
            font-family: 'Poppins', sans-serif;
            font-size: 1.35rem;
            font-weight: 600;
            opacity: 0.95;
            margin-bottom: 1rem;
            line-height: 1.4;
        }}
        .hero-desc {{
            font-size: 1rem;
            font-weight: 400;
            opacity: 0.9;
            max-width: 560px;
            line-height: 1.6;
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 6px 16px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
            backdrop-filter: blur(6px);
        }}

        /* ---------- Glass Card ---------- */
        .glass-card {{
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.6);
            border-radius: 22px;
            padding: 2rem 2.2rem;
            box-shadow: 0 12px 32px rgba(15, 60, 110, 0.10);
            margin-bottom: 1.5rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 18px 40px rgba(15, 60, 110, 0.15);
        }}

        .section-heading {{
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: {COLOR_PRIMARY_DARK};
            margin-bottom: 0.3rem;
        }}
        .section-subheading {{
            color: #33424F;
            font-size: 0.95rem;
            font-weight: 500;
            margin-bottom: 1.4rem;
        }}

        /* ---------- Result Cards ---------- */
        .result-card {{
            border-radius: 24px;
            padding: 2.2rem 2.4rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            animation: fadeInUp 0.6s ease;
        }}
        .result-card.low-risk {{
            background: linear-gradient(135deg, #E7F9EF 0%, #D3F3E0 100%);
            border: 1px solid {COLOR_SUCCESS};
        }}
        .result-card.moderate-risk {{
            background: linear-gradient(135deg, #FFF3E4 0%, #FFE8CC 100%);
            border: 1px solid {COLOR_WARNING};
        }}
        .result-card.high-risk {{
            background: linear-gradient(135deg, #FDECEC 0%, #FBDADA 100%);
            border: 1px solid {COLOR_DANGER};
        }}

        .result-icon {{
            font-size: 3.2rem;
            line-height: 1;
        }}
        .result-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0.4rem 0 0.2rem 0;
            color: #1A2733;
        }}
        .result-card.low-risk .result-title {{ color: #0F5C33; }}
        .result-card.moderate-risk .result-title {{ color: #8A4B0E; }}
        .result-card.high-risk .result-title {{ color: #8C1F1F; }}
        .result-subtitle {{
            font-size: 1rem;
            color: #2E3B4A;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ---------- Recommendation Chips ---------- */
        .rec-item {{
            background: rgba(255,255,255,0.75);
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
            font-weight: 500;
            color: #2E3B4A;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 4px solid {COLOR_PRIMARY};
            transition: transform 0.2s ease;
        }}
        .rec-item:hover {{
            transform: translateX(4px);
        }}

        /* ---------- Buttons ---------- */
        div.stButton > button {{
            background: linear-gradient(120deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.75rem 2.2rem;
            border-radius: 14px;
            border: none;
            box-shadow: 0 10px 24px rgba(15, 109, 191, 0.35);
            transition: all 0.2s ease-in-out;
            width: 100%;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 14px 30px rgba(15, 109, 191, 0.45);
            color: white;
        }}
        div.stButton > button:active {{
            transform: translateY(0px) scale(0.99);
        }}

        /* ---------- Metric Badges ---------- */
        .metric-badge {{
            background: rgba(255,255,255,0.7);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        }}
        .metric-badge .value {{
            font-family: 'Poppins', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: {COLOR_PRIMARY_DARK};
        }}
        .metric-badge .label {{
            font-size: 0.8rem;
            color: #6E8195;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLOR_PRIMARY_DARK} 0%, {COLOR_PRIMARY} 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: #F0F7FC !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.2);
        }}
        .sidebar-logo {{
            font-size: 2.4rem;
            text-align: center;
            margin-bottom: 0.2rem;
        }}
        .sidebar-title {{
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 1.3rem;
            text-align: center;
        }}
        .sidebar-caption {{
            text-align: center;
            font-size: 0.78rem;
            opacity: 0.85;
            margin-bottom: 1rem;
        }}
        .sidebar-info-box {{
            background: rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
            font-size: 0.83rem;
            margin-bottom: 0.6rem;
        }}

        /* ---------- Footer ---------- */
        .app-footer {{
            text-align: center;
            padding: 1.6rem 0 0.6rem 0;
            color: #6E8195;
            font-size: 0.85rem;
        }}
        .app-footer .tech-badges span {{
            display: inline-block;
            background: rgba(15,109,191,0.08);
            color: {COLOR_PRIMARY_DARK};
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 999px;
            margin: 0 4px;
            font-size: 0.78rem;
        }}

        /* ---------- Input Labels ---------- */
        label[data-testid="stWidgetLabel"] p {{
            font-weight: 600 !important;
            color: #2E3B4A !important;
        }}

        /* Divider spacing */
        hr {{ margin: 1.2rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_hero_section() -> None:
    """Render the top hero / landing section."""
    col_text, col_visual = st.columns([2.1, 1], gap="large")

    with col_text:
        st.markdown(
            f"""
            <div class="hero-container">
                <div class="hero-badge">✨ Powered by Machine Learning</div>
                <div class="hero-title">{APP_ICON} GlucoGuide</div>
                <div class="hero-subtitle">AI-Based Diabetes Prediction &<br>Personalized Risk Analysis</div>
                <div class="hero-desc">
                    GlucoGuide uses a trained machine learning model to analyze your
                    clinical profile — glucose levels, height &amp; weight, blood
                    pressure, HbA1c and more — and estimate your diabetes risk
                    category, complete with personalized, easy-to-understand
                    health recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_visual:
        hero_b64 = get_image_base64(HERO_IMAGE_PATH)
        if hero_b64:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:center; height:100%;">
                    <img src="data:image/jpeg;base64,{hero_b64}"
                         style="width:100%; max-width:420px; border-radius:24px;
                                filter: drop-shadow(0 16px 28px rgba(15,109,191,0.30));" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback visual if the illustration asset is missing
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:center; height:100%;">
                    <div style="font-size: 9rem; line-height:1; text-align:center; filter: drop-shadow(0 12px 20px rgba(15,109,191,0.35));">
                        🩸💉
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_row() -> None:
    """Render a row of trust / info metrics under the hero section."""
    metrics = [
        ("🎯", "5-Class", "Prediction Model"),
        ("📊", "11", "Clinical Inputs"),
    ]
    cols = st.columns(2)
    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-badge">
                    <div style="font-size:1.4rem;">{icon}</div>
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")


def render_input_form() -> Optional[PatientInput]:
    """Render the patient data entry form inside a glass card.

    Returns
    -------
    A PatientInput instance if the user submitted the form, otherwise None.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📋 Patient Clinical Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">Fill in the fields below as accurately as possible. '
        'All values should reflect the most recent lab results / measurements.</div>',
        unsafe_allow_html=True,
    )

    with st.form(key="patient_form", clear_on_submit=False):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            age = st.number_input(
                "🎂 Age",
                min_value=1, max_value=120, value=45, step=1,
                help="Patient's age in years. Valid range: 1–120.",
                placeholder="e.g. 45",
            )
            gender = st.selectbox(
                "⚧ Gender",
                options=["Female", "Male", "Other"],
                index=0,
                help="Biological sex recorded for the patient.",
            )
            height_cm = st.number_input(
                "📏 Height (cm)",
                min_value=100.0, max_value=250.0, value=170.0, step=1.0,
                help="Patient's height in centimeters. Valid range: 100–250 cm.",
                placeholder="e.g. 170",
            )
            weight_kg = st.number_input(
                "⚖️ Weight (kg)",
                min_value=20.0, max_value=300.0, value=72.0, step=0.5,
                help="Patient's body weight in kilograms. Valid range: 20–300 kg.",
                placeholder="e.g. 72",
            )
            hba1c = st.number_input(
                "🧪 HbA1c (%)",
                min_value=3.0, max_value=15.0, value=5.6, step=0.1,
                help="Glycated hemoglobin percentage. Normal: below 5.7%, Prediabetes: 5.7–6.4%, Diabetes: 6.5%+.",
                placeholder="e.g. 5.6",
            )
            glucose_fasting = st.number_input(
                "🩸 Fasting Glucose (mg/dL)",
                min_value=50.0, max_value=400.0, value=95.0, step=1.0,
                help="Blood glucose after at least 8 hours of fasting. Normal: 70–99 mg/dL.",
                placeholder="e.g. 95",
            )
            glucose_postprandial = st.number_input(
                "🍽️ Postprandial Glucose (mg/dL)",
                min_value=50.0, max_value=500.0, value=130.0, step=1.0,
                help="Blood glucose measured ~2 hours after eating. Normal: below 140 mg/dL.",
                placeholder="e.g. 130",
            )

        with col_right:
            cholesterol_total = st.number_input(
                "🫀 Total Cholesterol (mg/dL)",
                min_value=100.0, max_value=400.0, value=180.0, step=1.0,
                help="Total blood cholesterol level. Desirable: below 200 mg/dL.",
                placeholder="e.g. 180",
            )
            systolic_bp = st.number_input(
                "💓 Systolic Blood Pressure (mmHg)",
                min_value=80.0, max_value=220.0, value=118.0, step=1.0,
                help="Top number of a blood pressure reading. Normal: below 120 mmHg.",
                placeholder="e.g. 118",
            )
            family_history_label = st.selectbox(
                "🧬 Family History of Diabetes",
                options=["No", "Yes"],
                index=0,
                help="Whether an immediate family member (parent/sibling) has diabetes.",
            )
            physical_activity_minutes_per_week = st.number_input(
                "🏃 Physical Activity (min/week)",
                min_value=0.0, max_value=1500.0, value=150.0, step=5.0,
                help="Total minutes of moderate-to-vigorous exercise per week. WHO recommends 150+ minutes.",
                placeholder="e.g. 150",
            )
            insulin_level = st.number_input(
                "💉 Insulin Level (µIU/mL)",
                min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                help="Fasting serum insulin level. Typical fasting range: 2.6–24.9 µIU/mL.",
                placeholder="e.g. 10",
            )

        st.write("")
        submitted = st.form_submit_button("🔍 Predict My Risk", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return None

    # BMI is derived from height (cm) and weight (kg): BMI = weight / height(m)^2
    height_m = height_cm / 100.0
    computed_bmi = weight_kg / (height_m ** 2)

    return PatientInput(
        age=int(age),
        gender=gender,
        bmi=float(computed_bmi),
        height_cm=float(height_cm),
        weight_kg=float(weight_kg),
        hba1c=float(hba1c),
        glucose_fasting=float(glucose_fasting),
        glucose_postprandial=float(glucose_postprandial),
        cholesterol_total=float(cholesterol_total),
        systolic_bp=float(systolic_bp),
        family_history_diabetes=1 if family_history_label == "Yes" else 0,
        physical_activity_minutes_per_week=float(physical_activity_minutes_per_week),
        insulin_level=float(insulin_level),
    )


def build_gauge_figure(confidence: float, color: str) -> go.Figure:
    """Build a plotly circular gauge showing the model's confidence / risk score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Poppins"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#B7C6D3", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(31,170,89,0.12)"},
                    {"range": [30, 70], "color": "rgba(242,153,74,0.12)"},
                    {"range": [70, 100], "color": "rgba(235,87,87,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def render_result_card(result: PredictionResult) -> None:
    """Render the primary prediction result card (color-coded by risk level)."""
    risk_css_class = {
        "Low Risk": "low-risk",
        "Moderate Risk": "moderate-risk",
        "High Risk": "high-risk",
    }[result.risk_label]

    icon = {"Low Risk": "✅", "Moderate Risk": "⚠️", "High Risk": "🚨"}[result.risk_label]

    st.markdown(
        f"""
        <div class="result-card {risk_css_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-title">{result.predicted_class}</div>
            <div class="result-subtitle">Model confidence: <strong>{result.confidence * 100:.1f}%</strong> &nbsp;•&nbsp; Risk Category: <strong>{result.risk_label}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_breakdown(result: PredictionResult) -> None:
    """Render a horizontal bar chart of the probability for every diabetes class."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Full Probability Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">How confident the model is across all possible diagnostic categories.</div>',
        unsafe_allow_html=True,
    )

    sorted_probs = dict(sorted(result.class_probabilities.items(), key=lambda kv: kv[1], reverse=True))
    labels = list(sorted_probs.keys())
    values = [round(v * 100, 2) for v in sorted_probs.values()]
    colors = [
        RISK_COLOR_MAP.get(label, COLOR_DANGER) if label == result.predicted_class
        else "#C9D6E0"
        for label in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#1A2733", size=13)),
        font={"family": "Inter", "size": 13, "color": "#1A2733"},
    )
    fig.update_traces(textfont=dict(color="#1A2733", size=12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def get_recommendations(result: PredictionResult) -> list[str]:
    """Return a personalized list of recommendations based on the predicted class."""
    if result.is_low_risk:
        return [
            "🥗 Maintain a balanced, low-sugar diet rich in fiber and whole grains",
            "🏃 Continue at least 150 minutes of moderate exercise per week",
            "💧 Stay well-hydrated and limit sugary beverages",
            "🩺 Schedule a routine check-up and glucose screening annually",
            "😴 Prioritize 7–9 hours of quality sleep each night",
        ]
    if result.is_moderate_risk:
        return [
            "🩺 Consult a physician for a comprehensive metabolic evaluation",
            "🧪 Repeat HbA1c and fasting glucose tests in 3–6 months",
            "🍭 Reduce refined sugar and simple carbohydrate intake",
            "⚖️ Aim for gradual weight management (5–7% body weight if overweight)",
            "🏃 Increase physical activity to at least 150–200 minutes per week",
        ]
    # High risk categories: Type 1, Type 2, Gestational
    return [
        "🩺 Consult an endocrinologist or physician as soon as possible",
        "🧪 Get a confirmatory HbA1c and comprehensive metabolic panel",
        "🍭 Significantly reduce sugar and refined carbohydrate intake",
        "⚖️ Work with a healthcare provider on a personalized weight management plan",
        "🏃 Adopt a structured, medically-approved exercise routine",
        "💊 Discuss medication or insulin therapy options with your doctor",
    ]


def render_recommendations(result: PredictionResult) -> None:
    """Render the personalized recommendation list."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">Tailored guidance based on your predicted risk category.</div>',
        unsafe_allow_html=True,
    )
    for rec in get_recommendations(result):
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top:1rem; font-size:0.8rem; color:#8494A3;">
        ⚠️ <strong>Medical Disclaimer:</strong> GlucoGuide is an AI-assisted screening tool and
        does not constitute a medical diagnosis. Always consult a licensed healthcare
        professional for accurate diagnosis and treatment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_results_section(patient: PatientInput) -> None:
    """Run inference and render the complete results section, with error handling."""
    with st.spinner("🔬 Analyzing clinical profile and running AI inference..."):
        time.sleep(0.6)  # brief pause for a polished, deliberate feel
        try:
            result = run_prediction(patient)
        except RuntimeError as exc:
            st.markdown(
                f"""
                <div class="glass-card" style="border: 1px solid {COLOR_DANGER};">
                    <div class="section-heading" style="color:{COLOR_DANGER};">❌ Prediction Error</div>
                    <div class="section-subheading">{exc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    st.markdown("---")
    st.markdown('<div class="section-heading">🧾 Your Prediction Result</div>', unsafe_allow_html=True)
    st.write("")

    render_result_card(result)

    st.markdown(
        f"""
        <div class="metric-badge" style="display:inline-block; margin-bottom:1.2rem;">
            <div class="label">Calculated BMI</div>
            <div class="value">{patient.bmi:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_gauge, col_spacer = st.columns([1, 1.4])
    with col_gauge:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading" style="font-size:1.15rem;">🎯 Risk Score</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_gauge_figure(result.confidence, result.risk_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.progress(min(int(result.confidence * 100), 100))
        st.markdown("</div>", unsafe_allow_html=True)

    render_probability_breakdown(result)
    render_recommendations(result)


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar() -> None:
    """Render the professional sidebar with navigation and project info."""
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🩺</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-title">{APP_TITLE}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-caption">AI-Based Diabetes Prediction &amp; Personalized Risk Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("#### 🧭 Navigation")
        st.markdown("- 🏠 Home\n- 📋 Prediction Form\n- 📊 Risk Report\n- ℹ️ About")
        st.markdown("---")

        st.markdown("#### ℹ️ About This Project")
        st.markdown(
            """
            <div class="sidebar-info-box">
            GlucoGuide analyzes 11 clinical indicators using a trained
            XGBoost model to classify diabetes risk into five categories:
            No Diabetes, Pre-Diabetes, Type 1, Type 2, and Gestational.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🧠 Model Information")
        st.markdown(
            f"""
            <div class="sidebar-info-box">
            <strong>Algorithm:</strong> XGBoost Classifier<br>
            <strong>Version:</strong> {MODEL_VERSION}<br>
            <strong>Classes:</strong> 5 (multiclass)<br>
            <strong>Inputs:</strong> 11 clinical features
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 👨‍💻 Developer")
        st.markdown(
            """
            <div class="sidebar-info-box">
            Built by the <strong>GlucoGuide Engineering Team</strong><br>
            🔗 <a href="https://github.com/" target="_blank" style="color:#D6ECFA;">GitHub Repository</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("© 2026 GlucoGuide. For educational & screening purposes only.")


# ==============================================================================
# FOOTER
# ==============================================================================

def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        """
        <div class="app-footer">
            Developed with ❤️ using
            <div class="tech-badges" style="margin-top:0.4rem;">
                <span>🐍 Python</span>
                <span>🤖 XGBoost</span>
                <span>🔬 Scikit-learn</span>
                <span>🚀 Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="GlucoGuide",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_sidebar()

    # ---- Startup artifact check (surfaces missing/broken model files early) ----
    _, _, _, load_error = load_artifacts()
    if load_error is not None:
        st.error(f"⚠️ **Application cannot start:** {load_error}")
        st.info(
            "Please ensure `xgb_model.pkl`, `preprocessor.pkl`, and `label_encoder.pkl` "
            "are present inside the `models/` directory."
        )
        return

    render_hero_section()
    render_metric_row()

    patient = render_input_form()

    if patient is not None:
        render_results_section(patient)

    render_footer()


if __name__ == "__main__":
    main()
