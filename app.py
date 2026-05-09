import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import onnxruntime as ort
from scipy import signal
import warnings
import io
import os
from datetime import datetime

# ── PDF ──────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Cortex — ADHD Diagnostic System",
    page_icon="🧠",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────
if "diagnosis_history" not in st.session_state:
    st.session_state.diagnosis_history = []

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Brand Palette ──────────────────────────────────
       Gold     #E8A020   Light Gold  #F5C85A
       Rust     #C04B1A   Light Rust  #E8733A
       Sky      #4A9FD4   Light Sky   #7DC0E8
       Navy     #1A3A6B   Mid Navy    #2B5FA0
    ── ─────────────────────────────────────────────── */

    /* Main background — أغمق */
    .stApp {
        background: linear-gradient(160deg, #060E1F 0%, #0D2040 50%, #060E1F 100%);
    }
    .main .block-container { padding-top: 2rem; }

    /* All default text white */
    .stApp, .stApp p, .stApp li, .stApp span,
    .stMarkdown, .stMarkdown p { color: #E8F0FF !important; }

    /* Page titles */
    h1 { color: #F5C85A !important; font-weight: 800 !important; }
    h2 { color: #7DC0E8 !important; }
    h3 { color: #7DC0E8 !important; }

    /* Sidebar — أغمق */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #04080F 0%, #0D2040 100%) !important;
        border-right: 1px solid #1A3A6B;
    }
    [data-testid="stSidebar"] * { color: #E8F0FF !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: #7DC0E8 !important; font-weight: 500;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F5C85A !important; }
    [data-testid="stSidebar"] hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg,
            #F5C85A 0%,
            #E8A020 25%,
            #C04B1A 55%,
            #3A2A1A 80%,
            #060E1F 100%
        ) !important;
        opacity: 1 !important;
        margin: 1rem 0 !important;
        border-radius: 2px !important;
    }

    /* ── Gradient Divider class ───────────────────────── */
    .gradient-divider {
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg,
            #F5C85A 0%,
            #E8A020 25%,
            #C04B1A 55%,
            #3A2A1A 80%,
            #060E1F 100%
        );
        border: none;
        margin: 1.2rem 0;
        border-radius: 2px;
    }

    /* Override default Streamlit hr with gradient */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg,
            #F5C85A 0%,
            #E8A020 25%,
            #C04B1A 55%,
            #3A2A1A 80%,
            #060E1F 100%
        ) !important;
        opacity: 1 !important;
        margin: 1.2rem 0 !important;
        border-radius: 2px !important;
    }

    /* Cover page */
    .cover-title {
        font-size: 3.2rem; font-weight: 900; text-align: center; margin-top: 1rem;
        background: linear-gradient(90deg, #F5C85A, #E8A020);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .cover-sub {
        font-size: 1.2rem; color: #7DC0E8 !important;
        text-align: center; font-weight: 600; margin-bottom: 0.5rem;
    }
    .cover-desc {
        font-size: 0.95rem; color: #B8CFEE !important; text-align: center;
        max-width: 680px; margin: 0 auto 2rem auto; line-height: 1.9;
    }

    /* Result boxes */
    .result-box   { border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .result-adhd  {
        background: linear-gradient(135deg, rgba(192,75,26,0.25), rgba(232,115,58,0.15));
        border: 2px solid #E8733A;
    }
    .result-ctrl  {
        background: linear-gradient(135deg, rgba(74,159,212,0.25), rgba(125,192,232,0.15));
        border: 2px solid #4A9FD4;
    }
    .result-label { font-size: 2rem; font-weight: 800; margin-bottom: 0.4rem; }
    .result-adhd .result-label { color: #E8733A; }
    .result-ctrl  .result-label { color: #7DC0E8; }
    .result-prob  { font-size: 1rem; color: #B8CFEE !important; }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(232,160,32,0.12);
        border-left: 4px solid #E8A020;
        border-radius: 8px; padding: 0.9rem 1.2rem;
        font-size: 0.88rem; color: #D4B96A !important; line-height: 1.6; margin-top: 1.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(43,95,160,0.35) !important;
        border: 1px solid #4A9FD4 !important;
        border-radius: 10px; padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #7DC0E8 !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: #F5C85A !important; font-weight: 800; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #E8A020, #C04B1A) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #F5C85A, #E8733A) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 2px solid #E8733A !important;
        color: #E8733A !important; border-radius: 8px !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #2B5FA0, #4A9FD4) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(6,14,31,0.6) !important;
        border: 2px dashed #4A9FD4 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] * { color: #B8CFEE !important; }

    /* Labels and upload hints */
    label, .stFileUploader label,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stRadio label,
    .stMarkdown p strong { color: #F5C85A !important; font-weight: 600 !important; }

    /* Upload dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(6,14,31,0.5) !important;
        border: 2px dashed #E8A020 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #F5C85A !important;
        background: rgba(232,160,32,0.08) !important;
    }

    /* Browse files button inside uploader */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #E8A020, #C04B1A) !important;
        color: white !important; border: none !important;
        border-radius: 6px !important; font-weight: 600 !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background: rgba(6,14,31,0.9) !important;
        color: #E8F0FF !important;
        border: 1.5px solid #E8A020 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #F5C85A !important;
        box-shadow: 0 0 0 2px rgba(245,200,90,0.2) !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        background: rgba(6,14,31,0.9) !important;
        border: 1.5px solid #E8A020 !important;
        border-radius: 8px !important;
        color: #E8F0FF !important;
    }

    /* Page subtitle */
    .stApp .stMarkdown p { color: #E8C170 !important; }

    /* Subheader text */
    h3 { color: #F5C85A !important; }

    /* st.markdown bold */
    .stMarkdown strong { color: #F5C85A !important; font-weight: 700 !important; }

    /* st.subheader section titles */
    [data-testid="stHeading"] h3,
    [data-testid="stHeading"] h2 {
        color: #F5C85A !important;
        border-left: 4px solid #E8A020;
        padding-left: 0.6rem;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(6,14,31,0.8) !important;
        border-radius: 8px;
    }

    /* Alerts */
    .stAlert { border-radius: 8px !important; background: rgba(43,95,160,0.3) !important; }

    /* Spinner */
    .stSpinner { color: #F5C85A !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    cnn  = ort.InferenceSession("models/cnn_eeg_model.onnx")
    lr   = joblib.load("models/lr_hyperaktiv.pkl")
    meta = joblib.load("models/meta_model.pkl")
    sc   = joblib.load("models/scaler_hyp.pkl")
    sel  = joblib.load("models/selector.pkl")
    return cnn, lr, meta, sc, sel

cnn_model, lr_base, meta_model, scaler_hyp, selector = load_models()


# ── Signal Processing ─────────────────────────────────────
def bandpass(data, lo=1.0, hi=45.0, fs=128, order=4):
    nyq = fs / 2
    b, a = signal.butter(order, [lo/nyq, hi/nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=0)

def notch(data, freq=50.0, fs=128, Q=30):
    b, a = signal.iirnotch(freq/(fs/2), Q)
    return signal.filtfilt(b, a, data, axis=0)

def process_eeg(sig, win_len=256, step=128):
    sig = bandpass(sig)
    sig = notch(sig)
    wins = []
    for start in range(0, sig.shape[0] - win_len + 1, step):
        w = sig[start:start+win_len, :]
        if np.max(np.abs(w)) >= 150:
            continue
        w_norm = (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-8)
        wins.append(w_norm)
    return np.array(wins) if wins else None

def predict_eeg(wins):
    wins_f = wins.astype(np.float32)
    input_name = cnn_model.get_inputs()[0].name
    probs = cnn_model.run(None, {input_name: wins_f})[0].ravel()
    return probs


# ── PDF Report Generator ──────────────────────────────────
def generate_pdf_report(patient_name, patient_age, patient_gender,
                         p_eeg, p_hyp, meta_prob, has_eeg, has_hyp):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    BLUE       = colors.HexColor("#1E3A5F")
    LIGHT_BLUE = colors.HexColor("#2B6CB0")
    BG_BLUE    = colors.HexColor("#EBF4FF")
    GREEN      = colors.HexColor("#1A7A1A")
    RED        = colors.HexColor("#CC0000")
    GRAY       = colors.HexColor("#666666")
    DIVIDER    = colors.HexColor("#CCCCCC")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("title_s", parent=styles["Normal"],
        fontSize=24, textColor=BLUE, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=6)

    sub_style = ParagraphStyle("sub_s", parent=styles["Normal"],
        fontSize=12, textColor=LIGHT_BLUE, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=0)

    section_style = ParagraphStyle("sec_s", parent=styles["Normal"],
        fontSize=13, textColor=BLUE, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6)

    body_style = ParagraphStyle("body_s", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#333333"),
        fontName="Helvetica", leading=16)

    small_style = ParagraphStyle("small_s", parent=styles["Normal"],
        fontSize=9, textColor=GRAY, fontName="Helvetica",
        alignment=TA_CENTER)

    diag_style = ParagraphStyle("diag", parent=styles["Normal"],
        fontSize=20, textColor=RED if meta_prob >= 0.5 else GREEN,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)

    prob_style = ParagraphStyle("prob", parent=styles["Normal"],
        fontSize=13, textColor=GRAY, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=8)

    interp_style = ParagraphStyle("interp", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#444444"),
        fontName="Helvetica", alignment=TA_CENTER, leading=15)

    if has_eeg and has_hyp:
        interp_text = ("The integrated analysis of both EEG brain signals and behavioral data "
                       "indicates significant ADHD-associated patterns. Further clinical evaluation is recommended."
                       if meta_prob >= 0.5 else
                       "The integrated analysis of both EEG brain signals and behavioral data "
                       "shows no significant ADHD-associated patterns. Routine follow-up is advised if symptoms persist.")
    elif has_eeg:
        interp_text = ("EEG brain signal patterns indicate significant ADHD-associated findings."
                       if meta_prob >= 0.5 else
                       "EEG brain signal patterns show no significant ADHD-associated findings.")
    else:
        interp_text = ("Behavioral data patterns indicate significant ADHD-associated findings."
                       if meta_prob >= 0.5 else
                       "Behavioral data patterns show no significant ADHD-associated findings.")

    diagnosis_label = "ADHD Detected" if meta_prob >= 0.5 else "No ADHD Detected"

    now      = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    story = []

    # ── Logo ─────────────────────────────────────────────
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cortex_logo.png")
    if os.path.exists(logo_path):
        logo = RLImage(logo_path, width=3*cm, height=3*cm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 0.4*cm))

    # ── Header ────────────────────────────────────────────
    story.append(Paragraph("Cortex", title_style))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("ADHD Diagnostic System — Patient Report", sub_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_BLUE))
    story.append(Spacer(1, 0.5*cm))

    # ── Patient Info ──────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))
    patient_data = [
        ["Full Name",   patient_name],
        ["Age",         f"{patient_age} years"],
        ["Gender",      patient_gender],
        ["Report Date", date_str],
        ["Report Time", time_str],
    ]
    pt = Table(patient_data, colWidths=[5*cm, 12*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), BG_BLUE),
        ("TEXTCOLOR",     (0, 0), (0, -1), BLUE),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.6*cm))

    # ── Diagnosis Result ──────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Diagnosis Result", section_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(diagnosis_label, diag_style))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(f"Integrated Model Probability: {meta_prob:.1%}", prob_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(interp_text, interp_style))
    story.append(Spacer(1, 0.6*cm))

    # ── Disclaimer ────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER))
    story.append(Spacer(1, 0.3*cm))
    disclaimer = ("This report is generated by the Cortex AI Diagnostic System, "
                  "and does not replace a clinical diagnosis by a licensed medical professional.")
    story.append(Paragraph(disclaimer, small_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.image("cortex_logo.png", width=80)
st.sidebar.title("Cortex")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Navigation", [
    "Home",
    "EEG-Based ADHD Diagnosis",
    "Behavioral ADHD Diagnosis",
    "Integrated ADHD Diagnosis",
    "History"
])


# ══════════════════════════════════════════════════════════
# Page 0 — Home / Cover
# ══════════════════════════════════════════════════════════
if mode == "Home":
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        logo_path = os.path.join(os.path.dirname(__file__), "cortex_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)

    st.markdown('<div class="cover-title">Cortex</div>', unsafe_allow_html=True)
    st.markdown('<div class="cover-sub">AI-Powered ADHD Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cover-desc">
        Cortex is an AI-powered diagnostic system designed to improve the accuracy and speed
        of diagnosing Attention Deficit Hyperactivity Disorder (ADHD). It uses machine learning
        and deep learning to analyze behavioral and neurological data, providing evidence-based
        clinical decision support for doctors and mental health professionals.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# Page 2 — EEG-Based ADHD Diagnosis
# ══════════════════════════════════════════════════════════
elif mode == "EEG-Based ADHD Diagnosis":
    st.title("EEG-Based ADHD Diagnosis")
    st.markdown("<span style='color:#E8C170;font-size:0.97rem;'>Upload a brain signal file to get an AI-based ADHD diagnostic result.</span>", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload EEG file (.mat)", type=["mat"])
    if uploaded:
        import scipy.io, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            mat = scipy.io.loadmat(tmp_path)
            key = [k for k in mat.keys() if not k.startswith("_")][0]
            sig = mat[key]
            if hasattr(sig, "dtype") and sig.dtype.names:
                sig = sig[0][0]
            if sig.ndim == 1:
                sig = sig.reshape(-1, 1)
            if sig.shape[1] > sig.shape[0]:
                sig = sig.T

            with st.spinner("Processing brain signal..."):
                wins = process_eeg(sig)

            if wins is not None and len(wins) > 0:
                probs     = predict_eeg(wins)
                mean_prob = probs.mean()
                is_adhd   = mean_prob >= 0.5

                fig_g = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = mean_prob * 100,
                    number= {"suffix": "%", "font": {"size": 40}},
                    title = {"text": "ADHD Likelihood", "font": {"size": 18}},
                    gauge = {
                        "axis"     : {"range": [0, 100]},
                        "bar"      : {"color": "#E74C3C" if is_adhd else "#2ECC71"},
                        "steps"    : [
                            {"range": [0,  50], "color": "rgba(46,204,113,0.15)"},
                            {"range": [50, 100], "color": "rgba(231,76,60,0.15)"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                    }
                ))
                fig_g.update_layout(height=320, template="plotly_white")
                st.plotly_chart(fig_g, use_container_width=True)

                if is_adhd:
                    st.markdown(f"""
                    <div class="result-box result-adhd">
                        <div class="result-label">🔴 ADHD Indicators Detected</div>
                        <div class="result-prob">The brain signal shows patterns associated with ADHD
                        (Probability: {mean_prob:.1%})</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-ctrl">
                        <div class="result-label">🟢 No ADHD Indicators Detected</div>
                        <div class="result-prob">The brain signal appears within normal range
                        (Probability: {mean_prob:.1%})</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer-box">
                This result is based on EEG data only. For a complete diagnostic report,
                use the <b>Integrated ADHD Diagnosis</b> page with both EEG and behavioral data.
                </div>""", unsafe_allow_html=True)
            else:
                st.error("Could not process the uploaded file. Please check the signal format.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════
# Page 3 — Behavioral ADHD Diagnosis
# ══════════════════════════════════════════════════════════
elif mode == "Behavioral ADHD Diagnosis":
    st.title("Behavioral ADHD Diagnosis")
    st.markdown("<span style='color:#E8C170;font-size:0.97rem;'>Upload a behavioral data file to get an AI-based ADHD diagnostic result.</span>", unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    f = st.file_uploader("Upload behavioral data file (.csv)", type=["csv"])
    if f:
        try:
            df_in = pd.read_csv(f, sep=";").fillna(0)
            df_in = df_in.select_dtypes(include=["number", "object"])

            X = df_in.drop(columns=["ID"], errors="ignore")
            X = X.select_dtypes(include=["number"]).values
            X_sel = scaler_hyp.transform(selector.transform(X))
            probs = lr_base.predict_proba(X_sel)[:, 1]

            x_meta = np.column_stack([
                np.full(len(probs), 0.5), probs,
                np.zeros(len(probs)), np.ones(len(probs))
            ])
            meta_probs = meta_model.predict_proba(x_meta)[:, 1]
            mean_prob  = float(meta_probs.mean())
            is_adhd    = mean_prob >= 0.5

            fig_g = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = mean_prob * 100,
                number= {"suffix": "%", "font": {"size": 40}},
                title = {"text": "ADHD Likelihood", "font": {"size": 18}},
                gauge = {
                    "axis"     : {"range": [0, 100]},
                    "bar"      : {"color": "#E74C3C" if is_adhd else "#2ECC71"},
                    "steps"    : [
                        {"range": [0,  50], "color": "rgba(46,204,113,0.15)"},
                        {"range": [50, 100], "color": "rgba(231,76,60,0.15)"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                }
            ))
            fig_g.update_layout(height=320, template="plotly_white")
            st.plotly_chart(fig_g, use_container_width=True)

            if is_adhd:
                st.markdown(f"""
                <div class="result-box result-adhd">
                    <div class="result-label">🔴 ADHD Indicators Detected</div>
                    <div class="result-prob">Behavioral data shows patterns associated with ADHD
                    (Probability: {mean_prob:.1%})</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-ctrl">
                    <div class="result-label">🟢 No ADHD Indicators Detected</div>
                    <div class="result-prob">Behavioral data shows no significant ADHD patterns
                    (Probability: {mean_prob:.1%})</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer-box">
            This result is based on behavioral data only. For a complete diagnostic report,
            use the <b>Integrated ADHD Diagnosis</b> page with both EEG and behavioral data.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Processing error: {e}")


# ══════════════════════════════════════════════════════════
# Page 4 — Integrated ADHD Diagnosis
# ══════════════════════════════════════════════════════════
elif mode == "Integrated ADHD Diagnosis":
    st.title("Integrated ADHD Diagnosis")
    st.markdown("<span style='color:#E8C170;font-size:0.97rem;'>Enter patient information and upload data files to generate a full diagnostic report.</span>", unsafe_allow_html=True)

    # خط تدرج تحت العنوان

    # ── Patient Info ──────────────────────────────────────
    st.markdown("<h3 style='color:#F5C85A;border-left:4px solid #E8A020;padding-left:0.6rem;margin-bottom:1rem;'>👤 Patient Information</h3>", unsafe_allow_html=True)
    pi1, pi2, pi3 = st.columns(3)
    with pi1:
        patient_name   = st.text_input("Full Name", placeholder="e.g. Ahmed Al-Rashidi")
    with pi2:
        patient_age    = st.number_input("Age", min_value=1, max_value=120, value=25)
    with pi3:
        patient_gender = st.selectbox("Gender", ["Male", "Female"])

    # خط تدرج يفصل بين معلومات المريض ورفع البيانات
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── File Upload ───────────────────────────────────────
    st.markdown("<h3 style='color:#F5C85A;border-left:4px solid #E8A020;padding-left:0.6rem;margin-bottom:1rem;'>Upload Data Files</h3>", unsafe_allow_html=True)
    col_eeg, col_hyp = st.columns(2)
    with col_eeg:
        st.markdown("<span style='color:#F5C85A;font-weight:700;font-size:1rem;'>EEG File</span>", unsafe_allow_html=True)
        eeg_file = st.file_uploader("Upload .mat file", type=["mat"], key="meta_eeg")
    with col_hyp:
        st.markdown("<span style='color:#F5C85A;font-weight:700;font-size:1rem;'>Behavioral Data File</span>", unsafe_allow_html=True)
        hyp_file = st.file_uploader("Upload .csv file", type=["csv"], key="meta_hyp")

    # خط تدرج قبل النتيجة
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    p_eeg   = None
    p_hyp   = None
    has_eeg = 0
    has_hyp = 0

    if eeg_file:
        import scipy.io, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(eeg_file.read())
            tmp_path = tmp.name
        try:
            mat = scipy.io.loadmat(tmp_path)
            key = [k for k in mat.keys() if not k.startswith("_")][0]
            sig = mat[key]
            if hasattr(sig, "dtype") and sig.dtype.names:
                sig = sig[0][0]
            if sig.ndim == 1:
                sig = sig.reshape(-1, 1)
            if sig.shape[1] > sig.shape[0]:
                sig = sig.T
            with st.spinner("Processing EEG signal..."):
                wins = process_eeg(sig)
            if wins is not None and len(wins) > 0:
                probs_eeg = predict_eeg(wins)
                p_eeg = float(probs_eeg.mean())
                has_eeg = 1
            else:
                st.error("Could not process the EEG file. Please check the signal format.")
        except Exception as e:
            st.error(f"EEG Error: {e}")
        finally:
            os.unlink(tmp_path)

    if hyp_file:
        try:
            df_hyp = pd.read_csv(hyp_file, sep=";").fillna(0)
            X = df_hyp.drop(columns=["ID"], errors="ignore")
            X = X.select_dtypes(include=["number"]).values
            X_sel = scaler_hyp.transform(selector.transform(X))
            probs_hyp = lr_base.predict_proba(X_sel)[:, 1]
            p_hyp = float(probs_hyp.mean())
            has_hyp = 1
        except Exception as e:
            st.error(f"Behavioral Data Error: {e}")

    # ── Integrated Decision ───────────────────────────────
    if p_eeg is not None or p_hyp is not None:
        st.markdown("<h3 style='color:#F5C85A;border-left:4px solid #E8A020;padding-left:0.6rem;margin-bottom:1rem;'>Diagnostic Result</h3>", unsafe_allow_html=True)

        x_meta = np.array([[
            p_eeg  if p_eeg is not None else 0.5,
            p_hyp  if p_hyp is not None else 0.5,
            has_eeg, has_hyp
        ]])
        meta_prob = meta_model.predict_proba(x_meta)[0, 1]
        is_adhd   = meta_prob >= 0.5

        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = meta_prob * 100,
            number= {"suffix": "%", "font": {"size": 40}},
            title = {"text": "ADHD Likelihood", "font": {"size": 18}},
            gauge = {
                "axis"     : {"range": [0, 100]},
                "bar"      : {"color": "#E74C3C" if is_adhd else "#2ECC71"},
                "steps"    : [
                    {"range": [0,  50], "color": "rgba(46,204,113,0.15)"},
                    {"range": [50, 100], "color": "rgba(231,76,60,0.15)"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
            }
        ))
        fig_g.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

        if is_adhd:
            st.markdown(f"""
            <div class="result-box result-adhd">
                <div class="result-label">🔴 ADHD Detected</div>
                <div class="result-prob">Integrated model probability: {meta_prob:.1%}</div>
                <div class="result-prob" style="margin-top:0.6rem;font-size:0.92rem;color:#444;">
                    The combined analysis of available data indicates significant ADHD-associated patterns.
                    Further clinical evaluation is recommended to confirm this finding.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-ctrl">
                <div class="result-label">🟢 No ADHD Detected</div>
                <div class="result-prob">Integrated model probability: {meta_prob:.1%}</div>
                <div class="result-prob" style="margin-top:0.6rem;font-size:0.92rem;color:#444;">
                    The combined analysis of available data shows no significant ADHD-associated patterns.
                    Routine follow-up is advised if clinical symptoms persist.
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Save to History ───────────────────────────────
        st.session_state.diagnosis_history.append({
            "Date":      datetime.now().strftime("%Y-%m-%d"),
            "Time":      datetime.now().strftime("%H:%M:%S"),
            "Patient":   patient_name if patient_name.strip() else "Unknown",
            "Age":       patient_age,
            "Gender":    patient_gender,
            "EEG Prob":  f"{p_eeg:.1%}" if p_eeg is not None else "—",
            "HYP Prob":  f"{p_hyp:.1%}" if p_hyp is not None else "—",
            "META Prob": f"{meta_prob:.1%}",
            "Diagnosis": "ADHD" if is_adhd else "Control",
        })

        # ── PDF Download ──────────────────────────────────
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#F5C85A;border-left:4px solid #E8A020;padding-left:0.6rem;margin-bottom:1rem;'>Download Patient Report</h3>", unsafe_allow_html=True)

        if not patient_name.strip():
            st.warning("Please enter the patient's name to generate the report.")
        else:
            if st.button("Generate & Download PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    pdf_buffer = generate_pdf_report(
                        patient_name   = patient_name,
                        patient_age    = patient_age,
                        patient_gender = patient_gender,
                        p_eeg          = p_eeg  if p_eeg is not None else 0.5,
                        p_hyp          = p_hyp  if p_hyp is not None else 0.5,
                        meta_prob      = meta_prob,
                        has_eeg        = has_eeg,
                        has_hyp        = has_hyp
                    )
                fname = f"Cortex_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label     = "Download Report PDF",
                    data      = pdf_buffer,
                    file_name = fname,
                    mime      = "application/pdf"
                )
                st.success("Report ready! Click the button above to download.")
    else:
        st.warning("Please upload at least one file to begin.")


# ══════════════════════════════════════════════════════════
# Page 5 — History
# ══════════════════════════════════════════════════════════
elif mode == "History":
    st.title("Diagnosis History")
    st.markdown("Records are saved for this session only and will be cleared when the app is closed.")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.diagnosis_history:
        st.info("No diagnoses recorded yet. Run a META Fusion diagnosis to start.")
    else:
        df_history = pd.DataFrame(st.session_state.diagnosis_history)

        total  = len(df_history)
        adhd_n = (df_history["Diagnosis"] == "ADHD 🔴").sum()
        ctrl_n = total - adhd_n

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Diagnoses", total)
        c2.metric("ADHD Detected",   int(adhd_n))
        c3.metric("Control",         int(ctrl_n))

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.subheader("Session Records")
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        if st.button("Clear History", type="secondary"):
            st.session_state.diagnosis_history = []
            st.rerun()
