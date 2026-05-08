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
    .cover-title { font-size:3rem;font-weight:800;color:#1E3A5F;text-align:center;margin-top:1rem; }
    .cover-sub   { font-size:1.2rem;color:#2B6CB0;text-align:center;margin-bottom:0.5rem; }
    .cover-desc  { font-size:0.95rem;color:#555;text-align:center;max-width:680px;margin:0 auto 2rem auto;line-height:1.8; }
    .fact-card   { background:#F0F6FF;border-left:5px solid #2B6CB0;border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:0.8rem; }
    .fact-label  { font-size:0.75rem;color:#2B6CB0;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem; }
    .fact-value  { font-size:1rem;color:#1E3A5F;font-weight:600; }
    .step-card   { background:#FFFFFF;border:1.5px solid #D0E4FF;border-radius:12px;padding:1.3rem 1.4rem;text-align:center; }
    .step-num    { font-size:1.8rem;font-weight:800;color:#2B6CB0;margin-bottom:0.4rem; }
    .step-title  { font-size:1rem;font-weight:700;color:#1E3A5F; }
    .step-desc   { font-size:0.88rem;color:#666;margin-top:0.3rem;line-height:1.5; }
    .disclaimer-box { background:#FFF8E1;border-left:5px solid #F59E0B;border-radius:8px;padding:0.9rem 1.2rem;font-size:0.88rem;color:#555;line-height:1.6;margin-top:1.5rem; }
    .result-box  { border-radius:16px;padding:2rem;text-align:center;margin:1rem 0; }
    .result-adhd { background:#FFF0F0;border:2px solid #E74C3C; }
    .result-ctrl { background:#F0FFF4;border:2px solid #2ECC71; }
    .result-label{ font-size:2rem;font-weight:800;margin-bottom:0.3rem; }
    .result-prob { font-size:1rem;color:#666; }
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
        fontSize=22, textColor=BLUE, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=4)

    sub_style = ParagraphStyle("sub_s", parent=styles["Normal"],
        fontSize=12, textColor=LIGHT_BLUE, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=2)

    section_style = ParagraphStyle("sec_s", parent=styles["Normal"],
        fontSize=13, textColor=BLUE, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6)

    body_style = ParagraphStyle("body_s", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#333333"),
        fontName="Helvetica", leading=16)

    small_style = ParagraphStyle("small_s", parent=styles["Normal"],
        fontSize=9, textColor=GRAY, fontName="Helvetica",
        alignment=TA_CENTER)

    diagnosis_label = "ADHD Detected" if meta_prob >= 0.5 else "Control (No ADHD)"
    diagnosis_color = RED if meta_prob >= 0.5 else GREEN

    if has_eeg and has_hyp:
        data_source = "EEG Signal + Hyperaktiv Behavioral Data"
        recommendation = ("Further clinical evaluation is strongly recommended. "
                          "The system detected ADHD indicators across both EEG and behavioral data sources."
                          if meta_prob >= 0.5 else
                          "No significant ADHD indicators were detected. "
                          "Routine follow-up is advised if symptoms persist.")
    elif has_eeg:
        data_source = "EEG Signal only"
        recommendation = ("EEG analysis indicates ADHD patterns. "
                          "Behavioral assessment is recommended to confirm diagnosis."
                          if meta_prob >= 0.5 else
                          "EEG analysis shows no significant ADHD patterns. "
                          "Consider behavioral testing if clinically indicated.")
    else:
        data_source = "Hyperaktiv Behavioral Data only"
        recommendation = ("Behavioral data suggests ADHD. "
                          "EEG analysis is recommended for a complete assessment."
                          if meta_prob >= 0.5 else
                          "Behavioral data shows no significant ADHD indicators. "
                          "EEG analysis can be performed for further confirmation.")

    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    story = []

    # ── Logo ─────────────────────────────────────────────
    logo_path = "cortex_logo.png"
    if os.path.exists(logo_path):
        logo = RLImage(logo_path, width=3*cm, height=3*cm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 0.3*cm))

    # ── Header ────────────────────────────────────────────
    story.append(Paragraph("Cortex", title_style))
    story.append(Paragraph("ADHD Diagnostic System — Patient Report", sub_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_BLUE))
    story.append(Spacer(1, 0.4*cm))

    # ── Patient Info ──────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))
    patient_data = [
        ["Full Name",    patient_name],
        ["Age",          f"{patient_age} years"],
        ["Gender",       patient_gender],
        ["Report Date",  date_str],
        ["Report Time",  time_str],
        ["Data Source",  data_source],
    ]
    pt = Table(patient_data, colWidths=[5*cm, 12*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), BG_BLUE),
        ("TEXTCOLOR",   (0, 0), (0, -1), BLUE),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, DIVIDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.5*cm))

    # ── Diagnosis Result ──────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Diagnosis Result", section_style))

    diag_style = ParagraphStyle("diag", parent=styles["Normal"],
        fontSize=18, textColor=diagnosis_color, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=4)
    prob_style = ParagraphStyle("prob", parent=styles["Normal"],
        fontSize=13, textColor=GRAY, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=4)

    story.append(Paragraph(diagnosis_label, diag_style))
    story.append(Paragraph(f"META Model Probability: {meta_prob:.1%}", prob_style))
    story.append(Spacer(1, 0.4*cm))

    # ── Model Probabilities Table ─────────────────────────
    story.append(Paragraph("Model Probabilities", section_style))
    model_rows = [["Source", "Model", "ADHD Probability", "Contribution"]]
    if has_eeg:
        model_rows.append(["EEG Signal", "CNN", f"{p_eeg:.1%}", "High (coef: 5.164)"])
    if has_hyp:
        model_rows.append(["Hyperaktiv", "Logistic Regression", f"{p_hyp:.1%}", "Moderate (coef: 1.958)"])
    model_rows.append(["Combined", "META Fusion", f"{meta_prob:.1%}", "Final Decision"])

    mt = Table(model_rows, colWidths=[4*cm, 5*cm, 4*cm, 4*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ALIGN",        (2, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
        ("BACKGROUND",   (0, len(model_rows)-1), (-1, len(model_rows)-1), BG_BLUE),
        ("FONTNAME",     (0, len(model_rows)-1), (-1, len(model_rows)-1), "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, DIVIDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.5*cm))

    # ── META Equation ─────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("META Model Equation", section_style))
    eq_style = ParagraphStyle("eq", parent=styles["Normal"],
        fontSize=9, textColor=BLUE, fontName="Courier",
        backColor=colors.HexColor("#F0F4F8"),
        borderPadding=(6, 8, 6, 8), spaceAfter=6)
    story.append(Paragraph(
        "z = -3.650 + 5.164 x p_eeg + 1.958 x p_hyp + (-0.357) x has_eeg + 0.371 x has_hyp",
        eq_style))
    story.append(Spacer(1, 0.3*cm))

    # ── Recommendation ────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Clinical Recommendation", section_style))
    story.append(Paragraph(recommendation, body_style))
    story.append(Spacer(1, 0.5*cm))

    # ── Disclaimer ────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER))
    story.append(Spacer(1, 0.2*cm))
    disclaimer = ("This report is generated by the Cortex AI Diagnostic System and is intended "
                  "for research and supportive purposes only. It does not replace a clinical "
                  "diagnosis by a licensed medical professional.")
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
    "🏠 Home",
    "🔬 EEG Analysis",
    "📋 Hyperaktiv Analysis",
    "🧠 META Fusion Diagnosis",
    "🗂️ History"
])


# ══════════════════════════════════════════════════════════
# Page 0 — Home / Cover
# ══════════════════════════════════════════════════════════
if mode == "🏠 Home":
    # ── Hero ─────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        logo_path = os.path.join(os.path.dirname(__file__), "cortex_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=160)

    st.markdown('<div class="cover-title">Cortex</div>', unsafe_allow_html=True)
    st.markdown('<div class="cover-sub">AI-Powered ADHD Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cover-desc">
        Cortex is an AI-powered diagnostic system designed to improve the accuracy and speed
        of ADHD assessment. It uses machine learning to analyze behavioral and cognitive data,
        providing evidence-based clinical decision support for doctors and mental health professionals.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Facts ────────────────────────────────────────
    st.markdown("### Key Facts")
    kf1, kf2, kf3 = st.columns(3)
    with kf1:
        st.markdown("""
        <div class="fact-card">
            <div class="fact-label">🎯 Domain</div>
            <div class="fact-value">ADHD Diagnosis</div>
        </div>""", unsafe_allow_html=True)
    with kf2:
        st.markdown("""
        <div class="fact-card">
            <div class="fact-label">🤖 Type</div>
            <div class="fact-value">AI-Powered Medical Software</div>
        </div>""", unsafe_allow_html=True)
    with kf3:
        st.markdown("""
        <div class="fact-card">
            <div class="fact-label">💡 Core Benefit</div>
            <div class="fact-value">Supporting clinicians with faster, more confident ADHD assessments.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── How to Use ───────────────────────────────────────
    st.markdown("### How to Use")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">①</div>
            <div class="step-title">Upload Patient Data</div>
            <div class="step-desc">Upload an EEG brain signal file, a behavioral data file, or both.</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">②</div>
            <div class="step-title">AI Analysis</div>
            <div class="step-desc">Cortex analyzes the data using its AI models and fuses the results into one decision.</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">③</div>
            <div class="step-title">Download Report</div>
            <div class="step-desc">Get a personalized diagnostic report for the patient, ready to download as PDF.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <b>Disclaimer:</b> Cortex is intended for research and clinical support only.
    It does not replace a diagnosis by a licensed medical professional.
    </div>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════
# Page 2 — EEG Analysis
# ══════════════════════════════════════════════════════════
elif mode == "🔬 EEG Analysis":
    st.title("🔬 EEG Brain Signal Analysis")
    st.markdown("Upload a brain signal file to get an AI-based ADHD assessment.")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload EEG file (.mat)", type=["mat"])
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

            with st.spinner("🧠 Analyzing brain signal..."):
                wins = process_eeg(sig)

            if wins is not None and len(wins) > 0:
                probs    = predict_eeg(wins)
                mean_prob = probs.mean()
                is_adhd  = mean_prob >= 0.5

                # ── Gauge ─────────────────────────────────
                fig_g = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = mean_prob * 100,
                    number= {"suffix": "%", "font": {"size": 40}},
                    title = {"text": "ADHD Likelihood", "font": {"size": 18}},
                    gauge = {
                        "axis"     : {"range": [0, 100]},
                        "bar"      : {"color": "#E74C3C" if is_adhd else "#2ECC71"},
                        "steps"    : [
                            {"range": [0,  50], "color": "#D5F5E3"},
                            {"range": [50, 100], "color": "#FADBD8"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                    }
                ))
                fig_g.update_layout(height=320, template="plotly_white")
                st.plotly_chart(fig_g, use_container_width=True)

                # ── Result card ───────────────────────────
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
                ⚠️ This result is based on EEG analysis only. For a complete assessment,
                use the <b>META Fusion Diagnosis</b> page with both EEG and behavioral data.
                </div>""", unsafe_allow_html=True)
            else:
                st.error("Could not process the uploaded file. Please check the signal format.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════
# Page 3 — Hyperaktiv Analysis
# ══════════════════════════════════════════════════════════
elif mode == "📋 Hyperaktiv Analysis":
    st.title("📋 Behavioral Data Analysis")
    st.markdown("Upload a behavioral data file to get an AI-based ADHD assessment.")
    st.markdown("---")

    f = st.file_uploader("📂 Upload behavioral data file (.csv)", type=["csv"])
    if f:
        df_in = pd.read_csv(f, sep=";").fillna(0)
        df_in = df_in.select_dtypes(include=["number", "object"])

        if st.button("🔍 Run Analysis", type="primary"):
            try:
                ids = df_in["ID"] if "ID" in df_in.columns else None
                X = df_in.drop(columns=["ID"], errors="ignore")
                X = X.select_dtypes(include=["number"]).values
                X_sel = scaler_hyp.transform(selector.transform(X))
                probs = lr_base.predict_proba(X_sel)[:, 1]

                x_meta = np.column_stack([
                    np.full(len(probs), 0.5), probs,
                    np.zeros(len(probs)), np.ones(len(probs))
                ])
                meta_probs = meta_model.predict_proba(x_meta)[:, 1]

                # ── Summary cards ─────────────────────────
                adhd_n = int((meta_probs >= 0.5).sum())
                ctrl_n = len(meta_probs) - adhd_n
                total  = len(meta_probs)

                ca, cb, cc = st.columns(3)
                ca.metric("Total Records",   total)
                cb.metric("ADHD Detected 🔴", adhd_n)
                cc.metric("Control 🟢",       ctrl_n)

                # ── Donut chart ───────────────────────────
                if total > 0:
                    fig_pie = go.Figure(go.Pie(
                        labels = ["ADHD", "Control"],
                        values = [adhd_n, ctrl_n],
                        hole   = 0.55,
                        marker = dict(colors=["#E74C3C", "#2ECC71"]),
                        textinfo = "percent+label",
                        textfont = dict(size=14)
                    ))
                    fig_pie.update_layout(
                        title  = "Diagnosis Distribution",
                        height = 320,
                        template = "plotly_white",
                        showlegend = False
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ── Results table ─────────────────────────
                st.markdown("#### Individual Results")
                result_df = pd.DataFrame({
                    "ID"       : ids if ids is not None else range(total),
                    "META Prob": np.round(meta_probs, 3),
                    "Diagnosis": ["ADHD 🔴" if p >= 0.5 else "Control 🟢" for p in meta_probs]
                })
                st.dataframe(result_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Processing error: {e}")


# ══════════════════════════════════════════════════════════
# Page 4 — META Fusion Diagnosis
# ══════════════════════════════════════════════════════════
elif mode == "🧠 META Fusion Diagnosis":
    st.title("🧠 META Fusion Diagnosis")
    st.markdown("Enter patient information, upload files, and download the diagnostic report.")
    st.markdown("---")

    # ── Patient Info ──────────────────────────────────────
    st.subheader("👤 Patient Information")
    pi1, pi2, pi3 = st.columns(3)
    with pi1:
        patient_name   = st.text_input("Full Name", placeholder="e.g. Ahmed Al-Rashidi")
    with pi2:
        patient_age    = st.number_input("Age", min_value=1, max_value=120, value=25)
    with pi3:
        patient_gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")

    # ── File Upload ───────────────────────────────────────
    st.subheader("📂 Upload Data Files")
    col_eeg, col_hyp = st.columns(2)
    with col_eeg:
        st.markdown("**📡 EEG File**")
        eeg_file = st.file_uploader("Upload .mat file", type=["mat"], key="meta_eeg")
    with col_hyp:
        st.markdown("**📋 Hyperaktiv File**")
        hyp_file = st.file_uploader("Upload features.csv", type=["csv"], key="meta_hyp")

    st.markdown("---")

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
            with st.spinner("Analyzing EEG..."):
                wins = process_eeg(sig)
            if wins is not None and len(wins) > 0:
                probs_eeg = predict_eeg(wins)
                p_eeg = float(probs_eeg.mean())
                has_eeg = 1
                st.success(f"✅ EEG — ADHD Probability: {p_eeg:.1%}")
            else:
                st.error("No valid windows extracted from EEG.")
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
            st.success(f"✅ Hyperaktiv — ADHD Probability: {p_hyp:.1%}")
        except Exception as e:
            st.error(f"Hyperaktiv Error: {e}")

    # ── META Decision ─────────────────────────────────────
    if p_eeg is not None or p_hyp is not None:
        st.subheader("🎯 META Model Decision")

        x_meta = np.array([[
            p_eeg  if p_eeg  is not None else 0.5,
            p_hyp  if p_hyp  is not None else 0.5,
            has_eeg, has_hyp
        ]])
        meta_prob = meta_model.predict_proba(x_meta)[0, 1]
        label = "ADHD 🔴" if meta_prob >= 0.5 else "Control 🟢"

        c1, c2, c3 = st.columns(3)
        if p_eeg is not None:
            c1.metric("EEG Probability",  f"{p_eeg:.1%}")
        if p_hyp is not None:
            c2.metric("HYP Probability",  f"{p_hyp:.1%}")
        c3.metric("Final Diagnosis", label)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=meta_prob * 100,
            number={"suffix": "%"},
            title={"text": "ADHD Probability (META)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#E74C3C" if meta_prob >= 0.5 else "#2ECC71"},
                "steps": [
                    {"range": [0,  50], "color": "#D5F5E3"},
                    {"range": [50, 100], "color": "#FADBD8"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
            }
        ))
        fig_g.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

        if has_eeg and has_hyp:
            src = "Both sources (EEG + Hyperaktiv)"
        elif has_eeg:
            src = "EEG only"
        else:
            src = "Hyperaktiv only"
        st.info(f"Decision based on: {src}")

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
            "Diagnosis": "ADHD 🔴" if meta_prob >= 0.5 else "Control 🟢",
            "Source":    src
        })

        # ── PDF Download ──────────────────────────────────
        st.markdown("---")
        st.subheader("📄 Download Patient Report")

        if not patient_name.strip():
            st.warning("Please enter the patient's name to generate the report.")
        else:
            if st.button("📥 Generate & Download PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    pdf_buffer = generate_pdf_report(
                        patient_name   = patient_name,
                        patient_age    = patient_age,
                        patient_gender = patient_gender,
                        p_eeg          = p_eeg   if p_eeg  is not None else 0.5,
                        p_hyp          = p_hyp   if p_hyp  is not None else 0.5,
                        meta_prob      = meta_prob,
                        has_eeg        = has_eeg,
                        has_hyp        = has_hyp
                    )
                fname = f"Cortex_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label    = "⬇️ Download Report PDF",
                    data     = pdf_buffer,
                    file_name= fname,
                    mime     = "application/pdf"
                )
                st.success("Report ready! Click the button above to download.")
    else:
        st.warning("Please upload at least one file to begin.")


# ══════════════════════════════════════════════════════════
# Page 5 — History
# ══════════════════════════════════════════════════════════
elif mode == "🗂️ History":
    st.title("🗂️ Diagnosis History")
    st.markdown("Records are saved for this session only and will be cleared when the app is closed.")
    st.markdown("---")

    if not st.session_state.diagnosis_history:
        st.info("No diagnoses recorded yet. Run a META Fusion diagnosis to start.")
    else:
        df_history = pd.DataFrame(st.session_state.diagnosis_history)

        # ── Summary metrics ───────────────────────────────
        total  = len(df_history)
        adhd_n = (df_history["Diagnosis"] == "ADHD 🔴").sum()
        ctrl_n = total - adhd_n

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Diagnoses", total)
        c2.metric("ADHD Detected",   int(adhd_n))
        c3.metric("Control",         int(ctrl_n))

        st.markdown("---")

        # ── Records table ─────────────────────────────────
        st.subheader("Session Records")
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # ── Clear button ──────────────────────────────────
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.diagnosis_history = []
            st.rerun()
