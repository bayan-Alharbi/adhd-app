# =========================================================
# ADHD NEUROVISION — PREMIUM VISUAL IDENTITY EDITION
# Inspired by futuristic portfolio / AI medical branding
# Full visual redesign for Streamlit
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import onnxruntime as ort
from scipy import signal
import warnings
import time
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ADHD NeuroVision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# BRAND CSS — FUTURISTIC DARK AI MEDICAL STYLE
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(139,92,246,0.18), transparent 30%),
        radial-gradient(circle at bottom left, rgba(59,130,246,0.14), transparent 25%),
        linear-gradient(135deg, #050816 0%, #0B1023 45%, #111827 100%);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1023 0%, #111827 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
}

/* Headers */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

/* Paragraphs */
p, div, label, span {
    color: #CBD5E1;
}

/* Cards */
.stMetric {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(139,92,246,0.22);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 18px rgba(139,92,246,0.12);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #3B82F6);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    height: 3.2em;
    width: 100%;
    box-shadow: 0 0 15px rgba(124,58,237,0.35);
}

/* Inputs */
.stFileUploader, .stTextInput, .stSelectbox {
    background-color: rgba(15,23,42,0.55);
    border-radius: 14px;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: rgba(15,23,42,0.6);
    border-radius: 18px;
    padding: 10px;
}

/* Block spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Custom Hero */
.hero-box {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(135deg,
        rgba(124,58,237,0.18),
        rgba(59,130,246,0.12));
    border: 1px solid rgba(139,92,246,0.22);
    box-shadow: 0 0 30px rgba(124,58,237,0.15);
}

/* Section Divider */
hr {
    border: 1px solid rgba(139,92,246,0.15);
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero-box">
    <h1 style="font-size:52px; margin-bottom:10px;">
        🧠 ADHD NeuroVision
    </h1>
    <p style="font-size:20px; color:#C4B5FD;">
        AI-Powered Cognitive Diagnostic Platform
    </p>
    <p style="font-size:16px;">
        Advanced EEG + Hyperaktiv + META Fusion Intelligence for Precision ADHD Detection
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_models():
    cnn  = ort.InferenceSession("models/cnn_eeg_model.onnx")
    lr   = joblib.load("models/lr_hyperaktiv.pkl")
    meta = joblib.load("models/meta_model.pkl")
    sc   = joblib.load("models/scaler_hyp.pkl")
    sel  = joblib.load("models/selector.pkl")
    return cnn, lr, meta, sc, sel

cnn_model, lr_base, meta_model, scaler_hyp, selector = load_models()

# =========================================================
# EEG PROCESSING
# =========================================================
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

# =========================================================
# SIDEBAR BRAND
# =========================================================
st.sidebar.markdown("""
<h1 style='text-align:center; color:#C4B5FD;'>🧠</h1>
<h2 style='text-align:center;'>NeuroVision</h2>
<p style='text-align:center; color:#94A3B8;'>
Clinical AI Suite
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Platform Modules",
    [
        "📊 Performance Intelligence",
        "🔬 EEG Neural Analysis",
        "📋 Hyperaktiv Behavior Scan",
        "🧠 META Fusion Core"
    ]
)

# =========================================================
# PAGE 1
# =========================================================
if mode == "📊 Performance Intelligence":

    st.subheader("📊 Diagnostic Performance Matrix")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CNN Neural AUC", "0.995")
    c2.metric("LR Behavioral AUC", "0.958")
    c3.metric("META Fusion AUC", "0.993")
    c4.metric("META F1 Score", "93.6%")

    st.markdown("---")

    df = pd.DataFrame({
        "Model": ["CNN EEG", "LR Hyperaktiv", "META Fusion"],
        "Accuracy": [93.2, 82.4, 93.7],
        "Precision": [88.0, 80.0, 89.0],
        "Recall": [99.0, 88.9, 98.8],
        "F1": [93.2, 84.2, 93.6],
        "AUC": [99.5, 95.8, 99.3]
    })

    st.dataframe(
        df.style.highlight_max(axis=0, color="#312E81"),
        use_container_width=True,
        hide_index=True
    )

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[0.932, 0.880, 0.990, 0.932, 0.995],
        theta=metrics,
        fill='toself',
        name='CNN'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0.937, 0.890, 0.988, 0.936, 0.993],
        theta=metrics,
        fill='toself',
        name='META'
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PAGE 2
# =========================================================
elif mode == "🔬 EEG Neural Analysis":

    st.subheader("🔬 Neural EEG Diagnostic Engine")

    uploaded = st.file_uploader("Upload EEG Neural Matrix (.mat)", type=["mat"])

    if uploaded:

        import scipy.io
        import tempfile
        import os

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

            st.success(f"Neural Matrix Loaded | Shape: {sig.shape}")

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            wins = process_eeg(sig)

            if wins is not None:

                probs = predict_eeg(wins)
                mean_prob = probs.mean()

                a, b, c = st.columns(3)
                a.metric("ADHD Neural Probability", f"{mean_prob:.1%}")
                b.metric("EEG Segments", len(wins))
                c.metric("Cognitive Status",
                         "ADHD 🔴" if mean_prob >= 0.5 else "Control 🟢")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mean_prob * 100,
                    number={'suffix': "%"},
                    title={'text': "Neuro Probability Index"},
                    gauge={'axis': {'range': [0, 100]}}
                ))

                fig.update_layout(template="plotly_dark", height=420)

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No valid neural windows detected.")

        finally:
            os.unlink(tmp_path)

# =========================================================
# PAGE 3
# =========================================================
elif mode == "📋 Hyperaktiv Behavior Scan":

    st.subheader("📋 Behavioral Biomarker Analysis")

    f = st.file_uploader("Upload Hyperaktiv Dataset (.csv)", type=["csv"])

    if f:

        df_in = pd.read_csv(f, sep=";").fillna(0)

        st.dataframe(df_in.head(), use_container_width=True)

        if st.button("Run Behavioral Intelligence"):

            X = df_in.drop(columns=["ID"], errors="ignore").values

            X_sel = selector.transform(
                scaler_hyp.transform(X)
            )

            probs = lr_base.predict_proba(X_sel)[:, 1]

            result_df = pd.DataFrame({
                "Probability": np.round(probs, 3),
                "Diagnosis": [
                    "ADHD 🔴" if p >= 0.5 else "Control 🟢"
                    for p in probs
                ]
            })

            st.dataframe(result_df, use_container_width=True)

# =========================================================
# PAGE 4
# =========================================================
elif mode == "🧠 META Fusion Core":

    st.subheader("🧠 META Fusion Cognitive Intelligence")

    st.info("Upload EEG + Hyperaktiv to activate full multimodal fusion.")

    eeg_file = st.file_uploader("Upload EEG", type=["mat"])
    hyp_file = st.file_uploader("Upload Hyperaktiv", type=["csv"])

    if eeg_file or hyp_file:
        st.success("Fusion Core Ready — integrate your backend logic here.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#94A3B8;'>
NeuroVision™ | Precision Cognitive AI
</p>
""", unsafe_allow_html=True)
