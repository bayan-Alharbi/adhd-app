# =========================================================
# ADHD DIAGNOSTIC SYSTEM — UI/UX ENHANCED VERSION
# Copy-paste directly into Streamlit
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
    page_title="ADHD Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #E2E8F0;
}

h1, h2, h3 {
    color: #1E3A8A;
    font-weight: 700;
}

.stMetric {
    background-color: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    text-align: center;
}

.stButton>button {
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    background-color: #2563EB;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>🧠 ADHD Multi-Modal Diagnostic System</h1>
<p style='text-align:center; font-size:18px; color:gray;'>
AI-Powered EEG + Hyperaktiv + META Fusion Diagnosis
</p>
""", unsafe_allow_html=True)

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
# SIGNAL PROCESSING
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
# SIDEBAR
# =========================================================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
    width=90
)

st.sidebar.title("ADHD Diagnostic Suite")
st.sidebar.caption("Clinical Decision Support")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Navigation",
    [
        "📊 Model Results",
        "🔬 EEG Analysis",
        "📋 Hyperaktiv Analysis",
        "🧠 META Fusion Diagnosis"
    ]
)

# =========================================================
# PAGE 1 — MODEL RESULTS
# =========================================================
if mode == "📊 Model Results":

    st.header("📊 Model Performance Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CNN AUC", "0.995")
    c2.metric("LR AUC", "0.958")
    c3.metric("META AUC", "0.993")
    c4.metric("META F1", "93.6%")

    st.markdown("---")

    st.subheader("📋 Performance Comparison Table")

    df = pd.DataFrame({
        "Model": ["CNN (EEG)", "LR (Hyperaktiv)", "META Fusion"],
        "Accuracy": [93.2, 82.4, 93.7],
        "Precision": [88.0, 80.0, 89.0],
        "Recall": [99.0, 88.9, 98.8],
        "F1": [93.2, 84.2, 93.6],
        "AUC": [99.5, 95.8, 99.3]
    })

    st.dataframe(
        df.style.highlight_max(axis=0, color="#D1FAE5"),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Interactive Comparison")

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="CNN",
        x=metrics,
        y=[0.932, 0.880, 0.990, 0.932, 0.995]
    ))

    fig.add_trace(go.Bar(
        name="LR",
        x=metrics,
        y=[0.824, 0.800, 0.889, 0.842, 0.958]
    ))

    fig.add_trace(go.Bar(
        name="META",
        x=metrics,
        y=[0.937, 0.890, 0.988, 0.936, 0.993]
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=450,
        yaxis_range=[0.7, 1.05]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧮 META Equation")

    st.code(
        "z = -3.650 + 5.164·p_eeg + 1.958·p_hyp + (-0.357)·has_eeg + 0.371·has_hyp"
    )

    st.info("EEG contributes more strongly than Hyperaktiv to final classification.")

# =========================================================
# PAGE 2 — EEG ANALYSIS
# =========================================================
elif mode == "🔬 EEG Analysis":

    st.header("🔬 EEG Signal Diagnostic")

    uploaded = st.file_uploader("Upload EEG .mat File", type=["mat"])

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

            st.success(f"Loaded EEG Shape: {sig.shape}")

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            wins = process_eeg(sig)

            if wins is not None:

                probs = predict_eeg(wins)
                mean_prob = probs.mean()

                a, b, c = st.columns(3)

                a.metric("ADHD Probability", f"{mean_prob:.1%}")
                b.metric("Windows", len(wins))
                c.metric(
                    "Diagnosis",
                    "ADHD 🔴" if mean_prob >= 0.5 else "Control 🟢"
                )

                fig = go.Figure(
                    go.Histogram(
                        x=probs,
                        nbinsx=30
                    )
                )

                fig.add_vline(
                    x=0.5,
                    line_dash="dash",
                    annotation_text="Threshold"
                )

                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    title="Window-Level ADHD Probability"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No valid EEG windows extracted.")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            os.unlink(tmp_path)

# =========================================================
# PAGE 3 — HYPERAKTIV
# =========================================================
elif mode == "📋 Hyperaktiv Analysis":

    st.header("📋 Hyperaktiv Feature Analysis")

    f = st.file_uploader("Upload features.csv", type=["csv"])

    if f:

        df_in = pd.read_csv(f, sep=";").fillna(0)

        st.dataframe(df_in.head(), use_container_width=True)

        if st.button("🔍 Run Hyperaktiv Diagnosis"):

            try:
                ids = df_in["ID"] if "ID" in df_in.columns else None

                X = df_in.drop(columns=["ID"], errors="ignore").values

                X_sel = selector.transform(
                    scaler_hyp.transform(X)
                )

                probs = lr_base.predict_proba(X_sel)[:, 1]

                result_df = pd.DataFrame({
                    "ID": ids if ids is not None else range(len(probs)),
                    "Probability": np.round(probs, 3),
                    "Diagnosis": [
                        "ADHD 🔴" if p >= 0.5 else "Control 🟢"
                        for p in probs
                    ]
                })

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.success("Analysis Complete")

            except Exception as e:
                st.error(f"Processing Error: {e}")

# =========================================================
# PAGE 4 — META FUSION
# =========================================================
elif mode == "🧠 META Fusion Diagnosis":

    st.header("🧠 META Fusion Diagnosis")

    st.markdown("Upload EEG and/or Hyperaktiv data for final prediction.")

    eeg_file = st.file_uploader("Upload EEG (.mat)", type=["mat"])
    hyp_file = st.file_uploader("Upload Hyperaktiv (.csv)", type=["csv"])

    if eeg_file or hyp_file:

        st.info("META Fusion logic can now be inserted here using your original backend.")

        st.warning("Use your previous META backend code block inside this section.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed for Graduation Project | ADHD AI Diagnostic Platform")
