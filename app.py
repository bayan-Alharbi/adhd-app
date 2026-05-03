import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import onnxruntime as ort
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ADHD Diagnostic System",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_models():
    cnn  = ort.InferenceSession("models/cnn_eeg_model.onnx")
    lr   = joblib.load("models/lr_hyperaktiv.pkl")
    meta = joblib.load("models/meta_model.pkl")
    sc   = joblib.load("models/scaler_hyp.pkl")
    sel  = joblib.load("models/selector.pkl")
    return cnn, lr, meta, sc, sel

cnn_model, lr_base, meta_model, scaler_hyp, selector = load_models()

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

st.sidebar.title("🧠 ADHD Diagnostic")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode:", [
    "📊 Model Results",
    "🔬 EEG Analysis",
    "📋 Hyperaktiv Analysis",
    "🧠 META Fusion Diagnosis"
])

# ══════════════════════════════════════════════════════════
# Page 1 — Model Results
# ══════════════════════════════════════════════════════════
if mode == "📊 Model Results":
    st.title("📊 ADHD Diagnostic System — Model Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CNN — AUC",  "0.995")
    col2.metric("LR — AUC",   "0.958")
    col3.metric("META — AUC", "0.993")
    col4.metric("META F1",    "93.6%")

    st.markdown("---")
    st.subheader("Model Comparison")
    df = pd.DataFrame({
        "Model":     ["CNN (EEG)", "LR (Hyperaktiv)", "META Fusion"],
        "Accuracy":  ["93.2%", "82.4%", "93.7%"],
        "Precision": ["88.0%", "80.0%", "89.0%"],
        "Recall":    ["99.0%", "88.9%", "98.8%"],
        "F1-Score":  ["93.2%", "84.2%", "93.6%"],
        "AUC":       ["0.995", "0.958", "0.993"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Performance Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="CNN",  x=metrics,
                         y=[0.932, 0.880, 0.990, 0.932, 0.995],
                         marker_color="#E74C3C"))
    fig.add_trace(go.Bar(name="LR",   x=metrics,
                         y=[0.824, 0.800, 0.889, 0.842, 0.958],
                         marker_color="#3498DB"))
    fig.add_trace(go.Bar(name="META", x=metrics,
                         y=[0.937, 0.890, 0.988, 0.936, 0.993],
                         marker_color="#2ECC71"))
    fig.update_layout(barmode="group", yaxis_range=[0.7, 1.05],
                      height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("META Model Equation")
    st.code("z = -3.650 + 5.164·p_eeg + 1.958·p_hyp + (-0.357)·has_eeg + 0.371·has_hyp")
    st.info("EEG contributes more than Hyperaktiv to the final decision (5.16 vs 1.96)")

# ══════════════════════════════════════════════════════════
# Page 2 — EEG Analysis
# ══════════════════════════════════════════════════════════
elif mode == "🔬 EEG Analysis":
    st.title("🔬 EEG Signal Analysis")
    st.markdown("Upload a `.mat` file to get a diagnosis.")

    uploaded = st.file_uploader("Upload EEG file", type=["mat"])
    if uploaded:
        import scipy.io, tempfile, os
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

            st.success(f"File loaded | Shape: {sig.shape}")
            with st.spinner("Analyzing..."):
                wins = process_eeg(sig)

            if wins is not None and len(wins) > 0:
                probs = predict_eeg(wins)
                mean_prob = probs.mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("ADHD Probability", f"{mean_prob:.1%}")
                c2.metric("Windows",          str(len(wins)))
                c3.metric("Diagnosis",
                          "ADHD 🔴" if mean_prob >= 0.5 else "Control 🟢")

                fig2 = go.Figure(go.Histogram(
                    x=probs, nbinsx=30, marker_color="#E74C3C"))
                fig2.add_vline(x=0.5, line_dash="dash",
                               annotation_text="Threshold = 0.5")
                fig2.update_layout(
                    title="Window Probability Distribution",
                    xaxis_title="P(ADHD)",
                    yaxis_title="Count",
                    height=300, template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("No valid windows extracted from signal.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════
# Page 3 — Hyperaktiv Analysis
# ══════════════════════════════════════════════════════════
elif mode == "📋 Hyperaktiv Analysis":
    st.title("📋 Hyperaktiv Data Analysis")
    st.markdown("Upload a `features.csv` file to get predictions.")

    f = st.file_uploader("Upload features.csv", type=["csv"])
    if f:
        df_in = pd.read_csv(f, sep=";").fillna(0)
        st.dataframe(df_in.head(), use_container_width=True)

        if st.button("🔍 Run Analysis", type="primary"):
            try:
                ids = df_in["ID"] if "ID" in df_in.columns else None
                X = df_in.drop(columns=["ID"], errors="ignore").values
                X_sel = selector.transform(scaler_hyp.transform(X))
                probs = lr_base.predict_proba(X_sel)[:, 1]

                x_meta = np.column_stack([
                    np.full(len(probs), 0.5), probs,
                    np.zeros(len(probs)), np.ones(len(probs))
                ])
                meta_probs = meta_model.predict_proba(x_meta)[:, 1]

                result_df = pd.DataFrame({
                    "ID":        ids if ids is not None else range(len(probs)),
                    "LR Prob":   np.round(probs, 3),
                    "META Prob": np.round(meta_probs, 3),
                    "Diagnosis": ["ADHD 🔴" if p >= 0.5 else "Control 🟢"
                                  for p in meta_probs]
                })
                st.dataframe(result_df, use_container_width=True,
                             hide_index=True)
                adhd_n = (meta_probs >= 0.5).sum()
                st.info(f"ADHD: {adhd_n} | Control: {len(probs) - adhd_n}")
            except Exception as e:
                st.error(f"Processing error: {e}")

# ══════════════════════════════════════════════════════════
# Page 4 — META Fusion Diagnosis
# ══════════════════════════════════════════════════════════
elif mode == "🧠 META Fusion Diagnosis":
    st.title("🧠 META Fusion Diagnosis — EEG + Hyperaktiv")
    st.markdown("Upload both files to get the final META model decision.")
    st.markdown("---")

    col_eeg, col_hyp = st.columns(2)
    with col_eeg:
        st.subheader("📡 EEG File")
        eeg_file = st.file_uploader("Upload .mat file", type=["mat"],
                                     key="meta_eeg")
    with col_hyp:
        st.subheader("📋 Hyperaktiv File")
        hyp_file = st.file_uploader("Upload features.csv", type=["csv"],
                                     key="meta_hyp")

    st.markdown("---")

    p_eeg   = None
    p_hyp   = None
    has_eeg = 0
    has_hyp = 0

    if eeg_file:
        import scipy.io, tempfile, os
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
            X = df_hyp.drop(columns=["ID"], errors="ignore").values
            X_sel = selector.transform(scaler_hyp.transform(X))
            probs_hyp = lr_base.predict_proba(X_sel)[:, 1]
            p_hyp = float(probs_hyp.mean())
            has_hyp = 1
            st.success(f"✅ Hyperaktiv — ADHD Probability: {p_hyp:.1%}")
        except Exception as e:
            st.error(f"Hyperaktiv Error: {e}")

    if p_eeg is not None or p_hyp is not None:
        st.markdown("---")
        st.subheader("🎯 META Model Decision")

        x_meta = np.array([[
            p_eeg  if p_eeg is not None else 0.5,
            p_hyp  if p_hyp is not None else 0.5,
            has_eeg,
            has_hyp
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
                "bar":  {"color": "#E74C3C" if meta_prob >= 0.5
                                  else "#2ECC71"},
                "steps": [
                    {"range": [0,  50], "color": "#D5F5E3"},
                    {"range": [50, 100], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": 50
                }
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

    else:
        st.warning("Please upload at least one file to begin.")
