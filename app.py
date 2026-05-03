import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
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
    cnn  = tf.keras.models.load_model("models/cnn_eeg_model.h5")
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

st.sidebar.title("🧠 ADHD Diagnostic")
st.sidebar.markdown("---")
mode = st.sidebar.radio("اختر الوضع:", [
    "📊 نتائج الموديل",
    "🔬 تحليل EEG",
    "📋 تحليل Hyperaktiv"
])

if mode == "📊 نتائج الموديل":
    st.title("📊 نتائج النظام التشخيصي")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CNN — AUC",  "0.995")
    col2.metric("LR — AUC",   "0.958")
    col3.metric("META — AUC", "0.993")
    col4.metric("META F1",    "93.6%")

    st.markdown("---")
    st.subheader("مقارنة الموديلات")

    df = pd.DataFrame({
        "الموديل":   ["CNN (EEG)", "LR (Hyperaktiv)", "META Fusion"],
        "Accuracy":  ["93.2%", "82.4%", "93.7%"],
        "Precision": ["88.0%", "80.0%", "89.0%"],
        "Recall":    ["99.0%", "88.9%", "98.8%"],
        "F1-Score":  ["93.2%", "84.2%", "93.6%"],
        "AUC":       ["0.995", "0.958", "0.993"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("مقارنة بصرية")
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

    st.subheader("معادلة الـ META Model")
    st.code("z = -3.650 + 5.164·p_eeg + 1.958·p_hyp + (-0.357)·has_eeg + 0.371·has_hyp")
    st.info("EEG يساهم أكثر من Hyperaktiv في القرار (5.16 مقابل 1.96)")

elif mode == "🔬 تحليل EEG":
    st.title("🔬 تحليل إشارة EEG")
    st.markdown("ارفعي ملف .mat للتشخيص")

    uploaded = st.file_uploader("ارفعي ملف EEG", type=["mat"])

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

            st.success(f"تم التحميل | الشكل: {sig.shape}")

            with st.spinner("جارٍ التحليل..."):
                wins = process_eeg(sig)

            if wins is not None and len(wins) > 0:
                probs = cnn_model.predict(wins, verbose=0).ravel()
                mean_prob = probs.mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("احتمالية ADHD", f"{mean_prob:.1%}")
                c2.metric("عدد النوافذ",   str(len(wins)))
                c3.metric("التشخيص",
                          "ADHD 🔴" if mean_prob >= 0.5 else "Control 🟢")

                fig2 = go.Figure(go.Histogram(
                    x=probs, nbinsx=30, marker_color="#E74C3C"))
                fig2.add_vline(x=0.5, line_dash="dash",
                               annotation_text="threshold=0.5")
                fig2.update_layout(title="توزيع احتمالات النوافذ",
                                   height=300, template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("لم يتم استخراج نوافذ صالحة")
        except Exception as e:
            st.error(f"خطأ: {e}")
        finally:
            os.unlink(tmp_path)

elif mode == "📋 تحليل Hyperaktiv":
    st.title("📋 تحليل بيانات Hyperaktiv")

    f = st.file_uploader("ارفعي ملف features.csv", type=["csv"])
    if f:
        df_in = pd.read_csv(f, sep=";").fillna(0)
        st.dataframe(df_in.head(), use_container_width=True)

        if st.button("🔍 تحليل الكل", type="primary"):
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
                    "التشخيص":  ["ADHD 🔴" if p >= 0.5 else "Control 🟢"
                                  for p in meta_probs]
                })
                st.dataframe(result_df, use_container_width=True,
                             hide_index=True)
                adhd_n = (meta_probs >= 0.5).sum()
                st.info(f"ADHD: {adhd_n} | Control: {len(probs) - adhd_n}")
            except Exception as e:
                st.error(f"خطأ في المعالجة: {e}")
