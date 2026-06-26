"""Predictive Maintenance — Streamlit app.

Live machine-failure prediction from sensor inputs, with a SHAP explanation
of the prediction and a model-performance overview.

Run: streamlit run app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from data import MODEL_FEATURES, engineer_features  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["clf"])


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def clean(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


model = load_model()
metrics = load_metrics()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("⚙️ Machine sensor readings")
type_ = st.sidebar.selectbox(
    "Product quality variant", ["L", "M", "H"],
    help="L = low, M = medium, H = high quality variant",
)
air = st.sidebar.slider("Air temperature [K]", 295.0, 305.0, 298.0, 0.1)
proc = st.sidebar.slider("Process temperature [K]", 305.0, 314.0, 308.5, 0.1)
rpm = st.sidebar.slider("Rotational speed [rpm]", 1160, 2890, 1500, 1)
torque = st.sidebar.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.1)
wear = st.sidebar.slider("Tool wear [min]", 0, 260, 100, 1)

raw = pd.DataFrame([{
    "Type": type_,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": wear,
}])
X = engineer_features(raw)[MODEL_FEATURES]
proba = float(model.predict_proba(X)[0, 1])

# ------------------------------------------------------------------ header
st.title("🛠️ Predictive Maintenance — Machine Failure Prediction")
st.caption(
    "Predicts the probability that a milling machine will fail, from live sensor "
    "readings. Model: XGBoost on the UCI AI4I 2020 dataset. Adjust the sliders on the left."
)

tab_predict, tab_perf, tab_about = st.tabs(["🔮 Prediction", "📊 Model performance", "ℹ️ About"])

with tab_predict:
    col1, col2 = st.columns([1, 1])

    with col1:
        if proba >= 0.6:
            level, color = "HIGH RISK", "#d62728"
        elif proba >= 0.3:
            level, color = "MEDIUM RISK", "#ff7f0e"
        else:
            level, color = "LOW RISK", "#2ca02c"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": " %"},
            title={"text": f"Failure probability — <b>{level}</b>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 60], "color": "#fff3e0"},
                    {"range": [60, 100], "color": "#ffebee"},
                ],
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(gauge, width="stretch")

        if proba >= 0.6:
            st.error("⚠️ Schedule maintenance now — high chance of failure.")
        elif proba >= 0.3:
            st.warning("🔧 Inspect soon — elevated risk.")
        else:
            st.success("✅ Machine looks healthy — normal operation.")

    with col2:
        st.subheader("Why this prediction?")
        st.caption("SHAP feature contributions — red pushes toward failure, blue toward healthy.")
        explainer = get_explainer(model)
        Xt = model.named_steps["prep"].transform(X)
        names = [clean(n) for n in model.named_steps["prep"].get_feature_names_out()]
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        contrib = pd.Series(sv[0], index=names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.2))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib.values]
        ax.barh(contrib.index, contrib.values, color=colors)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on failure log-odds)")
        plt.tight_layout()
        st.pyplot(fig)

with tab_perf:
    if metrics:
        best = metrics.get("best_model", "model")
        res = metrics["results"][best]
        st.subheader(f"Best model: {best}")
        c1, c2, c3 = st.columns(3)
        c1.metric("PR-AUC", f"{res['test_pr_auc']:.3f}")
        c2.metric("ROC-AUC", f"{res['test_roc_auc']:.3f}")
        c3.metric("F1 (failure)", f"{res['test_f1']:.3f}")
        st.caption(
            f"Evaluated on {metrics['n_test']:,} held-out machines "
            f"(failure rate {metrics['positive_rate']:.1%})."
        )
        st.markdown("**Model comparison**")
        st.dataframe(
            pd.DataFrame(metrics["results"]).T[["test_pr_auc", "test_roc_auc", "test_f1"]]
            .rename(columns={"test_pr_auc": "PR-AUC", "test_roc_auc": "ROC-AUC", "test_f1": "F1"})
            .style.format("{:.3f}").background_gradient(cmap="Greens"),
            width="stretch",
        )
    for img, cap in [
        ("reports/roc_pr_curves.png", "ROC & Precision-Recall curves"),
        ("reports/confusion_matrix.png", "Confusion matrix"),
        ("reports/shap_summary.png", "Global feature importance (SHAP)"),
    ]:
        p = ROOT / img
        if p.exists():
            st.image(str(p), caption=cap)

with tab_about:
    st.markdown(
        """
**Predictive Maintenance** flags machines likely to fail so maintenance can be scheduled
*before* a breakdown — reducing downtime and cost.

- **Data:** [UCI AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 records.
- **Inputs:** product quality variant + 5 sensors (air & process temperature, rotational speed, torque, tool wear).
- **Engineered features:** temperature difference, mechanical power, tool-wear strain.
- **Model:** XGBoost with class-imbalance handling, evaluated with PR-AUC (the right metric for rare events).
- **Explainability:** SHAP, both global and per-prediction.

Built by **Oussama Elayeb**.
        """
    )
