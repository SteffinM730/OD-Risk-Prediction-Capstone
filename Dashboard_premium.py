import sys
try:
    import streamlit as st
except Exception as e:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Error:", e)
    sys.exit(1)

import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Executive Risk & OD Dashboard",
    layout="wide"
)

# =====================================================
# EXECUTIVE STYLING
# =====================================================
st.markdown("""
<style>
.big-title {
    font-size:32px !important;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Executive Business Risk & OD Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visualizations")
MODEL_PATH = os.path.join(OUTPUT_DIR, "risk_ann_model.pth")
ranking_path = os.path.join(OUTPUT_DIR, "business_type_od_ranking.csv")

if not os.path.exists(ranking_path):
    st.error("⚠ Run Main.py first to generate outputs.")
    st.stop()

df = pd.read_csv(ranking_path, index_col=0)

# =====================================================
# KPI SECTION
# =====================================================
st.subheader("📌 Executive KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("🏆 Best Business Type", df.index[0])
col2.metric("⚠ Highest Risk Business", df.index[-1])
col3.metric("💰 Best OD Score", round(df["OD_Approval_Score"].max(), 3))

st.markdown("---")

# =====================================================
# 🎛 DYNAMIC OD WEIGHT CONTROLS
# =====================================================
st.subheader("⚙ Adjustable OD Scoring Weights")

w1 = st.slider("Risk Weight", 0.0, 1.0, 0.5)
w2 = st.slider("OD Utilization Weight", 0.0, 1.0, 0.3)
w3 = st.slider("Capital Strength Weight", 0.0, 1.0, 0.2)

df["Dynamic_OD_Score"] = (
    (1 - df["High_Risk_Rate"]) * w1 +
    (1 - df["Avg_OD_Utilization"]) * w2 +
    (df["Avg_Capital"]) * w3
)

df_dynamic = df.sort_values("Dynamic_OD_Score", ascending=False)

st.subheader("🏆 Live OD Ranking (Based on Selected Weights)")
st.dataframe(df_dynamic, width="stretch")

st.markdown("---")

# =====================================================
# 📊 CHARTS
# =====================================================
st.subheader("📉 Risk Comparison")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(df.index, df["High_Risk_Rate"])
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

st.subheader("💰 Dynamic OD Score Comparison")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(df_dynamic.index, df_dynamic["Dynamic_OD_Score"])
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# =====================================================
# 🧠 LIVE ANN RISK PREDICTION
# =====================================================
st.subheader("🧠 Live Risk Prediction Simulator")

if os.path.exists(MODEL_PATH):

    # -------------------------------
    # Load Model Checkpoint
    # -------------------------------
    checkpoint = torch.load(MODEL_PATH)

    input_dim = checkpoint["input_dim"]
    feature_columns = checkpoint["feature_columns"]

    # -------------------------------
    # Define ANN Architecture
    # -------------------------------
    class RiskANN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    model = RiskANN(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    st.markdown("Enter **scaled feature values** used during training:")

    user_inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(feature_columns):
        col = cols[i % 3]
        user_inputs[feature] = col.number_input(feature, value=0.0)

    if st.button("Predict Risk"):

        input_array = np.array(
            [user_inputs[feature] for feature in feature_columns]
        )

        tensor_input = torch.tensor(
            input_array.reshape(1, -1),
            dtype=torch.float32
        )

        with torch.no_grad():
            risk_prob = model(tensor_input).item()

        st.metric("Predicted Risk Probability", round(risk_prob, 4))

        if risk_prob < 0.4:
            st.success("Low Risk – Ideal for OD approval")
        elif risk_prob < 0.7:
            st.warning("Moderate Risk – Controlled OD")
        else:
            st.error("High Risk – Avoid OD / High Interest")

else:
    st.warning("ANN model not found. Run Main.py first.")

st.markdown("---")

# =====================================================
# 📍 CLUSTER VISUALS
# =====================================================
st.subheader("📍 Clustering Insights")

image_2d = os.path.join(VISUAL_DIR, "cluster_cloud_2d.png")
image_3d = os.path.join(VISUAL_DIR, "cluster_cloud_3d.png")

col1, col2 = st.columns(2)

if os.path.exists(image_2d):
    col1.image(image_2d, caption="PCA 2D Cluster View")

if os.path.exists(image_3d):
    col2.image(image_3d, caption="PCA 3D Cluster View")

st.success("✅ Executive Dashboard Running Successfully")