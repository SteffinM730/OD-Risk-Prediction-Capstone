import sys
try:
    import streamlit as st
except Exception as e:
    print("Streamlit is not installed or failed to import. To run the dashboard, create a virtual environment and install dependencies: pip install -r requirements.txt")
    print("Error detail:", e)
    sys.exit(1)

import pandas as pd
import os
import matplotlib.pyplot as plt

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Business Risk & OD Dashboard",
    layout="wide"
)

st.title("📊 Business Risk & OD Approval Dashboard")
st.markdown("---")

# =====================================================
# Load Output Data
# =====================================================
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visualizations")

ranking_path = os.path.join(OUTPUT_DIR, "business_type_od_ranking.csv")

if not os.path.exists(ranking_path):
    st.error("⚠ Please run Main.py first to generate outputs.")
    st.stop()

df = pd.read_csv(ranking_path, index_col=0)

# =====================================================
# KPI SECTION
# =====================================================
st.subheader("📌 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric(
    "🏆 Best Business Type (OD)",
    df.index[0]
)

col2.metric(
    "⚠ Highest Risk Business Type",
    df.index[-1]
)

col3.metric(
    "💰 Best OD Approval Score",
    round(df["OD_Approval_Score"].max(), 3)
)

st.markdown("---")

# =====================================================
# Interactive Business Selector
# =====================================================
st.subheader("🔍 Business Type Explorer")

selected_business = st.selectbox(
    "Select a Business Type:",
    df.index.tolist()
)

business_data = df.loc[selected_business]

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Rate",
            round(business_data["High_Risk_Rate"], 3))

col2.metric("Avg OD Utilization",
            round(business_data["Avg_OD_Utilization"], 3))

col3.metric("OD Approval Score",
            round(business_data["OD_Approval_Score"], 3))

st.markdown("---")

# =====================================================
# Ranking Table
# =====================================================
st.subheader("📋 Full Business Ranking (Best First)")
st.dataframe(df, use_container_width=True)

# =====================================================
# Risk Rate Chart
# =====================================================
st.subheader("📉 High Risk Rate Comparison")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(df.index, df["High_Risk_Rate"])
plt.xticks(rotation=45)
plt.ylabel("High Risk Rate")
plt.title("Business Risk Comparison")
plt.tight_layout()

st.pyplot(fig1)

# =====================================================
# OD Approval Score Chart
# =====================================================
st.subheader("💰 OD Approval Score Comparison")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(df.index, df["OD_Approval_Score"])
plt.xticks(rotation=45)
plt.ylabel("OD Approval Score")
plt.title("OD Approval Score by Business Type")
plt.tight_layout()

st.pyplot(fig2)

# =====================================================
# Cluster Visualizations
# =====================================================
st.markdown("---")
st.subheader("📍 Clustering Visualizations")

col1, col2 = st.columns(2)

image_2d = os.path.join(VISUAL_DIR, "cluster_cloud_2d.png")
image_3d = os.path.join(VISUAL_DIR, "cluster_cloud_3d.png")

if os.path.exists(image_2d):
    col1.image(image_2d, caption="PCA 2D Cluster View")

if os.path.exists(image_3d):
    col2.image(image_3d, caption="PCA 3D Cluster View")

# =====================================================
# ANN Performance
# =====================================================
st.markdown("---")
st.subheader("🧠 ANN Model Performance")

metrics_path = os.path.join(OUTPUT_DIR, "ann_metrics_report.txt")
roc_path = os.path.join(OUTPUT_DIR, "ann_roc_curve.png")
training_path = os.path.join(OUTPUT_DIR, "training_metrics.png")

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        st.text(f.read())

col1, col2 = st.columns(2)

if os.path.exists(roc_path):
    col1.image(roc_path, caption="ROC Curve")

if os.path.exists(training_path):
    col2.image(training_path, caption="Training Loss & Accuracy")

st.markdown("---")

st.success("✅ Dashboard Running Successfully")