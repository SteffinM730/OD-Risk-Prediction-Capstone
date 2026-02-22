"""
Risk Prediction Pipeline - Main Entry Point

Pipeline Steps:
1. Load and preprocess data
2. Scale features
3. Perform clustering analysis with PCA visualization
4. Train ANN for risk prediction
5. Rank Business Types for OD approval
6. Generate OD approval score
"""

import os
import pandas as pd

from src.data import load_dataset
from src.preprocessing import select_features, scale_numeric_features
from src.models import run_clustering_visuals, train_ann_for_risk


def main():

    print("=" * 70)
    print("RISK PREDICTION MODEL - PIPELINE EXECUTION")
    print("=" * 70)

    # =====================================================
    # Step 1: Load Dataset
    # =====================================================
    print("\n[Step 1] Loading dataset...")
    df = load_dataset("1L_real_world_business_financial_stress_dataset.csv")
    print(f"  OK Loaded data shape: {df.shape}")

    # =====================================================
    # Step 2: Feature Selection
    # =====================================================
    print("\n[Step 2] Selecting features...")
    df_cleaned = select_features(df)
    print(f"  OK Original Shape: {df.shape}")
    print(f"  OK Cleaned Shape: {df_cleaned.shape}")

    # =====================================================
    # Step 3: Scaling
    # =====================================================
    print("\n[Step 3] Scaling numeric features...")
    df_scaled, X_scaled, scaler = scale_numeric_features(df_cleaned)
    print("  OK Features scaled using StandardScaler")

    # =====================================================
    # Step 4: Clustering
    # =====================================================
    print("\n[Step 4] Performing clustering analysis...")
    df_clustered = run_clustering_visuals(df_scaled, X_scaled)
    print("  OK Clustering completed successfully")

    # -----------------------------------------------------
    # Business Type Distribution by Cluster
    # -----------------------------------------------------
    print("\nBusiness Type Distribution by Cluster:")

    cluster_business_summary = (
        df_clustered
        .groupby(["ClusterId", "Business_Type"])
        .size()
        .unstack(fill_value=0)
    )

    print(cluster_business_summary)

    # =====================================================
    # Step 5: ANN Training
    # =====================================================
    print("\n[Step 5] Training Artificial Neural Network...")
    ann_model, df_predictions = train_ann_for_risk(df_scaled)
    print("  OK ANN training completed successfully")

    # =====================================================
    # Step 6: Business-Type Risk Ranking
    # =====================================================
    print("\n[Step 6] Ranking Business Types for OD Approval...")

    business_risk_summary = (
        df_predictions.groupby("Business_Type")
        .agg(
            Total_Businesses=("Predicted_Risk_Class", "count"),
            High_Risk_Rate=("Predicted_Risk_Class", "mean"),
            Avg_Risk_Probability=("Predicted_Risk_Prob", "mean"),
            Avg_OD_Utilization=("OD_Utilization", "mean"),
            Avg_Capital=("Capital", "mean"),
            Avg_Cash_Flow=("Cash_Inflow_Adjusted", "mean")
        )
    )

    # Lower High_Risk_Rate = safer
    business_risk_summary = business_risk_summary.sort_values(
        "High_Risk_Rate", ascending=True
    )

    print("\nBusiness Type Ranking (Lowest Risk First):")
    print(business_risk_summary)

    # =====================================================
    # Step 7: OD Approval Score
    # =====================================================
    print("\n[Step 7] Calculating OD Approval Score...")

    business_risk_summary["OD_Approval_Score"] = (
        (1 - business_risk_summary["High_Risk_Rate"]) * 0.5 +
        (1 - business_risk_summary["Avg_OD_Utilization"]) * 0.3 +
        (business_risk_summary["Avg_Capital"]) * 0.2
    )

    business_risk_summary = business_risk_summary.sort_values(
        "OD_Approval_Score", ascending=False
    )

    print("\nBusiness Types Ranked by OD Approval Score (Best First):")
    print(business_risk_summary)

    # =====================================================
    # Save Outputs
    # =====================================================
    output_dir = r"C:\Users\steff\OneDrive\Desktop\Capstone\outputs"
    os.makedirs(output_dir, exist_ok=True)

    ranking_path = os.path.join(output_dir, "business_type_od_ranking.csv")
    business_risk_summary.to_csv(ranking_path)

    print(f"\nSaved Business OD Ranking to: {ranking_path}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 70)

    print("\nFinal Interpretation:")
    print("• Businesses at TOP → Best candidates for OD approval & lower interest.")
    print("• Businesses at BOTTOM → Higher risk, stricter lending policy required.")


if __name__ == "__main__":
    main()