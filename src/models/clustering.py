import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Get project root from Main.py location
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def run_clustering_visuals(
    df,
    X_scaled,
    k_range=range(2, 8),
    out_cloud_2d="cluster_cloud_2d.png",
    out_cloud_3d="cluster_cloud_3d.png",
    sample_size=10000,
):
    """
    Perform silhouette analysis, fit optimal KMeans,
    generate PCA 2D & 3D visualizations,
    and interpret financial strength clusters.

    Returns:
        df_clustered (DataFrame with ClusterId column)
    """

    # =====================================================
    # Create output directory
    # =====================================================
    outputs_dir = os.path.join(_project_root, "outputs", "visualizations")
    os.makedirs(outputs_dir, exist_ok=True)

    out_cloud_2d = os.path.join(outputs_dir, out_cloud_2d)
    out_cloud_3d = os.path.join(outputs_dir, out_cloud_3d)

    # =====================================================
    # Reproducible Sampling for Silhouette
    # =====================================================
    n_samples = X_scaled.shape[0]

    if n_samples > sample_size:
        print(
            f"Sampling {sample_size} rows from {n_samples} "
            f"for faster silhouette analysis..."
        )
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(
            n_samples, size=sample_size, replace=False
        )
        X_sample = X_scaled[sample_indices]
    else:
        print(f"Using all {n_samples} rows for silhouette analysis.")
        X_sample = X_scaled

    # =====================================================
    # Silhouette Score Analysis
    # =====================================================
    sil_scores = []
    ks = list(k_range)

    print(f"Computing silhouette scores for k from {ks[0]} to {ks[-1]}...")

    for i, k in enumerate(ks):
        print(f"  k={k} ({i+1}/{len(ks)})")

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        sil_scores.append(score)

        print(f"    Silhouette Score: {score:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sil_scores, marker="o")
    plt.xlabel("k (#clusters)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores by k")
    plt.savefig(
        os.path.join(outputs_dir, "silhouette_scores.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    best_k = ks[int(np.argmax(sil_scores))]
    print("\nBest k by silhouette:", best_k)

    # =====================================================
    # Final KMeans on Full Dataset
    # =====================================================
    print(f"\nFitting K-Means with best_k={best_k}...")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["ClusterId"] = clusters

    print("\nCluster counts:")
    print(df_clustered["ClusterId"].value_counts().sort_index())

    print("\nCluster profile (mean values):")
    cluster_profile = (
        df_clustered
        .groupby("ClusterId")
        .mean(numeric_only=True)
        .round(3)
    )
    print(cluster_profile)

    # =====================================================
    # Financial Strength Interpretation
    # =====================================================
    print("\nFinancial Strength Analysis Between Clusters:")

    # Custom strength score logic
    strength_score = (
        cluster_profile["Revenue_per_Day"] +
        cluster_profile["Capital"] -
        cluster_profile["OD_Utilization"]
    )

    strongest_cluster = strength_score.idxmax()
    weakest_cluster = strength_score.idxmin()

    print(f"\nCluster {strongest_cluster} → Financially STRONG")
    print(f"Cluster {weakest_cluster} → Financially STRESSED")

    comparison_df = cluster_profile.loc[
        [strongest_cluster, weakest_cluster],
        [
            "Revenue_per_Day",
            "Capital",
            "OD_Utilization",
            "Cash_Inflow_Adjusted",
            "Cash_Outflow_Adjusted",
        ]
    ]

    print("\nKey Financial Differences:")
    print(comparison_df)

    print("\nInterpretation:")
    print("• Strong cluster has higher revenue & capital")
    print("• Strong cluster has lower OD utilization")
    print("• Weak cluster relies more on OD & has lower capital")

    # =====================================================
    # Business Type Distribution
    # =====================================================
    if "Business_Type" in df.columns:
        print("\nBusiness Type Distribution by Cluster:")
        cluster_business_summary = (
            df_clustered
            .groupby(["ClusterId", "Business_Type"])
            .size()
            .unstack(fill_value=0)
        )
        print(cluster_business_summary)

    # =====================================================
    # PCA 2D Visualization (Financial Strength Highlighted)
    # =====================================================
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_scaled)

    centers_scaled = kmeans.cluster_centers_
    centers_2d = pca2.transform(centers_scaled)

    total_var_2d = pca2.explained_variance_ratio_.sum()

    colors = [
        "green" if c == strongest_cluster else "red"
        for c in clusters
    ]

    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[:, 0], X_2d[:, 1],
                c=colors, s=18, alpha=0.7)

    plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                marker="X",
                s=260,
                c=["green" if i == strongest_cluster else "red"
                   for i in range(len(centers_2d))])

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title(
        f"Financial Clusters (PCA 2D) | k={best_k} | "
        f"Var={total_var_2d:.2f}\n"
        f"Green = Strong | Red = Stressed"
    )

    plt.savefig(out_cloud_2d, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nPCA 2D explained variance ratio:",
          pca2.explained_variance_ratio_)
    print("Total variance explained (2 comps):",
          total_var_2d)

    # =====================================================
    # PCA 3D Visualization (Financial Strength Highlighted)
    # =====================================================
    from mpl_toolkits.mplot3d import Axes3D

    pca3 = PCA(n_components=3, random_state=42)
    X_3d = pca3.fit_transform(X_scaled)
    centers_3d = pca3.transform(centers_scaled)

    total_var_3d = pca3.explained_variance_ratio_.sum()

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors_3d = [
        "green" if c == strongest_cluster else "red"
        for c in clusters
    ]

    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
               c=colors_3d, s=12, alpha=0.6)

    ax.scatter(centers_3d[:, 0], centers_3d[:, 1],
               centers_3d[:, 2],
               marker="X",
               s=300,
               c=["green" if i == strongest_cluster else "red"
                  for i in range(len(centers_3d))])

    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_zlabel("PCA-3")

    ax.set_title(
        f"Financial Clusters (PCA 3D) | k={best_k} | "
        f"Var={total_var_3d:.2f}\n"
        f"Green = Strong | Red = Stressed"
    )

    plt.savefig(out_cloud_3d, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nPCA 3D explained variance ratio:",
          pca3.explained_variance_ratio_)
    print("Total variance explained (3 comps):",
          total_var_3d)

    return df_clustered