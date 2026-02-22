import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numeric_features(df):
    """
    Scale only numeric columns using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        df_scaled (pd.DataFrame) → full dataframe with scaled numeric values
        X_scaled (ndarray) → scaled numeric matrix (for PCA/KMeans)
        scaler (StandardScaler) → fitted scaler object for inverse transformation
    """

    df_scaled = df.copy()

    # Select numeric columns
    numeric_cols = df_scaled.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # Numeric matrix for clustering
    X_scaled = df_scaled[numeric_cols].values

    return df_scaled, X_scaled, scaler
