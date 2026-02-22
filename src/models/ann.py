import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    f1_score
)


def train_ann_for_risk(df_scaled):
    """
    Train ANN for business risk prediction.

    Returns:
        model
        df_result (full dataset with predictions attached)
    """

    # =====================================================
    #  CREATE TARGET VARIABLE
    # =====================================================
    df = df_scaled.copy()

    df["Risk_Label"] = (
        (df["Credit_Score"] < df["Credit_Score"].median()) &
        (df["Debt_to_Revenue_Ratio"] > df["Debt_to_Revenue_Ratio"].median()) |
        (df["Cash_Conversion_Cycle"] > df["Cash_Conversion_Cycle"].median())
    ).astype(int)

    print("\nClass Distribution:")
    print(df["Risk_Label"].value_counts())

    # =====================================================
    #  PREPARE DATA
    # =====================================================
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["Risk_Label"])
    y = df["Risk_Label"]

    feature_columns = X.columns.tolist()
    input_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    # =====================================================
    #  DEFINE ANN MODEL
    # =====================================================
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

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =====================================================
    #  TRAINING
    # =====================================================
    epochs = 50
    losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            preds = (outputs > 0.5).float()
            acc = (preds == y_train_tensor).float().mean().item()
            train_accuracies.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    # =====================================================
    #  EVALUATION
    # =====================================================
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor).numpy().flatten()
        y_pred_class = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
    auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred_class)

    print("\nAccuracy:", accuracy)
    print("RMSE:", rmse)
    print("AUC-ROC:", auc)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))

    # =====================================================
    #  SAVE OUTPUTS
    # =====================================================
    output_dir = r"C:\Users\steff\OneDrive\Desktop\Capstone\outputs"
    os.makedirs(output_dir, exist_ok=True)

    #  SAVE CHECKPOINT (Dashboard Compatible)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_columns": feature_columns
    }, os.path.join(output_dir, "risk_ann_model.pth"))

    # Save metrics report
    with open(os.path.join(output_dir, "ann_metrics_report.txt"), "w") as f:
        f.write("ANN RISK MODEL REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"AUC-ROC: {auc}\n")
        f.write(f"F1 Score: {f1}\n\n")
        f.write(classification_report(y_test, y_pred_class))

    # =====================================================
    #  PLOTS
    # =====================================================

    # Training Loss + Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.legend()
    plt.title("Training Loss & Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ann_roc_curve.png"))
    plt.close()

    print("\nSaved all ANN outputs to:", output_dir)

    # =====================================================
    # FULL DATASET PREDICTIONS
    # =====================================================
    X_full_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        full_pred_prob = model(X_full_tensor).numpy().flatten()
        full_pred_class = (full_pred_prob > 0.5).astype(int)

    df_result = df.copy()
    df_result["Predicted_Risk_Prob"] = full_pred_prob
    df_result["Predicted_Risk_Class"] = full_pred_class

    return model, df_result