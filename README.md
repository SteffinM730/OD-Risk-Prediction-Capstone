# Risk Prediction Model - Capstone Project

A comprehensive machine learning pipeline for business financial stress prediction using clustering and neural networks.

## Project Structure

```
Capstone/
├── src/                              # Source code package
│   ├── __init__.py
│   ├── data/                         # Data loading module
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   ├── features.py              # Feature selection
│   │   └── scaling.py               # Feature scaling
│   └── models/                       # ML models
│       ├── __init__.py
│       ├── clustering.py             # K-Means clustering & PCA
│       └── ann.py                    # Artificial Neural Network
├── data/
│   └── raw/                          # Raw datasets
│       └── 1L_real_world_business_financial_stress_dataset.csv
├── outputs/                          # Generated outputs
│   ├── models/                       # Trained models
│   │   ├── risk_ann_model.pth
│   │   └── ann_loss_curve.png
│   └── visualizations/               # Generated visualizations
│       ├── silhouette_scores.png
│       ├── cluster_cloud_2d.png
│       └── cluster_cloud_3d.png
├── logs/                             # Log files
├── Main.py                           # Entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## Pipeline Overview

### 1. **Data Loading** (`src/data/loader.py`)
   - Load CSV dataset from `data/raw/`
   - Returns pandas DataFrame

### 2. **Feature Selection** (`src/preprocessing/features.py`)
   - Select 14 key business financial features
   - Drop unnecessary columns

### 3. **Feature Scaling** (`src/preprocessing/scaling.py`)
   - StandardScaler normalization
   - Outputs scaled matrix for clustering algorithms

### 4. **Clustering Analysis** (`src/models/clustering.py`)
   - K-Means clustering with silhouette analysis
   - Optimized sampling (10k rows) for fast silhouette computation
   - Full data clustering (100k rows)
   - 2D & 3D PCA visualizations

### 5. **Risk Prediction Model** (`src/models/ann.py`)
   - 4-layer Artificial Neural Network (PyTorch)
   - Binary classification: Low Risk vs High Risk
   - Sigmoid activation, BCE loss
   - 80-20 train-test split

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python Main.py
```

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
torch
```

## Usage

Run the complete pipeline:
```bash
python Main.py
```

The script will:
1. Load and preprocess data
2. Scale features
3. Perform K-Means clustering with 2D/3D PCA visualizations
4. Train and evaluate the ANN model
5. Save outputs to `outputs/` directory

## Output Files

### Visualizations (`outputs/visualizations/`)
- `silhouette_scores.png` - Silhouette scores for different k values
- `cluster_cloud_2d.png` - 2D PCA clustering visualization
- `cluster_cloud_3d.png` - 3D PCA clustering visualization

### Models (`outputs/models/`)
- `risk_ann_model.pth` - Trained neural network weights
- `ann_loss_curve.png` - Training loss curve

## Model Architecture

### ANN Model
- **Input Layer**: 12 features (numeric)
- **Hidden Layer 1**: 64 neurons + ReLU
- **Hidden Layer 2**: 32 neurons + ReLU
- **Hidden Layer 3**: 16 neurons + ReLU
- **Output Layer**: 1 neuron + Sigmoid (binary classification)
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 30

### Clustering
- **Algorithm**: K-Means
- **Silhouette-based K Selection**: 2-7 clusters tested
- **Dimensionality Reduction**: PCA (2D & 3D)

## Performance

Typical results on test data:
- **ANN Accuracy**: 100% (on test set)
- **Best K (Silhouette)**: 2 clusters
- **PCA Variance Explained (2D)**: 61.5%
- **PCA Variance Explained (3D)**: 70.3%

## Author

Capstone Project Team

## License

All rights reserved
