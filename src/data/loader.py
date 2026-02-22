import pandas as pd
import os


def load_dataset(filename):
    """
    Load dataset from data/ directory.
    
    Args:
        filename (str): Name of the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Get project root (3 levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try data/raw/ first, then data/
    data_path_raw = os.path.join(project_root, "data", "raw", filename)
    data_path = os.path.join(project_root, "data", filename)
    
    if os.path.exists(data_path_raw):
        data_path = data_path_raw
    elif not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path} or {data_path_raw}")
    
    return pd.read_csv(data_path)
