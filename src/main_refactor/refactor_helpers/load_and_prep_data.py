import pandas as pd 
from pathlib import Path 
import numpy as np 

def load_and_prep_data(data_path: Path, tgt_variable: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df: pd.DataFrame = pd.read_csv(data_path)
    for col in ["YEAR_MONTH", "Year-Month", "DATE"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Isolate the features
    feature_columns: list[str] = [col for col in df.columns if col != tgt_variable]
    print(f"Features: {feature_columns}")
    
    # Create the feature and target vectors
    X: np.ndarray = df[feature_columns].values
    y: np.ndarray = df[tgt_variable].values 
    
    return X, y, feature_columns