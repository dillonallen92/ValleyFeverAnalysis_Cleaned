import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from refactor_helpers.load_and_prep_data import load_and_prep_data
from pathlib import Path  
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    # Load and prepare the data
    county_name = "Fresno"
    data_path: Path = Path(f"data/{county_name.lower()}_agg_drought_baseline.csv")
    tgt_variable: str = "VFRate"
    X, y, feature_columns = load_and_prep_data(data_path, tgt_variable)

    # Scale the features and target variable
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # train/test split
    split_index = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]  
    
    # Auto ARIMA to find the best parameters
    auto_model = auto_arima(y_train, exogenous=X_train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    print(auto_model.summary())

    