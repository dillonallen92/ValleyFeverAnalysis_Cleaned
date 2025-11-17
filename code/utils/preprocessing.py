import numpy as np 
import numpy.typing as npt
import pandas as pd 
from typing import TypeVar, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import torch 
from sklearn.preprocessing import MinMaxScaler

@dataclass 
class ScaledDataset:
  X_train: Any 
  X_test:  Any
  y_train: Any
  y_test:  Any
  scaler_X: MinMaxScaler
  scaler_y: MinMaxScaler
  


def read_data(data_path: Path) -> pd.DataFrame:
  '''
  This function reads the csv file and produces a dataframe. Because we do not
  care about the months or year (right now), this function drops that column so
  the dataframe only contains the important features and target vector
  
  INPUT:
    data_path : PATH - path object that contains the relative 
                       path to the datafile
  
  OUTPUT:
    df : pd.DataFrame - Pandas DataFrame object containing the important 
                        features and target vector columns
  '''
  df: pd.DataFrame = pd.read_csv(data_path)
  # if there is a year-month column, lets drop that
  data_cols : list[str] = [x.lower() for x in df.columns]
  if 'year-month' in data_cols:
    df.drop('Year-Month', axis='columns', inplace=True)
  return df

def create_feature_target_vecs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
  # target column name
  df_cpy = df.copy()
  tgt_col_name : str = "VFRate"
  y_tgt: np.ndarray = df[tgt_col_name].to_numpy()
  df_cpy = df_cpy.drop(tgt_col_name, axis='columns', inplace=True)
  X: np.ndarray = df_cpy.to_numpy()
  return X, y_tgt
  
def create_sequences(X: np.ndarray, y: np.ndarray, window_size:int) -> Tuple[np.ndarray, np.ndarray]:
  X_sequences, y_sequences = [], []
  for i in range(len(X) - window_size):
    X_sequences.append(X[i:i+window_size])
    y_sequences.append(y[i+window_size])
    
  X_seq: np.ndarray = np.array(X_sequences)
  y_seq: np.ndarray = np.array(y_sequences)
    
  return X_seq, y_seq 

def train_test_split(X_seq : np.ndarray, y_seq: np.ndarray, test_frac : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  split_index : int = int(len(X_seq) * (1 - test_frac))
  X_train, X_test = X_seq[:split_index], X_seq[split_index:]
  y_train, y_test = y_seq[:split_index], y_seq[split_index:]
  
  return X_train, X_test, y_train, y_test

def scale_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dataset:
  scaler_X: MinMaxScaler = MinMaxScaler()
  scaler_y : MinMaxScaler = MinMaxScaler()
  
  num_samples, window_size, num_features = X_train.shape
  test_samples = X_test.shape[0]
  
  X_train_reshaped = X_train.reshape(-1, num_features)
  X_test_reshaped  = X_test.reshape(-1, num_features)
  
  X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(num_samples, window_size, num_features)
  X_test_scaled  = scaler_X.transform(X_test_reshaped).reshape(test_samples, window_size, num_features)
  
  y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
  y_test_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
  
  scaled_dataset : ScaledDataset = ScaledDataset(
    X_train_scaled,
    X_test_scaled, 
    y_train_scaled, 
    y_test_scaled,
    scaler_X,
    scaler_y
  )
  
  return scaled_dataset
  


def main() -> None:
  fresno_path: Path = Path("data/Fresno_Aggregate.csv")
  agg_df : pd.DataFrame = read_data(fresno_path)
  print(agg_df)
  X, y = create_feature_target_vecs(agg_df)
  X_seq, y_seq = create_sequences(X, y, window_size=12)
  test_frac: float = 0.2
  X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_frac)
  scaledData = scale_data(X_train, X_test, y_train, y_test)
  
  

if __name__ == "__main__":
  main()