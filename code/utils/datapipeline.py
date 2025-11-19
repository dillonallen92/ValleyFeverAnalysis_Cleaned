from preprocessing import (
  read_data, create_feature_target_vecs, 
  create_sequences, train_test_split, scale_data
)
import torch 
from dataclasses import dataclass 
from typing import Any
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

@dataclass 
class TensorDataset:
  X_train: Any 
  X_test:  Any
  y_train: Any 
  y_test:  Any 
  scaler_X: MinMaxScaler
  scaler_y: MinMaxScaler

class DataPipeline():
  def __init__(self, file_path: Path):
    self.file_path = file_path 
  
  