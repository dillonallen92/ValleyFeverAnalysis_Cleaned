from preprocessing import (
  read_data, create_feature_target_vecs, 
  create_sequences, train_test_split, scale_data
)
from config_file_parser import config_file_parser
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
  def __init__(self, data_file_path: Path, config_file_path: Path, sliding_window_path: Path):
    self.data_file_path = data_file_path 
    self.config_file_path = config_file_path
  
  