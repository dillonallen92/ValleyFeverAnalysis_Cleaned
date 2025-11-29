from code.utils.preprocessing.basic import (
  read_data, create_feature_target_vecs, 
  create_sequences, train_test_split, scale_data
)
from code.utils.window_sizes.masking import(
  generate_padded_data, create_masking_vector
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
  mask : Any
  scaler_X: MinMaxScaler
  scaler_y: MinMaxScaler

class DataPipeline():
  def __init__(self, data_file_path: Path, config_data: dict[str, Any], window_sizes: dict[str, int]):
    self.data_file_path = data_file_path 
    self.config_data = config_data
    self.window_sizes = window_sizes
    self.df = None 
    self.X = None 
    self.y = None 
    self.mask = None 
  
  def load_data(self):
    df = read_data(self.data_file_path)
    
    for col in ["YEAR_MONTH", "Year-Month", "DATE"]:
      if col in df.columns:
        df = df.drop(columns = [col])
    
    self.df = df
    return df 
  
  def build_windows(self):
    features = [col for col in self.df.columns if col != "VFRate"]
    X = self.df[features]
    y = self.df["VFRate"].to_numpy()
    
    padded, y_adj = generate_padded_data(X, self.window_sizes, y)
    mask = create_masking_vector(X, self.window_sizes)
    
    self.X = padded 
    self.y = y_adj
    self.mask = mask 
  
  def split(self):
    test_frac = 1 - float(self.config_data["train_frac"])
    
    num_samples = self.X.shape[-1]
    split_idx = int(num_samples * (1 - test_frac))
    
    X_train = self.X[:, :, :split_idx]
    X_test  = self.X[:, :, split_idx:]
    y_train = self.y[:split_idx]
    y_test  = self.y[split_idx:]
    
    mask = self.mask
    
    return X_train, X_test, y_train, y_test, mask 
  
  def scale(self, X_train, X_test, y_train, y_test, mask):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    max_win, feat, ntrain = X_train.shape 
    Xtrain_flat = X_train.transpose(2,0,1).reshape(-1, feat)
    Xtest_flat  = X_test.transpose(2,0,1).reshape(-1, feat)
    
    Xtrain_scaled = scaler_X.fit_transform(Xtrain_flat).reshape(ntrain, max_win, feat).transpose(1,2,0)
    ntest = X_test.shape[-1]
    Xtest_scaled = scaler_X.transform(Xtest_flat).reshape(ntest, max_win, feat).transpose(1,2,0)
    ytrain_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    ytest_scaled  = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    
    return TensorDataset(
      X_train = torch.tensor(Xtrain_scaled, dtype = torch.float32),
      X_test = torch.tensor(Xtest_scaled, dtype = torch.float32),
      y_train = torch.tensor(ytrain_scaled, dtype = torch.float32).view(-1,1),
      y_test = torch.tensor(ytest_scaled, dtype = torch.float32).view(-1,1),
      mask  = torch.tensor(mask, dtype=torch.float32),
      scaler_X = scaler_X,
      scaler_y= scaler_y
    )
  
  
  
  