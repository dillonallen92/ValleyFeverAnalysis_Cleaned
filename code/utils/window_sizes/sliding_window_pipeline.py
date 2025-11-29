import numpy as np 
import torch 
from sklearn.preprocessing import MinMaxScaler 
from dataclasses import dataclass 

@dataclass
class SlidingWindowBatch:
  X_train: torch.Tensor
  X_test: torch.Tensor
  y_train: torch.Tensor
  y_test: torch.Tensor
  scaler_y: MinMaxScaler

class SlidingWindowPipeline():
  def __init__(self, X, y, test_frac = 0.2):
    self.X = X
    self.y = y 
    self.test_frac = test_frac 
  
  def create_single_feature_sequences(self, feature_index, window_size) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_feature = self.X[:, feature_index].reshape(-1,1)
    
    X_seq, y_seq = [], [] 
    for ii in range(len(X_feature) - window_size):
      X_seq.append(X_feature[ii:ii+window_size])
      y_seq.append(self.y[ii + window_size])
      
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    split_idx = int(len(X_seq) * (1 - self.test_frac))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    return X_train, X_test, y_train, y_test
  
  def scale(self, X_train, X_test, y_train, y_test) -> SlidingWindowBatch:
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    num_samples, window_size, num_feats = X_train.shape
    X_train_flat = X_train.reshape(-1, num_feats)
    X_test_flat  = X_test.reshape(-1, num_feats)
    
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(num_samples, window_size, num_feats)
    X_test_scaled  = scaler_X.transform(X_test_flat).reshape(X_test.shape[0], window_size, num_feats)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return SlidingWindowBatch(
      X_train = torch.tensor(X_train_scaled, dtype=torch.float32),
      X_test  = torch.tensor(X_test_scaled, dtype=torch.float32),
      y_train = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1),
      y_test = torch.tensor(y_test_scaled, dtype = torch.float32).view(-1, 1), 
      scaler_y = scaler_y
    )