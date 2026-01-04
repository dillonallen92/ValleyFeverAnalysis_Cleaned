import torch.nn as nn 
import torch 

# File for custom loss functions
class RMSELoss(nn.Module):
  def __init__(self, eps=1e-8):
    super().__init__()
    self.mse = nn.MSELoss()
    self.eps = eps 
  
  def forward(self, y_pred, y_true):
    return torch.sqrt(self.mse(y_pred, y_true) + self.eps)