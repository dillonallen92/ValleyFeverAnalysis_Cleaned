import torch 
import torch.nn as nn 

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout):
    super().__init__()
    self.lstm = nn.LSTM(
      input_size = input_size,
      hidden_size = hidden_size, 
      num_layers = num_layers,
      batch_first = True,
      dropout = dropout if num_layers > 1 else 0.0
    )
    self.fc = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :]).squeeze(-1)