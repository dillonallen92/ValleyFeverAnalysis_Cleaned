import torch 
import torch.nn as nn 

class MaskedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # PyTorch applies dropout only when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:     (batch, seq_len, input_size)
        mask:  (batch, seq_len, input_size) entries of 0/1 indicating which inputs are valid.
        """
        masked_x = x * mask
        outputs, _ = self.lstm(masked_x)
        outputs = self.dropout(outputs)

        valid_steps = (mask.sum(dim=-1) > 0).float()
        pooled = (outputs * valid_steps.unsqueeze(-1)).sum(dim=1)
        denom = valid_steps.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = pooled / denom
        return self.fc(pooled)