import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=768, use_bias=True, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 2, bias=use_bias)

    def forward(self, x):
        x = x.mean(dim=1)  # average over time (temporal dimension)
        x = self.dropout(x)
        return self.linear(x)
