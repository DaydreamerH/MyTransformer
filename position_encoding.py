import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, num_hiddens, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        self.X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)/torch.pow(1000, torch.arange(0, max_len, 2, dtype=torch.float32)/num_hiddens)
        self.P[:, :, 0::2] = torch.sin(self.X)
        self.P[:, :, 1::2] = torch.cos(self.X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)