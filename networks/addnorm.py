import torch
from torch import nn


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)

        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y)+X)