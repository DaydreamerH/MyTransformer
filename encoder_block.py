from ffn import PositionWiseFFN
from addnorm import AddNorm
from attention import MultiHeadAttention
import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                 norm_shape, ffn_num_inputs, ffn_num_hiddens, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, num_hiddens)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(self.attention(X, X, X, valid_lens))
        return self.addnorm2(self.ffn(Y))