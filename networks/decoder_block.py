import torch
from torch import nn
from attention import MultiHeadAttention
from addnorm import AddNorm
from ffn import PositionWiseFFN


class DecoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, 
                num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens, i, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.i = i
        self.attention1 = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.concat((state[2][self.i], X), axis = 1)
        state[2][self.i] = key_values

        if self.istraining:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(enc_outputs, enc_outputs, Y, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
