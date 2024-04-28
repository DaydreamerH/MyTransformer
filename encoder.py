import torch
import math
from torch import nn
from position_encoding import PositionalEncoding
from encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, norm_shape, 
                dropout, num_heads, num_hiddens, ffn_num_inputs, ffn_num_hiddens, 
                num_layers, use_bias=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.position_encoding = PositionalEncoding(dropout, num_hiddens)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        
        for i in range(num_layers):
            self.blks.add_module(f"block{i}", 
            EncoderBlock(query_size, key_size, value_size, num_hiddens, 
                         num_heads, dropout, norm_shape, ffn_num_inputs, ffn_num_hiddens, use_bias))
    
    def forward(self, X, valid_lens, *args):
        X = self.position_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None]*len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
