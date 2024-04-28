import torch
import math
from torch import nn
from decoder_block import DecoderBlock
from position_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, 
                num_hiddens, num_heads, ffn_num_inputs, ffn_num_hiddens, 
                dropout, num_layers, norm_shape, use_bias=False, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.position_encoding = PositionalEncoding(dropout, num_hiddens)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.blks = nn.Sequential()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        for i in range(num_layers):
            self.blks.add_module(f"blk:{i}", 
            DecoderBlock(query_size, key_size, value_size, num_hiddens, 
                         num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens, i, use_bias))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state):
        X = self.position_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks)for _ in range(2)]
        for i, blk in range(self.blks):
            X, state = self.blk(X, state)
            self._attention_weights[0][i] = self.blks.attention1.attention_weights
            self._attention_weights[1][i] = self.blks.attention2.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self.attention_weights

    def init_state(self, enc_outputs, enc_valid_lens):
        return (enc_outputs, enc_valid_lens, [None]*self.num_layers)
            