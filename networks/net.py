from decoder import Decoder
from encoder import Encoder
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)
    

def transformer(src_vocab_size, tgt_vocab_size, query_size, key_size, norm_shape, num_layers, device, 
                value_size, num_hiddens, num_heads, ffn_num_inputs, ffn_num_hiddens, dropout, use_bias=False):
    encoder = Encoder(src_vocab_size, query_size, key_size, value_size, 
                      norm_shape, dropout, 
                      num_heads, num_hiddens, ffn_num_inputs, ffn_num_hiddens, num_layers, use_bias)
    decoder = Decoder(tgt_vocab_size, query_size, key_size, value_size, 
                      norm_shape, dropout, 
                      num_heads, num_hiddens, ffn_num_inputs, ffn_num_hiddens, num_layers, use_bias)
    return EncoderDecoder(encoder, decoder).to(device)