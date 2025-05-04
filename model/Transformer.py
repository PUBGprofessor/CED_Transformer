import torch.nn as nn
from model.encoder import Encoder, Encoder_
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        # 输入：[batch_size, src_seq_len, d_input]
        # 输出：[batch_size, src_seq_len, d_model]
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, dropout) 

        # 输入：[batch_size, trg_seq_len, d_model]
        # 输出：[batch_size, trg_seq_len, d_model]
        self.decoder = Decoder(trg_vocab, d_model, N, heads, d_ff, dropout)

        # 输入：[batch_size, trg_seq_len, d_model]
        # 输出：[batch_size, trg_seq_len, trg_vocab]
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask=None, look_ahead_mask=None, trg_mask=None):
    # def forward(self, src, src_mask=None, look_ahead_mask=None, trg_mask=None):
        e_outputs = self.encoder(src, src_mask)
        # e_outputs: [batch_size, src_seq_len, d_model]

        # trg: [batch_size, trg_seq_len]
        d_output = self.decoder(trg, e_outputs, look_ahead_mask, trg_mask)
        # output = self.out(e_outputs)  # [batch, seq_len, trg_vocab]
        output = self.out(d_output)
        return output
    
class TransformerCED(nn.Module):
    def __init__(self, src_vocab, trg_vocab=10, emb_dim=16, text_dim=4, other_feat_dim=8, d_model=128, N=3, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        # 输入：[batch_size, src_seq_len, d_input]
        # 输出：[batch_size, src_seq_len, d_model]
        self.encoder = Encoder_(src_vocab, d_model, N, heads, d_ff, dropout, emb_dim=16, text_dim=4, other_feat_dim=8) 

        # 输入：[batch_size, trg_seq_len, d_model]
        # 输出：[batch_size, trg_seq_len, d_model]
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, d_ff, dropout)

        # 输入：[batch_size, trg_seq_len, d_model]
        # 输出：[batch_size, trg_seq_len, trg_vocab]
        self.out = nn.Linear(d_model, trg_vocab)

    # def forward(self, src, trg, src_mask=None, look_ahead_mask=None, trg_mask=None):
    def forward(self, src, src_mask=None, look_ahead_mask=None, trg_mask=None):
        e_outputs = self.encoder(src, src_mask)
        # e_outputs: [batch_size, src_seq_len, d_model]

        # trg: [batch_size, trg_seq_len]
        # d_output = self.decoder(trg, e_outputs, look_ahead_mask, trg_mask)
        output = self.out(e_outputs)  # [batch, seq_len, trg_vocab]
        # output = self.out(d_output)
        return output