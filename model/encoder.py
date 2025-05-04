import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from model.layer import Norm, FeedForward
from model.attention import MultiHeadAttention

from utils.common import get_clones

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096):
        super().__init__()
        self.d_model = d_model # 嵌入维度

        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))

        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe) # 使得 pe 成为一个 buffer，避免在训练时更新
        
    def forward(self, x):
        # x : [batch_size, seq_len, d_model]
        # 使得单词嵌入表示相对大一些 why？
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x # [batch_size, seq_len, d_model]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x) # ADD & NORM 为什么是先norm再add？
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.N =N
        # self.embed = torch.nn.Embedding(vocab_size,d_model) # [b, inp_seq_len] => [b, inp_seq_len, d_model]
        self.embed = nn.Linear(vocab_size, d_model) # [b, inp_seq_len, d_input] => [b, inp_seq_len, d_model]
        self.pe =PositionalEncoder(d_model) # [b, inp_seq_len, d_model]
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm =Norm(d_model)

    def forward(self, src, mask):
        x= self.embed(src)
        x= self.pe(x)
        for i in range(self.N):
            x=self.layers[i](x, mask)
        return self.norm(x)

class Encoder_(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff=2048, dropout=0.1, emb_dim=16, text_dim=4, other_feat_dim=8):
        super().__init__()
        self.N =N
        self.text_dim = text_dim
        self.emb_dim = emb_dim
        
        # self.embed = torch.nn.Embedding(vocab_size,d_model) # [b, inp_seq_len] => [b, inp_seq_len, d_model]
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(other_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model - text_dim * emb_dim)
        )
        # self.embed = nn.Linear(vocab_size, d_model) # [b, inp_seq_len, d_input] => [b, inp_seq_len, d_model]
        self.pe =PositionalEncoder(d_model) # [b, inp_seq_len, d_model]
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm =Norm(d_model)

    def forward(self, src, mask):
        # src : [batch, seq_len, input = text_len + 8]
        seq_len = src.shape[1]
        text = src[:, :, :4].long() # [batch, seq, 4]
        # print(text)
        other_features = src[:, :, 4:] # [batch, seq, 8]

        emb = self.embedding(text)  # [batch, seq_len, 4, emb_dim]
        emb = emb.reshape(-1, seq_len, self.text_dim * self.emb_dim) # [batch, seq_len, text_dim * emb_dim]
        x = torch.cat([self.fc1(other_features), emb], dim=2) # [batch, seq_len, d_model]
        # print(x.shape)
        x = self.pe(x)
        for i in range(self.N):
            x=self.layers[i](x, mask)
        return self.norm(x)