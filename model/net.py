import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, vocab_size=50, emb_dim=16, text_dim=4, other_feat_dim=8, out_d=10):
        """
        vocab_size: text编码中使用的字典大小
        emb_dim:  每个字符嵌入维度
        text_dim: 字符数
        other_feat_dim: 其他特征的维度（例如 font size, line width 等数量）
        out_d: 输出维度
        """
        super(Net, self).__init__()
        self.text_dim = text_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(other_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64 + self.emb_dim * self.text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_d),
        )

    def forward(self, sentenses):
        """
        无需batch, seq_len就是batch_size
        sentenses: [seq_len, d=(text_dim + 8)]
        return: [seq_len, out_d]
        """
        seq_len = sentenses.shape[0]
        text = sentenses[:, :4].long() # [seq, 4]
        # print(text)
        other_features = sentenses[:, 4:] # [seq, 8]

        emb = self.embedding(text)  # [seq_len, 4, emb_dim]
        emb = emb.reshape(seq_len, self.text_dim * self.emb_dim) ## [seq_len, 4 * emb_dim]

        out = self.fc2(torch.cat([self.fc1(other_features), emb], dim=1))

        return out
