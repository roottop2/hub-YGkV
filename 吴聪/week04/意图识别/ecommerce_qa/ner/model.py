# -*- coding: utf-8 -*-
"""BiLSTM + CRF 命名实体识别模型。

结构：字向量(Embedding) -> 双向LSTM -> 全连接(发射分数) -> CRF解码
与单纯 BiLSTM + Softmax 相比，CRF 额外建模了标签间的转移约束。
"""
import torch
import torch.nn as nn

from ecommerce_qa.ner.crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_labels: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)   # 双向 -> 2*hidden
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(num_labels)

    def emission(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """计算每个 token 的发射分数。

        Args:
            tokens: (B, T) 字 id
            lengths: (B,) 每个序列的有效长度
        Returns:
            (B, T, L) 发射分数
        """
        emb = self.dropout(self.embedding(tokens))                     # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)
        return self.fc(out)

    def loss(self, tokens: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """训练损失（负对数似然）。

        Args:
            tokens: (B, T)
            tags:   (B, T) 标签 id
            mask:   (B, T) 0/1
        """
        lengths = mask.sum(dim=1)                                      # (B,)
        emissions = self.emission(tokens, lengths)                     # (B, T, L)
        emissions = emissions.transpose(0, 1)                          # (T, B, L)
        return self.crf.nll(emissions, tags.transpose(0, 1), mask.transpose(0, 1))

    def decode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi 解码，返回 (B, T) 的标签 id。"""
        lengths = mask.sum(dim=1)
        emissions = self.emission(tokens, lengths)
        emissions = emissions.transpose(0, 1)                          # (T, B, L)
        return self.crf.decode(emissions, mask.transpose(0, 1))
