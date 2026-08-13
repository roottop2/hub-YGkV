# -*- coding: utf-8 -*-
"""自实现的条件随机场（CRF）解码层。

为什么不直接用现成库（如 torchcrf）：
  1. 环境里没有安装；
  2. 自己实现约 80 行，能彻底看懂"标签转移"这一核心思想，
     这正是 NER 里 BiLSTM + CRF 比单纯 BiLSTM 强的关键。

CRF 在 NER 中的作用：
  BiLSTM 已经给每个字预测了一个"打分"（发射分数 emission）。
  CRF 额外学习"标签之间的转移分数"（转移矩阵 trans），
  保证解码出的标签序列是"整体最优"的，比如：
    允许  B-BRAND -> I-BRAND     （品牌内部连续）
    惩罚  I-BRAND -> O            （品牌没开始就结束了）
  解码用 Viterbi 算法找全局最优路径。

形状约定：
  emissions (T, B, L)：T=序列长度，B=批量，L=标签数
  mask      (T, B)   ：1 表示有效位置，0 表示 padding
  tags      (T, B)   ：标签 id
"""
import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        # 转移分数：trans[i, j] = 从标签 i 转移到标签 j 的分数
        self.trans = nn.Parameter(torch.randn(num_labels, num_labels) * 0.1)
        # 起始分数 / 结束分数
        self.start_trans = nn.Parameter(torch.zeros(num_labels))
        self.end_trans = nn.Parameter(torch.zeros(num_labels))

    def _log_sum_exp(self, x: torch.Tensor) -> torch.Tensor:
        """数值稳定的 log-sum-exp，对最后一维归约。"""
        m = x.max(dim=-1, keepdim=True).values        # (..., 1)
        # m 去掉多余维度后再加，避免 (...,1) 与 (...) 的错误广播
        return m.squeeze(-1) + (x - m).exp().sum(dim=-1).log()

    # ------------------------------------------------------------------
    # 前向算法：计算 log Z（所有合法路径分数的 log-sum-exp）
    # ------------------------------------------------------------------
    def forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回每个序列的 log Z，形状 (B,)。"""
        T, B, L = emissions.shape
        # 初始分数 = 起始转移 + 第一位置发射
        score = self.start_trans.unsqueeze(0) + emissions[0]          # (B, L)
        for t in range(1, T):
            # 目标：score_cur[j] = logsumexp_i ( score_prev[i] + trans[i,j] ) + emit[j]
            # 把张量排成 (B, L, L) = [b, j, i]：
            #   score.unsqueeze(1)        -> (B, 1, L)  [当前标签 j 在 dim1]
            #   trans.t().unsqueeze(0)    -> (1, L, L)  [j, i] = trans[i,j]
            # 然后对最后一维 i 做 logsumexp，得到 (B, L)（按当前标签 j）
            new_score = self._log_sum_exp(
                score.unsqueeze(1) + self.trans.t().unsqueeze(0)
            )
            new_score = new_score + emissions[t]                      # (B, L)
            # padding 位置不更新，保持原值
            score = torch.where(mask[t].unsqueeze(1).bool(), new_score, score)
        # 结束：每个标签的累计分 + 结束转移，再 logsumexp
        score = score + self.end_trans.unsqueeze(0)                   # (B, L)
        return self._log_sum_exp(score)                               # (B,)

    # ------------------------------------------------------------------
    # 计算某个真实路径的分数（用于训练）
    # ------------------------------------------------------------------
    def score_path(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回每个序列的真实路径分数，形状 (B,)。"""
        T, B, L = emissions.shape
        batch_idx = torch.arange(B, device=emissions.device)
        # 起始转移 + 第一位置发射
        score = self.start_trans[tags[0]] + emissions[0, batch_idx, tags[0]]   # (B,)
        for t in range(1, T):
            step = (
                score
                + self.trans[tags[t - 1], tags[t]]
                + emissions[t, batch_idx, tags[t]]
            )
            score = torch.where(mask[t].bool(), step, score)
        # 结束转移：取每个序列最后一个有效位置的标签
        lengths = mask.sum(dim=0) - 1                                 # (B,)
        last_labels = tags[lengths, batch_idx]                        # (B,)
        return score + self.end_trans[last_labels]

    # ------------------------------------------------------------------
    # 负对数似然损失
    # ------------------------------------------------------------------
    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回批量平均的负对数似然（越小越好）。

        推导：log P(真实路径) = 路径分数 - log Z
        损失 = -log P(真实路径) = log Z - 路径分数
        所以是 (log_z - path_score)，符号不能写反，否则训练方向就错了。
        """
        log_z = self.forward_alg(emissions, mask)                     # (B,)
        path_score = self.score_path(emissions, tags, mask)           # (B,)
        return (log_z - path_score).mean()

    # ------------------------------------------------------------------
    # Viterbi 解码：找全局最优路径（推理用）
    # ------------------------------------------------------------------
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回最优标签序列，形状 (B, T)。"""
        T, B, L = emissions.shape
        viterbi = self.start_trans.unsqueeze(0) + emissions[0]        # (B, L)
        backpointers = []
        for t in range(1, T):
            # 对每个"当前标签 j"，找使 (上一步分 + 转移分) 最大的"上一步标签 i"
            prev = viterbi.unsqueeze(2) + self.trans.unsqueeze(0)     # (B, L, L) [i, j]
            best_values, best_indices = prev.max(dim=1)               # (B, L)
            new_viterbi = best_values + emissions[t]
            valid = mask[t].unsqueeze(1).bool()                       # (B, 1)
            viterbi = torch.where(valid, new_viterbi, viterbi)
            # padding 位置的回溯指针清零，避免干扰
            best_indices = torch.where(valid, best_indices, torch.zeros_like(best_indices))
            backpointers.append(best_indices)                         # (B, L)
        # 最后一个有效位置取最大分数的标签
        last_label = viterbi.argmax(dim=1)                            # (B,)
        paths = torch.zeros(B, T, dtype=torch.long, device=emissions.device)
        paths[:, T - 1] = last_label
        cur = last_label
        for t in range(T - 1, 0, -1):
            # 回溯：t-1 位置的最优标签
            cur = backpointers[t - 1][torch.arange(B, device=emissions.device), cur]
            paths[:, t - 1] = cur
        return paths
