# -*- coding: utf-8 -*-
"""R-BERT 关系抽取模型。

核心思想：把句子的头实体用 [E1][/E1] 包裹、尾实体用 [E2][/E2] 包裹，
BERT 编码后取 [CLS]、[E1] 位置、[E2] 位置的向量拼接，再分类。

快速模式下冻结 BERT 大部分参数（只训练 embedding 层中的新标记向量
与分类头），大幅降低 CPU 训练时间。
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from ecommerce_qa import config


class RBERT(nn.Module):
    def __init__(self, num_relations: int, model_name: str = None,
                 freeze_bert: bool = True):
        super().__init__()
        model_name = model_name or config.BERT_MODEL_NAME
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size   # 768
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_relations),
        )
        self.dropout = nn.Dropout(0.3)

        if freeze_bert:
            # 只训练 embedding 层（让新增的 [E1][E2] 向量可学），其余冻结
            for name, p in self.bert.named_parameters():
                p.requires_grad = "embeddings" in name

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        """返回每个样本的关系类别 logits (B, num_relations)。

        Args:
            input_ids: (B, T)
            attention_mask: (B, T)
            e1_pos: (B,) [E1] 的位置
            e2_pos: (B,) [E2] 的位置
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state                    # (B, T, H)
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        head_vec = hidden[batch, e1_pos, :]                   # (B, H) [E1] 位置
        tail_vec = hidden[batch, e2_pos, :]                   # (B, H) [E2] 位置
        cls_vec = hidden[:, 0, :]                             # (B, H) [CLS] 位置
        x = torch.cat([cls_vec, head_vec, tail_vec], dim=-1)  # (B, 3H)
        x = self.dropout(x)
        return self.fc(x)
