# -*- coding: utf-8 -*-
"""R-BERT 数据集：在句子中给头/尾实体插入标记 [E1][/E1][E2][/E2]。"""
import json

import torch
from torch.utils.data import Dataset

from ecommerce_qa import config

# 实体标记（R-BERT 的核心：让 BERT 知道"哪里"是头实体、哪里是尾实体）
REL_SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def locate(text: str, head: str, tail: str) -> tuple:
    """定位头尾实体在句子中的 (起, 止) 位置，处理 tail 藏在 head 内的情况。

    例如 text="小米14 是 小米 生产的"：head="小米14"，tail="小米"，
    必须找到第 2 个"小米"（下标 5），而不是 head 内部的"小米"。
    """
    h_start = text.find(head)
    if h_start == -1:
        raise ValueError(f"头实体 {head!r} 不在句子中: {text!r}")
    # 优先从 head 结束位置之后找 tail，避免 tail 是 head 的子串
    t_start = text.find(tail, h_start + len(head))
    if t_start == -1:
        t_start = text.find(tail)
    if t_start == -1:
        raise ValueError(f"尾实体 {tail!r} 不在句子中: {text!r}")
    return h_start, h_start + len(head), t_start, t_start + len(tail)


def build_marked_text(text: str, head: str, tail: str) -> str:
    """按头尾实体出现顺序插入标记，返回带标记的句子。"""
    hs, he, ts, te = locate(text, head, tail)
    if he <= ts:   # head 在 tail 前面
        return (
            text[:hs] + "[E1]" + head + "[/E1]"
            + text[he:ts] + "[E2]" + tail + "[/E2]" + text[te:]
        )
    else:          # tail 在 head 前面
        return (
            text[:ts] + "[E2]" + tail + "[/E2]"
            + text[te:hs] + "[E1]" + head + "[/E1]" + text[he:]
        )


class RelDataset(Dataset):
    """关系抽取数据集。每个样本：(句子, 头实体, 尾实体, 关系类别)。"""

    def __init__(self, samples: list, tokenizer, e1_id: int, e2_id: int,
                 relation2idx: dict):
        self.samples = samples
        self.tokenizer = tokenizer
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.relation2idx = relation2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        marked = build_marked_text(s["text"], s["head"], s["tail"])
        enc = self.tokenizer(marked, truncation=True, max_length=64)
        ids = enc["input_ids"]
        e1_pos = ids.index(self.e1_id)   # [E1] 的位置
        e2_pos = ids.index(self.e2_id)   # [E2] 的位置
        label = self.relation2idx[s["relation"]]
        return ids, e1_pos, e2_pos, label


def collate(batch, pad_id: int) -> tuple:
    """把一批样本填充成等长张量。"""
    ids, e1_pos, e2_pos, labels = zip(*batch)
    max_len = max(len(i) for i in ids)
    input_ids, masks = [], []
    for i in ids:
        pad_len = max_len - len(i)
        input_ids.append(i + [pad_id] * pad_len)
        masks.append([1] * len(i) + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        torch.tensor(e1_pos, dtype=torch.long),
        torch.tensor(e2_pos, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def load_samples(path: str) -> list:
    """从 jsonl 读取关系抽取样本。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_relation2idx() -> dict:
    """关系名 -> id（顺序与 config.RELATIONS 一致，固定可复现）。"""
    return {r: i for i, r in enumerate(config.RELATIONS)}
