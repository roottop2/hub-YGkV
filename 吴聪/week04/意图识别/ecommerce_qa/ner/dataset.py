# -*- coding: utf-8 -*-
"""NER 数据集与词表/标签映射。

采用"字级"处理：把句子拆成单个汉字，避免中文分词的错误传播。
"""
import json

import torch
from torch.utils.data import Dataset

from ecommerce_qa import config


def build_vocab(records: list) -> dict:
    """根据标注记录构造 字符 -> id 词表。"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for r in records:
        for ch in r["tokens"]:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def build_label_map() -> dict:
    """构造 标签名 -> id。O=0，然后是各实体的 B-/I-。"""
    labels = ["O"]
    for e in config.ENTITIES:
        labels.append(f"B-{e}")
        labels.append(f"I-{e}")
    return {lab: i for i, lab in enumerate(labels)}


class NerDataset(Dataset):
    """字级 NER 数据集。每个样本是 (字符列表, 标签列表)。"""

    def __init__(self, records: list, vocab: dict, label_map: dict):
        self.records = records
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        token_ids = [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in r["tokens"]]
        tag_ids = [self.label_map[t] for t in r["tags"]]
        return token_ids, tag_ids


def collate(batch) -> tuple:
    """把一批 (字id列表, 标签id列表) 填充到等长，返回 (tokens, tags, mask)。"""
    token_ids, tag_ids = zip(*batch)
    max_len = max(len(t) for t in token_ids)
    tokens, tags, masks = [], [], []
    for tok, tag in zip(token_ids, tag_ids):
        assert len(tok) == len(tag), "token 与 tag 长度不一致，检查标注数据"
        length = len(tok)
        tokens.append(tok + [0] * (max_len - length))          # <PAD>=0
        tags.append(tag + [0] * (max_len - length))            # O=0
        masks.append([1] * length + [0] * (max_len - length))
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(tags, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def load_records(path) -> list:
    """从 jsonl 读取 NER 标注记录。"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_vocab(vocab: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
