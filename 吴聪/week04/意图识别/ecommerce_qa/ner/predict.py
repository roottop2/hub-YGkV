# -*- coding: utf-8 -*-
"""NER 预测器：加载训练好的模型，对句子做实体抽取。"""
import json

import torch

from ecommerce_qa import config
from ecommerce_qa.ner.model import BiLSTMCRF


class NerPredictor:
    """命名实体识别预测器。

    用法：
        ner = NerPredictor.load()
        entities = ner.predict("耐克 Air Max 270 多少钱")
        # -> [("耐克", "BRAND"), ("Air Max 270", "PRODUCT_MODEL")]
    """

    def __init__(self, model, vocab: dict, label_map: dict):
        self.model = model
        self.vocab = vocab
        self.label_map = label_map
        # 反向映射：id -> 标签名
        self.id2label = {v: k for k, v in label_map.items()}
        # 实体类型 -> (B-, I-) 前缀
        self.type_map = {e: (f"B-{e}", f"I-{e}") for e in config.ENTITIES}

    @classmethod
    def load(cls, model_dir=None) -> "NerPredictor":
        """从磁盘加载模型。模型不存在时返回 None（由调用方决定是否训练）。"""
        model_dir = model_dir or (config.MODELS_DIR / "ner")
        model_path = model_dir / "model.pt"
        vocab_path = model_dir / "vocab.json"
        label_path = model_dir / "label_map.json"
        if not (model_path.exists() and vocab_path.exists() and label_path.exists()):
            return None
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        model = BiLSTMCRF(
            vocab_size=len(vocab),
            embed_dim=config.NER["embed_dim"],
            hidden_dim=config.NER["hidden_dim"],
            num_labels=len(label_map),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return cls(model, vocab, label_map)

    def _predict_ids(self, text: str) -> list:
        """把句子变成字 id，预测标签 id 序列。"""
        chars = list(text)
        token_ids = [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in chars]
        tokens = torch.tensor([token_ids], dtype=torch.long)
        mask = torch.ones(1, len(chars), dtype=torch.long)
        with torch.no_grad():
            label_ids = self.model.decode(tokens, mask)[0]
        return label_ids.tolist()[: len(chars)]

    def predict_tags(self, text: str) -> list:
        """返回逐字符标签序列（供评测 F1 使用）。"""
        return [self.id2label[i] for i in self._predict_ids(text)]

    def predict(self, text: str) -> list:
        """返回实体列表 [(实体文本, 实体类型), ...]。"""
        labels = [self.id2label[i] for i in self._predict_ids(text)]
        entities = []
        cur_type = None
        start = 0
        for idx, lab in enumerate(labels + ["O"]):
            if lab.startswith("B-"):
                if cur_type is not None:
                    entities.append((text[start:idx], cur_type))
                cur_type = lab[2:]
                start = idx
            elif lab.startswith("I-"):
                if cur_type is None:   # 孤立 I，忽略
                    pass
                # 类型变化时按新实体开始（少见，保守处理）
            else:  # O 或标签结束
                if cur_type is not None:
                    entities.append((text[start:idx], cur_type))
                    cur_type = None
        # 去重保序
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result
