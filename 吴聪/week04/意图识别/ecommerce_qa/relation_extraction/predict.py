# -*- coding: utf-8 -*-
"""R-BERT 关系抽取预测器。"""
import json

import torch
from transformers import AutoTokenizer

from ecommerce_qa import config
from ecommerce_qa.relation_extraction.dataset import (
    REL_SPECIAL_TOKENS, build_marked_text,
)
from ecommerce_qa.relation_extraction.model import RBERT


class RelPredictor:
    """关系抽取预测器。

    用法：
        rel = RelPredictor.load()
        rel.predict("小米14 是 小米 生产的", head="小米14", tail="小米")
        # -> "生产品牌"
    """

    def __init__(self, model, tokenizer, relations: list, e1_id: int, e2_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.relations = relations
        self.e1_id = e1_id
        self.e2_id = e2_id

    @classmethod
    def load(cls, model_dir=None) -> "RelPredictor":
        """模型不存在时返回 None。"""
        model_dir = model_dir or (config.MODELS_DIR / "relation")
        model_path = model_dir / "model.pt"
        rel_path = model_dir / "relations.json"
        tok_dir = model_dir / "tokenizer"
        if not (model_path.exists() and rel_path.exists() and tok_dir.exists()):
            return None
        with open(rel_path, "r", encoding="utf-8") as f:
            relations = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        model = RBERT(num_relations=len(relations))
        # 词表比默认 BERT 多了 4 个实体标记，必须先 resize 再载权重
        model.bert.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return cls(
            model, tokenizer, relations,
            tokenizer.convert_tokens_to_ids("[E1]"),
            tokenizer.convert_tokens_to_ids("[E2]"),
        )

    def predict(self, text: str, head: str, tail: str) -> str:
        """返回句子中 head 与 tail 的关系名。"""
        marked = build_marked_text(text, head, tail)
        enc = self.tokenizer(marked, truncation=True, max_length=64)
        ids = enc["input_ids"]
        e1_pos = ids.index(self.e1_id)
        e2_pos = ids.index(self.e2_id)
        with torch.no_grad():
            logits = self.model(
                torch.tensor([ids]),
                torch.tensor([enc["attention_mask"]]),
                torch.tensor([e1_pos]),
                torch.tensor([e2_pos]),
            )
        idx = logits.argmax(dim=-1).item()
        return self.relations[idx]
