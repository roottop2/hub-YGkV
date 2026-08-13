# -*- coding: utf-8 -*-
"""R-BERT 关系抽取训练。"""
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ecommerce_qa import config
from ecommerce_qa.data_generation import annotator, generator
from ecommerce_qa.relation_extraction.dataset import (
    REL_SPECIAL_TOKENS, RelDataset, build_relation2idx, collate, load_samples,
)
from ecommerce_qa.relation_extraction.model import RBERT
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)


def train_relation() -> None:
    """训练 R-BERT 并保存到 MODELS_DIR/relation/。"""
    rel_path = config.ANNOTATED_DIR / "relation.jsonl"
    if not rel_path.exists():
        logger.info("标注数据不存在，先生成数据与标注...")
        generator.generate_all()
        annotator.build_relation_dataset(
            out_path=rel_path, max_samples=config.RELATION["max_samples"]
        )

    samples = load_samples(rel_path)
    relation2idx = build_relation2idx()
    logger.info("关系样本 %d 条，关系类别 %d 种", len(samples), len(relation2idx))

    # 加载 BERT 分词器并注册实体标记
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": REL_SPECIAL_TOKENS})
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    dataset = RelDataset(samples, tokenizer, e1_id, e2_id, relation2idx)
    loader = DataLoader(
        dataset, batch_size=config.RELATION["batch_size"], shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )

    model = RBERT(
        num_relations=len(relation2idx),
        freeze_bert=config.RELATION["freeze_bert"],
    )
    model.bert.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = torch.nn.CrossEntropyLoss()

    epochs = config.RELATION["epochs"]
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for input_ids, masks, e1, e2, labels in loader:
            logits = model(input_ids, masks, e1, e2)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        logger.info("epoch %d/%d  loss=%.4f", epoch + 1, epochs,
                    total_loss / max(n, 1))

    # 保存产物：模型权重、关系顺序、分词器
    out_dir = config.MODELS_DIR / "relation"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "relations.json", "w", encoding="utf-8") as f:
        json.dump(config.RELATIONS, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    logger.info("R-BERT 模型已保存到 %s", out_dir)


if __name__ == "__main__":
    train_relation()
