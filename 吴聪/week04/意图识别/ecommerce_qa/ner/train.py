# -*- coding: utf-8 -*-
"""NER 训练脚本：在 CPU 上小规模训练 BiLSTM+CRF。"""
import torch
from torch.utils.data import DataLoader

from ecommerce_qa import config
from ecommerce_qa.data_generation import annotator
from ecommerce_qa.ner.dataset import (
    NerDataset, build_label_map, build_vocab, collate, load_records, save_vocab,
)
from ecommerce_qa.ner.model import BiLSTMCRF
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)


def train_ner() -> None:
    """训练 NER 模型并把产物保存到 MODELS_DIR/ner/。"""
    ner_path = config.ANNOTATED_DIR / "ner.jsonl"
    if not ner_path.exists():
        logger.info("标注数据不存在，先生成数据与标注...")
        from ecommerce_qa.data_generation import generator
        generator.generate_all()
        annotator.build_ner_dataset(
            out_path=ner_path, max_sentences=config.NER["max_sentences"]
        )

    records = load_records(ner_path)
    vocab = build_vocab(records)
    label_map = build_label_map()
    logger.info("训练样本 %d 句，词表 %d，标签 %d 类",
                len(records), len(vocab), len(label_map))

    dataset = NerDataset(records, vocab, label_map)
    loader = DataLoader(dataset, batch_size=config.NER["batch_size"],
                        shuffle=True, collate_fn=collate)

    model = BiLSTMCRF(
        vocab_size=len(vocab),
        embed_dim=config.NER["embed_dim"],
        hidden_dim=config.NER["hidden_dim"],
        num_labels=len(label_map),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = config.NER["epochs"]
    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for tokens, tags, mask in loader:
            loss = model.loss(tokens, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        logger.info("epoch %d/%d  loss=%.4f", epoch + 1, epochs,
                    total_loss / max(n_batch, 1))

    # 保存产物
    out_dir = config.MODELS_DIR / "ner"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_vocab(vocab, out_dir / "vocab.json")
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        import json
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info("NER 模型已保存到 %s", out_dir)


if __name__ == "__main__":
    train_ner()
