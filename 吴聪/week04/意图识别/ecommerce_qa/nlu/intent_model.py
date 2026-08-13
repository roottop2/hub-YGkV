# -*- coding: utf-8 -*-
"""意图识别：微调 BERT 分类器。

意图集合（见 config.INTENTS）：查商品参数 / 查订单状态 / 咨询售后 /
查退换货政策 / 闲聊。快速模式下冻结 BERT 主干只训分类头。
"""
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification

from ecommerce_qa import config
from ecommerce_qa.data_generation import annotator, generator
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)


class _IntentDataset(Dataset):
    def __init__(self, samples: list, tokenizer, intent2idx: dict):
        self.data = []
        for s in samples:
            enc = tokenizer(s["text"], truncation=True, max_length=32)
            self.data.append((
                enc["input_ids"], enc["attention_mask"],
                intent2idx[s["intent"]],
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _collate(batch, pad_id: int):
    ids, masks, labels = zip(*batch)
    max_len = max(len(i) for i in ids)
    input_ids, attn = [], []
    for i, m in zip(ids, masks):
        pad_len = max_len - len(i)
        input_ids.append(i + [pad_id] * pad_len)
        attn.append(m + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class IntentClassifier:
    """意图分类器。"""

    def __init__(self, model, tokenizer, intents: list):
        self.model = model
        self.tokenizer = tokenizer
        self.intents = intents

    @classmethod
    def train(cls) -> None:
        """训练并保存到 MODELS_DIR/intent/。"""
        path = config.ANNOTATED_DIR / "intent.jsonl"
        if not path.exists():
            logger.info("标注数据不存在，先生成数据与标注...")
            generator.generate_all()
            annotator.build_intent_dataset(out_path=path)

        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            samples = [_json.loads(line) for line in f if line.strip()]
        intent2idx = {i: idx for idx, i in enumerate(config.INTENTS)}
        logger.info("意图样本 %d 条，类别 %d 种", len(samples), len(intent2idx))

        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(
            config.BERT_MODEL_NAME, num_labels=len(config.INTENTS)
        )
        if config.INTENT["freeze_bert"]:
            for p in model.bert.parameters():
                p.requires_grad = False   # 只训分类头

        dataset = _IntentDataset(samples, tokenizer, intent2idx)
        pad_id = tokenizer.pad_token_id or 0
        loader = DataLoader(
            dataset, batch_size=config.INTENT["batch_size"], shuffle=True,
            collate_fn=lambda b: _collate(b, pad_id),
        )
        # BERT 主干用标准的微调学习率 3e-5，分类头用较大 1e-3
        # （若统一用 1e-3，BERT 权重会被大步长冲垮，输出塌缩成常数）
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": 3e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ])
        criterion = torch.nn.CrossEntropyLoss()

        epochs = config.INTENT["epochs"]
        for epoch in range(epochs):
            total_loss, n = 0.0, 0
            for input_ids, attn, labels in loader:
                logits = model(input_ids, attention_mask=attn).logits
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n += 1
            logger.info("epoch %d/%d  loss=%.4f", epoch + 1, epochs,
                        total_loss / max(n, 1))

        out_dir = config.MODELS_DIR / "intent"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir / "tokenizer")
        with open(out_dir / "intents.json", "w", encoding="utf-8") as f:
            _json.dump(config.INTENTS, f, ensure_ascii=False, indent=2)
        logger.info("意图模型已保存到 %s", out_dir)

    @classmethod
    def load(cls) -> "IntentClassifier":
        """模型不存在时返回 None。"""
        model_dir = config.MODELS_DIR / "intent"
        if not (model_dir / "config.json").exists():
            return None
        model = BertForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        with open(model_dir / "intents.json", "r", encoding="utf-8") as f:
            intents = json.load(f)
        return cls(model, tokenizer, intents)

    def predict(self, text: str) -> tuple:
        """返回 (意图名, 置信度)。"""
        enc = self.tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax(dim=-1))
        return self.intents[idx], float(probs[idx])
