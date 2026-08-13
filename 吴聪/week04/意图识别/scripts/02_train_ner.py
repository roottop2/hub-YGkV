# -*- coding: utf-8 -*-
"""脚本 02：训练 NER（BiLSTM + 自实现 CRF）。

运行：python scripts/02_train_ner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa.ner.train import train_ner  # noqa: E402


def main():
    print("== 02 训练 NER ==")
    train_ner()
    print("完成：NER 模型已保存到 models/ner/")


if __name__ == "__main__":
    main()
