# -*- coding: utf-8 -*-
"""脚本 05：训练意图识别（微调 BERT）。

运行：python scripts/05_train_intent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa.nlu.intent_model import IntentClassifier  # noqa: E402


def main():
    print("== 05 训练意图识别 ==")
    IntentClassifier.train()
    print("完成：意图模型已保存到 models/intent/")


if __name__ == "__main__":
    main()
