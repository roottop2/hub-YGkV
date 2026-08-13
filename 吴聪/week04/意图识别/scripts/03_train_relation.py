# -*- coding: utf-8 -*-
"""脚本 03：训练关系抽取（R-BERT）。

运行：python scripts/03_train_relation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa.relation_extraction.train import train_relation  # noqa: E402


def main():
    print("== 03 训练关系抽取 ==")
    train_relation()
    print("完成：R-BERT 模型已保存到 models/relation/")


if __name__ == "__main__":
    main()
