# -*- coding: utf-8 -*-
"""脚本 01：生成模拟数据 与 自动标注。

运行：python scripts/01_generate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa import config  # noqa: E402
from ecommerce_qa.data_generation import annotator, generator  # noqa: E402


def main():
    print("== 01 生成模拟数据 ==")
    generator.generate_all()
    annotator.build_ner_dataset(
        out_path=config.ANNOTATED_DIR / "ner.jsonl",
        max_sentences=config.NER["max_sentences"],
    )
    annotator.build_relation_dataset(
        out_path=config.ANNOTATED_DIR / "relation.jsonl",
        max_samples=config.RELATION["max_samples"],
    )
    annotator.build_intent_dataset(out_path=config.ANNOTATED_DIR / "intent.jsonl")
    print("完成：原始数据 + 三套标注已写入 data/ 目录。")


if __name__ == "__main__":
    main()
