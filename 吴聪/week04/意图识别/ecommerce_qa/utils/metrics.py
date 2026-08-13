# -*- coding: utf-8 -*-
"""评测指标：token 级精确率 / 召回率 / F1。"""
from typing import Iterable, Sequence


def sequence_f1(
    gold_tags: Sequence[Sequence[str]],
    pred_tags: Sequence[Sequence[str]],
    ignore: tuple = ("O",),
) -> dict:
    """按 token 计算序列标注的精确率/召回率/F1。

    Args:
        gold_tags: 标注序列列表，如 [["B-BRAND","I-BRAND","O"], ...]
        pred_tags: 预测序列列表（与 gold 一一对应）
        ignore:    忽略的标签（默认忽略 O，即非实体的位置不计入）

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    tp = fp = fn = 0
    for gold_seq, pred_seq in zip(gold_tags, pred_tags):
        for g, p in zip(gold_seq, pred_seq):
            if g in ignore and p in ignore:
                continue
            if g == p and g not in ignore:
                tp += 1          # 都识别为非忽略标签且相同：命中
            elif g not in ignore:
                fn += 1          # 标注是实体，但预测为其他：漏检
            else:
                fp += 1          # 标注非实体，但预测为实体：误检
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
