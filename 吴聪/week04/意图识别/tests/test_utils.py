# -*- coding: utf-8 -*-
"""工具层测试：评测指标与分词。"""
from ecommerce_qa.utils.metrics import sequence_f1
from ecommerce_qa.utils.tokenizer import tokenize


def test_sequence_f1_perfect():
    """完全一致时 F1 应为 1.0。"""
    r = sequence_f1([["B-BRAND", "I-BRAND", "O"]], [["B-BRAND", "I-BRAND", "O"]])
    assert r["f1"] == 1.0


def test_sequence_f1_miss_all():
    """全部漏检时召回率应为 0。"""
    r = sequence_f1([["B-BRAND", "O"]], [["O", "O"]])
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_sequence_f1_ignores_o():
    """gold 与 pred 都是 O 的位置不算错误。"""
    r = sequence_f1([["O", "B-BRAND"]], [["O", "O"]])
    assert r["recall"] == 0.0  # B-BRAND 漏检 -> 召回 0
    assert r["precision"] == 0.0


def test_tokenize_nonempty():
    """分词结果应是非空列表。"""
    toks = tokenize("这款手机支持5G吗")
    assert isinstance(toks, list)
    assert len(toks) >= 1
