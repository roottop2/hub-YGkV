# -*- coding: utf-8 -*-
"""中文分词工具（基于 jieba），供 BM25 检索等使用。"""
import jieba


def tokenize(text: str) -> list:
    """对文本做 jieba 精确分词，返回词列表。

    例：tokenize("这款手机支持5G吗") -> ['这款', '手机', '支持', '5', 'G', '吗']
    """
    if not text:
        return []
    return list(jieba.cut(text.strip()))
