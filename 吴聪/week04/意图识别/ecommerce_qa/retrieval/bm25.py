# -*- coding: utf-8 -*-
"""自实现 BM25 检索算法（约 60 行，不依赖外部检索库）。

BM25 是经典的"词频 + 逆文档频率"打分算法，用于衡量文档与查询的相关性。
公式（对每个命中词求和）：
    score(d, q) = Σ  IDF(w) · tf·(k1+1) / ( tf + k1·(1-b+b·|d|/avgdl) )
    IDF(w)     = ln( (N - df_w + 0.5) / (df_w + 0.5) + 1 )
其中：
    tf    = 词 w 在文档 d 中出现的次数
    df_w  = 包含词 w 的文档数（df）
    N     = 文档总数
    |d|   = 文档 d 的词数，avgdl = 所有文档平均词数
    k1, b = 可调参数（k1 控制词频饱和，b 控制长度惩罚）
"""
import json
import math
from collections import Counter

from ecommerce_qa.utils.tokenizer import tokenize


class BM25Index:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        """构建倒排统计信息。

        Args:
            docs: 文档字符串列表
            k1: 词频饱和参数（默认 1.5）
            b: 长度归一化参数（默认 0.75）
        """
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(self.N, 1)
        # 统计每个词的文档频率 df
        df = Counter()
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] += 1
        self.df = df

    def _score_one(self, query_tokens: list, doc_tokens: list) -> float:
        """计算查询与单篇文档的 BM25 分数。"""
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for w in set(query_tokens):
            if w not in self.df or tf[w] == 0:
                continue
            idf = math.log((self.N - self.df[w] + 0.5) / (self.df[w] + 0.5) + 1)
            denom = tf[w] + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += idf * tf[w] * (self.k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> list:
        """返回 [(文档下标, 分数), ...]，按分数降序，分数为 0 的不返回。"""
        q_tokens = tokenize(query)
        scores = [self._score_one(q_tokens, t) for t in self.doc_tokens]
        ranked = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]

    # ---- 持久化：只需保存原始文档，加载时重建索引 ----
    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path, **kwargs) -> "BM25Index":
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return cls(docs, **kwargs)
