# -*- coding: utf-8 -*-
"""检索模块测试：BM25 排序、持久化、图检索。"""
from ecommerce_qa.knowledge.memory_store import MemoryGraphStore
from ecommerce_qa.retrieval.bm25 import BM25Index
from ecommerce_qa.retrieval.graph_retrieval import GraphRetriever


def test_bm25_keyword_rank():
    """包含查询关键词的文档应排在最前面。"""
    docs = [
        "耐克 Air Max 270 跑步鞋",
        "华为 Mate 60 Pro 5G 手机",
        "小米 智能手表 运动监测",
    ]
    index = BM25Index(docs)
    results = index.search("耐克 跑步鞋")
    assert results, "应至少命中一篇文档"
    top_doc = results[0][0]
    assert docs[top_doc] == "耐克 Air Max 270 跑步鞋"


def test_bm25_no_match_returns_empty():
    """没有任何匹配时应返回空列表。"""
    index = BM25Index(["苹果 手机"])
    assert index.search("跑鞋") == []


def test_bm25_persist(tmp_path):
    """BM25 保存/加载往返后打分一致。"""
    docs = ["小米15 手机", "耐克 跑鞋"]
    index = BM25Index(docs)
    path = tmp_path / "bm25.json"
    index.save_json(path)
    loaded = BM25Index.load_json(path)
    assert loaded.docs == docs
    assert loaded.search("小米")[0][0] == 0


def test_graph_retriever():
    """图检索返回实体的邻居三元组。"""
    store = MemoryGraphStore()
    store.add_triples([
        ("Air Max 270", "生产品牌", "耐克"),
        ("Air Max 270", "价格为", "899元"),
        ("小米15", "生产品牌", "小米"),
    ])
    retriever = GraphRetriever(store)
    result = retriever.search(["Air Max 270"])
    assert len(result) == 2
    assert ("Air Max 270", "价格为", "899元") in result
    # 不存在的实体应被忽略
    assert retriever.search(["不存在的商品"]) == []
