# -*- coding: utf-8 -*-
"""知识融合与图谱存储测试。"""
import json

from ecommerce_qa.knowledge.fusion import merge_triples, unify_entity, unify_relation
from ecommerce_qa.knowledge.memory_store import MemoryGraphStore


def test_unify_entity_alias():
    """别名映射：nike -> 耐克。"""
    assert unify_entity("nike") == "耐克"
    assert unify_entity("Nike") == "耐克"     # 大小写不敏感
    assert unify_entity("耐克") == "耐克"     # 标准名保持


def test_unify_relation_synonym():
    """关系同义词映射。"""
    assert unify_relation("品牌") == "生产品牌"
    assert unify_relation("属于品牌") == "生产品牌"
    assert unify_relation("生产品牌") == "生产品牌"   # 标准名保持


def test_merge_triples_dedupe():
    """融合后应去重，且统一了别名实体。"""
    triples = [
        ("nike", "品牌", "Air Max 270"),
        ("耐克", "品牌", "Air Max 270"),
        ("耐克", "生产品牌", "Air Max 270"),
    ]
    merged = merge_triples(triples)
    # 三条都归一为同一三元组，去重后只剩一条
    assert len(merged) == 1
    assert merged[0] == ("耐克", "生产品牌", "Air Max 270")


def test_memory_store_queries():
    """内存图的基本增查。"""
    store = MemoryGraphStore()
    store.add_triples([
        ("Air Max 270", "生产品牌", "耐克"),
        ("Air Max 270", "价格为", "899元"),
        ("小米15", "生产品牌", "小米"),
    ])
    assert store.node_count == 5
    assert store.edge_count == 3

    assert store.get_entity("耐克") == "耐克"
    assert store.get_entity("不存在") is None

    # 按头实体查
    all_from_am = store.query_triples(subject="Air Max 270")
    assert len(all_from_am) == 2
    # 按关系查
    producers = store.query_triples(relation="生产品牌")
    assert len(producers) == 2
    # 按头+关系查
    r = store.query_triples(subject="小米15", relation="生产品牌")
    assert r == [("小米15", "生产品牌", "小米")]

    # 邻居查询
    neighbors = store.get_neighbors("Air Max 270")
    assert len(neighbors) == 2
    brand_neighbors = store.get_neighbors("Air Max 270", relation="生产品牌")
    assert brand_neighbors == [("耐克", "生产品牌")]


def test_memory_store_persist(tmp_path):
    """内存图导出/导入往返。"""
    store = MemoryGraphStore()
    store.add_triple("耐克", "生产品牌", "Air Max 270")
    path = tmp_path / "graph.json"
    store.save_json(path)
    loaded = MemoryGraphStore.load_json(path)
    assert loaded.query_triples() == [("耐克", "生产品牌", "Air Max 270")]
    assert loaded.node_count == 2
