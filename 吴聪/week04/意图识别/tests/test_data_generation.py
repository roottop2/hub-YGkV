# -*- coding: utf-8 -*-
"""数据生成与自动标注测试。

注意：本测试全部使用内存数据，不写入真实 data/ 目录，
避免把演示用的模拟数据覆盖成小样本。
"""
from ecommerce_qa import config
from ecommerce_qa.data_generation import generator, annotator


def _sample_data():
    """构造一小批内存数据。"""
    products = generator.generate_products(6)
    orders = generator.generate_orders(4, products)
    policies = generator.generate_policies()
    return products, orders, policies


def test_products_fields():
    """商品数据字段齐全且价格/库存为数值。"""
    products, _, _ = _sample_data()
    for p in products:
        assert "id" in p and "brand" in p and "model" in p
        assert "category" in p and "price" in p and "stock" in p and "params" in p
        assert isinstance(p["price"], int) and isinstance(p["stock"], int)


def test_orders_reference_products():
    """订单引用的商品 id 应存在于商品列表中。"""
    products, orders, _ = _sample_data()
    ids = {p["id"] for p in products}
    for o in orders:
        assert o["product_id"] in ids
        assert o["order_id"] and o["status"]


def test_ner_tags_aligned():
    """NER 标注：标签长度=句子字符数，且存在非 O 标签。"""
    products, orders, policies = _sample_data()
    records = annotator.build_ner_dataset(
        products=products, orders=orders, policies=policies
    )
    assert len(records) > 0
    for r in records:
        assert len(r["tokens"]) == len(r["tags"])
        assert "".join(r["tokens"])  # 非空句子
        assert any(t != "O" for t in r["tags"]), f"句子没有实体标注: {r}"


def test_relation_head_tail_in_text():
    """关系标注：头/尾实体都应出现在句子中。"""
    products, orders, policies = _sample_data()
    records = annotator.build_relation_dataset(
        products=products, orders=orders, policies=policies
    )
    assert len(records) > 0
    for r in records:
        assert r["head"] in r["text"], f"头实体不在句中: {r}"
        assert r["tail"] in r["text"], f"尾实体不在句中: {r}"
        assert r["relation"] in config.RELATIONS


def test_intent_labels_valid():
    """意图标注：标签都在 INTENTS 集合内。"""
    products, _, _ = _sample_data()
    records = annotator.build_intent_dataset(products=products)
    assert len(records) > 0
    for r in records:
        assert r["intent"] in config.INTENTS, f"非法意图: {r}"
        assert r["text"]
