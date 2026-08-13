# -*- coding: utf-8 -*-
"""脚本 04：构建知识图谱 与 BM25 索引。

三元组来源：
  1. 结构化数据直接生成（权威、准确）；
  2. NER + R-BERT 从自然语句抽取（演示"文本 -> 三元组"的技术路径）；
  3. 经知识融合（实体统一/消歧/关系对齐）去重后写入图谱。

运行：python scripts/04_build_graph.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa import config  # noqa: E402
from ecommerce_qa.knowledge.fusion import merge_triples  # noqa: E402
from ecommerce_qa.knowledge.memory_store import MemoryGraphStore  # noqa: E402
from ecommerce_qa.ner.predict import NerPredictor  # noqa: E402
from ecommerce_qa.relation_extraction.predict import RelPredictor  # noqa: E402
from ecommerce_qa.retrieval.bm25 import BM25Index  # noqa: E402


def _load(name):
    path = config.RAW_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _structured_triples(products, orders, policies):
    """从结构化数据直接生成三元组（权威来源）。"""
    triples = []
    for p in products:
        model = p["model"]
        triples.append((model, "生产品牌", p["brand"]))
        triples.append((model, "属于类别", p["category"]))
        triples.append((model, "价格为", f'{p["price"]}元'))
        triples.append((model, "库存为", f'{p["stock"]}件'))
        for param in p["params"]:
            triples.append((model, "参数为", param))

    pid2model = {p["id"]: p["model"] for p in products}
    for o in orders:
        model = pid2model.get(o["product_id"])
        if not model:
            continue
        triples.append((o["order_id"], "订单包含", model))
        triples.append((o["order_id"], "订单状态为", o["status"]))

    # 政策作用于部分商品
    random.seed(config.SEED)
    for p in products[:30]:
        pol = random.choice(policies)
        triples.append((p["model"], "适用政策", pol["title"]))
    return triples


def _text_triples(products, ner, rel):
    """用 NER + R-BERT 从自然语句抽取三元组（演示技术路径）。"""
    extracted = []
    if ner is None or rel is None:
        print("  （未找到 NER/关系模型，跳过文本抽取）")
        return extracted
    for p in products[:30]:
        model, brand, category = p["model"], p["brand"], p["category"]
        price = f'{p["price"]}元'
        candidates = [
            (f"{model} 是 {brand} 生产的", model, brand),
            (f"{model} 属于 {category}", model, category),
            (f"{model} 的价格是 {price}", model, price),
        ]
        for text, head, tail in candidates:
            try:
                relation = rel.predict(text, head, tail)
                extracted.append((head, relation, tail))
            except Exception:
                continue
    return extracted


def main():
    print("== 04 构建知识图谱 ==")
    products = _load("products")
    orders = _load("orders")
    policies = _load("policies")

    # 1) 结构化三元组
    triples = _structured_triples(products, orders, policies)
    print(f"结构化三元组：{len(triples)} 条")

    # 2) 文本抽取三元组（演示 NER + R-BERT 链路）
    ner = NerPredictor.load()
    rel = RelPredictor.load()
    text_triples = _text_triples(products, ner, rel)
    print(f"文本抽取三元组：{len(text_triples)} 条")

    # 3) 知识融合 + 去重
    merged = merge_triples(triples + text_triples)
    print(f"融合后三元组：{len(merged)} 条")

    # 4) 建图 + 保存
    store = MemoryGraphStore(merged)
    store.save_json(config.GRAPH_DIR / "graph.json")
    print(f"知识图谱：{store.node_count} 节点 / {store.edge_count} 关系")

    # 5) 构建 BM25 检索文档
    docs = []
    for p in products:
        params = " ".join(p["params"])
        docs.append(f"{p['brand']} {p['model']} {p['category']} "
                    f"价格{p['price']}元 库存{p['stock']}件 参数{params}")
    for pol in policies:
        docs.append(f"{pol['title']} {pol['content']}")
    pid2model = {p["id"]: p["model"] for p in products}
    for o in orders:
        model = pid2model.get(o["product_id"], "")
        docs.append(f"订单 {o['order_id']} {o['status']} 包含 {model}")

    bm25 = BM25Index(docs)
    bm25.save_json(config.GRAPH_DIR / "bm25_index.json")
    print(f"BM25 索引：{bm25.N} 篇文档")

    print("完成：图谱与索引已保存到 data/graph/")


if __name__ == "__main__":
    main()
