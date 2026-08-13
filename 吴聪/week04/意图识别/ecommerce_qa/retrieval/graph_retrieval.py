# -*- coding: utf-8 -*-
"""图检索：根据 NLU 提取的实体，沿图谱查询邻居三元组。"""
from ecommerce_qa.knowledge.graph_store import GraphStore


class GraphRetriever:
    """基于知识图谱的结构化检索。

    对每个识别出的实体，取它以头实体出发的邻居三元组，
    如 (耐克) -> (Air Max 270, 生产品牌) 表示"耐克生产了 Air Max 270"。
    """

    def __init__(self, store: GraphStore):
        self.store = store

    def search(self, entities: list, top_k: int = 10) -> list:
        """按实体集合检索，返回去重后的三元组列表。"""
        result = []
        seen = set()
        for ent in entities:
            # 实体可能在图谱中不存在（如新商品），跳过
            if self.store.get_entity(ent) is None:
                continue
            for obj, rel in self.store.get_neighbors(ent):
                triple = (ent, rel, obj)
                if triple not in seen:
                    seen.add(triple)
                    result.append(triple)
            if len(result) >= top_k:
                break
        return result[:top_k]
