# -*- coding: utf-8 -*-
"""基于 networkx 的内存图实现（演示默认）。"""
import json

import networkx as nx

from ecommerce_qa.knowledge.graph_store import GraphStore


class MemoryGraphStore(GraphStore):
    """用 networkx 有向图存储三元组。边带 relation 属性。"""

    def __init__(self, triples: list = None):
        self.graph = nx.DiGraph()
        if triples:
            self.add_triples(triples)

    # ---- 写入 ----
    def add_triple(self, subject: str, relation: str, object: str) -> None:
        self.graph.add_edge(subject, object, relation=relation)

    def add_triples(self, triples: list) -> None:
        for s, r, o in triples:
            self.add_triple(s, r, o)

    # ---- 查询 ----
    def get_entity(self, name: str):
        return name if self.graph.has_node(name) else None

    def query_triples(self, subject=None, relation=None, object=None) -> list:
        result = []
        for u, v, data in self.graph.edges(data=True):
            if subject is not None and u != subject:
                continue
            if object is not None and v != object:
                continue
            if relation is not None and data.get("relation") != relation:
                continue
            result.append((u, data["relation"], v))
        return result

    def get_neighbors(self, entity: str, relation=None) -> list:
        result = []
        for _, v, data in self.graph.out_edges(entity, data=True):
            if relation is None or data.get("relation") == relation:
                result.append((v, data["relation"]))
        return result

    # ---- 统计 ----
    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        """导出为可 JSON 化的字典。"""
        return {
            "triples": [list(t) for t in self.query_triples()],
        }

    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path) -> "MemoryGraphStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls([tuple(t) for t in data["triples"]])
