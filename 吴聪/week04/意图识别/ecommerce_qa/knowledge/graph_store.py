# -*- coding: utf-8 -*-
"""图存储抽象接口。

统一"读写知识图谱"的方式，屏蔽底层实现。
演示版用 MemoryGraphStore（内存图），生产版可换成 Neo4jGraphStore，
上层代码（检索、对话管理）无需改动。
"""
from abc import ABC, abstractmethod


class GraphStore(ABC):
    """知识图谱的统一读写接口。三元组形式：(头实体, 关系, 尾实体)。"""

    @abstractmethod
    def add_triple(self, subject: str, relation: str, object: str) -> None:
        """写入一条三元组。"""

    @abstractmethod
    def get_entity(self, name: str):
        """按名字查实体节点，不存在返回 None。"""

    @abstractmethod
    def query_triples(self, subject=None, relation=None, object=None) -> list:
        """按任意条件查询三元组，条件为 None 表示不限。

        Returns:
            [(subject, relation, object), ...]
        """

    @abstractmethod
    def get_neighbors(self, entity: str, relation=None) -> list:
        """查询实体的相邻三元组（作为头实体出发）。

        Returns:
            [(object, relation), ...]
        """

    @property
    @abstractmethod
    def node_count(self) -> int:
        """节点数。"""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """关系数。"""
