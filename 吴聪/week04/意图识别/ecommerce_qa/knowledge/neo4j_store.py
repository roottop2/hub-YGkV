# -*- coding: utf-8 -*-
"""Neo4j 图数据库实现（生产预留）。

本演示默认使用 MemoryGraphStore（内存图，零安装）。
要切换为真实 Neo4j，需要：
  1. 安装 Neo4j 社区版（依赖 Java 17）并启动；
  2. pip install neo4j；
  3. 按下面的 Cypher 示例建索引，并实现本类的查询方法。

示例 Cypher（建节点与关系）：
    CREATE (n:实体 {name: '耐克'})
    CREATE (m:实体 {name: 'Air Max 270'})
    CREATE (m)-[:生产品牌]->(n)
    查询：
    MATCH (m)-[r]->(n) WHERE m.name = 'Air Max 270' RETURN r, n.name
"""
from ecommerce_qa.knowledge.graph_store import GraphStore


class Neo4jGraphStore(GraphStore):
    """与 GraphStore 同接口的 Neo4j 实现（未安装时不可用）。"""

    def __init__(self, uri, user, password):
        raise NotImplementedError(
            "Neo4j 为生产预留实现。本演示请使用 MemoryGraphStore。"
            "如需使用：安装 Neo4j + neo4j 驱动后实现本类各方法。"
        )
