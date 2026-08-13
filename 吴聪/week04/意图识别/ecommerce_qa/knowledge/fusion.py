# -*- coding: utf-8 -*-
"""知识融合：实体消歧 + 实体统一 + 关系对齐。

对应原技术方案的三步：
  1. 实体统一（规则）：nike -> 耐克
  2. 实体消歧（TF-IDF 相似度）：相近写法视为同一实体
  3. 关系对齐（同义词映射）：品牌/属于品牌 -> 生产品牌
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 实体别名表（规范化后的小写名 -> 标准名）
ENTITY_ALIASES = {
    "nike": "耐克",
    "adidas": "阿迪达斯",
    "huawei": "华为",
    "xiaomi": "小米",
    "iphone 15": "iPhone 15",
    "iphone15": "iPhone 15",
    "苹果手机": "苹果",
    "apple": "苹果",
}

# 关系同义词映射（-> 唯一标准关系名）
RELATION_SYNONYMS = {
    "品牌": "生产品牌",
    "属于品牌": "生产品牌",
    "生产商": "生产品牌",
    "价格": "价格为",
    "价格是": "价格为",
    "库存": "库存为",
    "类别": "属于类别",
    "状态": "订单状态为",
}


def normalize(name: str) -> str:
    """规范化：去空白、转小写，便于做别名匹配。"""
    return str(name).strip().lower()


def unify_entity(name: str) -> str:
    """实体统一：别名表命中返回标准名；否则保留原名（仅去空白）。

    注意：不能把未命中的实体转小写，否则 "Air Max 270" 会变成
    "air max 270"，导致后面图谱精确查询时对不上。
    """
    norm = normalize(name)
    if norm in ENTITY_ALIASES:
        return ENTITY_ALIASES[norm]
    return str(name).strip()


def unify_relation(relation: str) -> str:
    """关系对齐：把同义词映射到标准关系名。"""
    return RELATION_SYNONYMS.get(str(relation).strip(), str(relation).strip())


def disambiguate(triples: list, threshold: float = 0.9) -> list:
    """实体消歧：TF-IDF 字符向量化，余弦相似度 >= 阈值视为同一实体并合并。

    把高相似的实体名统一成"最早出现的那个"。例如 "耐克" 与 "耐克 "
    相似度很高，会被合并。
    """
    entities = []
    for s, r, o in triples:
        entities.extend([s, o])
    entities = sorted(set(entities))
    if len(entities) < 2:
        return triples

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2))
    X = vectorizer.fit_transform(entities).toarray()

    canonical = {}          # 实体 -> 规范名
    for i, ent in enumerate(entities):
        for j in range(i):
            sim = cosine_similarity(X[i:i + 1], X[j:j + 1])[0][0]
            if sim >= threshold:
                canonical[ent] = canonical.get(entities[j], entities[j])
                break
        else:
            canonical[ent] = ent

    return [
        (canonical.get(s, s), r, canonical.get(o, o))
        for s, r, o in triples
    ]


def merge_triples(triples: list) -> list:
    """完整融合流程：统一实体 -> 对齐关系 -> 消歧 -> 去重（保序）。"""
    triples = [
        (unify_entity(s), unify_relation(r), unify_entity(o))
        for s, r, o in triples
    ]
    triples = disambiguate(triples)
    return list(dict.fromkeys(triples))
