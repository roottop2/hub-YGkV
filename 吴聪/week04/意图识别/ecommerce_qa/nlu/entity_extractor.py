# -*- coding: utf-8 -*-
"""实体抽取器：NER 模型 + 词典匹配 + 正则兜底 三重保障。

设计理由：快速模式下 NER 模型在少量数据上训练，覆盖不了所有实体类型。
词典（从原始数据构建）保证品牌/型号/状态/政策等高频实体稳定命中，
正则兜底保证订单号/价格这类"有固定格式"的实体必中。
"""
import json
import re
from collections import defaultdict

from ecommerce_qa import config


class EntityExtractor:
    """把用户句子中的实体按类型抽取出来。"""

    def __init__(self, ner_predictor=None, known_entities: dict = None):
        self.ner = ner_predictor
        self.known = known_entities or {}

    @staticmethod
    def build_known_entities() -> dict:
        """从原始数据构建已知实体词典。

        Returns:
            {"BRAND": [..], "PRODUCT_MODEL": [..], "PRODUCT_CATEGORY": [..],
             "STATUS": [..], "POLICY": [..]}
        """
        def _load(name):
            p = config.RAW_DIR / f"{name}.json"
            if not p.exists():
                return []
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        products, orders, policies = _load("products"), _load("orders"), _load("policies")

        known = {
            "BRAND": sorted({p["brand"] for p in products}),
            "PRODUCT_MODEL": sorted({p["model"] for p in products}),
            "PRODUCT_CATEGORY": sorted({p["category"] for p in products}),
            "STATUS": sorted({o["status"] for o in orders}),
            "POLICY": sorted({pol["title"] for pol in policies}),
        }
        return known

    def extract(self, text: str) -> dict:
        """返回 {实体类型: [实体值, ...]}。"""
        result = defaultdict(list)
        seen = set()

        def add(etype, value):
            if not value:
                return
            if (etype, value) not in seen:
                seen.add((etype, value))
                result[etype].append(value)

        # 1) 词典匹配（最可靠，先做）
        for etype, names in self.known.items():
            for name in names:
                if name and name in text:
                    add(etype, name)

        # 2) NER 模型补充（只接受闭集类型中的已知值，保证精确）
        #    说明：品牌/型号/状态/政策在我们的数据里是封闭集合，NER 的
        #    泛化结果可能含噪点（如把"可以7天无理由"误判为品牌），
        #    所以只在命中已知词典时才采纳。ORDER_ID/PRICE/QUANTITY
        #    这类固定格式交给第 3 步的正则。
        if self.ner is not None:
            try:
                for ent, etype in self.ner.predict(text):
                    known_vals = self.known.get(etype)
                    if known_vals is not None and ent in known_vals:
                        add(etype, ent)
            except Exception:
                pass  # NER 异常不影响整体抽取

        # 3) 正则兜底（固定格式的实体）
        for oid in re.findall(r"\d{12}", text):
            add("ORDER_ID", oid)
        for price in re.findall(r"\d+\s*元", text):
            add("PRICE", price)
        for qty in re.findall(r"[一二两三四五六七八九十百千]+\s*件|\d+\s*件", text):
            add("QUANTITY", qty)

        return dict(result)
