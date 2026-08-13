# -*- coding: utf-8 -*-
"""自动标注器：为模拟数据生成三套训练数据。

1. NER 数据集：句子 + 逐字符 BIO 标签（供 BiLSTM+CRF 训练）；
2. 关系数据集：句子 + 头实体/尾实体/关系（供 R-BERT 训练）；
3. 意图数据集：问句 + 意图标签（供 BERT 意图分类训练）。

为什么能"自动标注"：句子由模板 + 已知实体拼出来，实体在句中的
起止位置是确定的，所以标签可以精确算出，无需人工标注。
"""
import json
import random

from ecommerce_qa import config
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)


def _load(name: str) -> list:
    """从 RAW_DIR 读取一份 JSON 数据。"""
    path = config.RAW_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path, records) -> None:
    """把记录列表写成 jsonl（每行一个 JSON）。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _render(template: str, parts: list) -> tuple:
    """把实体填入模板，返回 (句子, 逐字符BIO标签)。

    Args:
        template: 含 {0},{1}... 占位符的模板字符串
        parts: [(实体类型, 实体文本), ...]，按 {0}->parts[0] 依次对应

    Returns:
        (text, tags)：text 为句子，tags 为长度与 text 相同的标签列表
        （每个字符一个标签，如 "O"/"B-BRAND"/"I-BRAND"）
    """
    text = template.format(*[p[1] for p in parts])
    tags = ["O"] * len(text)
    pos = 0
    for etype, val in parts:
        start = text.find(val, pos)
        if start == -1:  # 找不到就跳过（理论上不会发生）
            continue
        for i, ch in enumerate(val):
            tags[start + i] = f"B-{etype}" if i == 0 else f"I-{etype}"
        pos = start + len(val)
    return text, tags


# --------------------------------------------------------------------------
# NER 数据集
# --------------------------------------------------------------------------

def build_ner_dataset(out_path=None, max_sentences: int = None,
                      products=None, orders=None, policies=None) -> list:
    """从商品/订单/政策数据生成 NER 标注句子。

    三类句子在截断时会均衡抽样（各约 1/3），避免政策类句子
    被数量庞大的商品类句子淹没，导致 POLICY 实体学不好。
    可直接传 products/orders/policies（测试用），默认从磁盘读取。

    Returns:
        [{"tokens": [...], "tags": [...]}, ...]
    """
    products = products if products is not None else _load("products")
    orders = orders if orders is not None else _load("orders")
    policies = policies if policies is not None else _load("policies")

    product_records, order_records, policy_records = [], [], []

    # --- 商品类句子 ---
    for p in products:
        brand, model, category = p["brand"], p["model"], p["category"]
        price = f"{p['price']}元"
        qty = random.choice(["两件", "三件", "一件"])
        templates = [
            ("{0} {1} 多少钱", [("BRAND", brand), ("PRODUCT_MODEL", model)]),
            ("我想买{0}的{1}", [("BRAND", brand), ("PRODUCT_MODEL", model)]),
            ("{0} 属于{1}吗", [("PRODUCT_MODEL", model), ("PRODUCT_CATEGORY", category)]),
            ("{0} 的价格是{1}", [("PRODUCT_MODEL", model), ("PRICE", price)]),
            ("{0} 有货吗", [("BRAND", brand)]),
            ("我买了{0}{1}", [("QUANTITY", qty), ("PRODUCT_MODEL", model)]),
        ]
        for tpl, parts in templates:
            text, tags = _render(tpl, parts)
            product_records.append({"tokens": list(text), "tags": tags})

    # --- 订单类句子 ---
    for o in orders:
        order_id, user, status = o["order_id"], o["user"], o["status"]
        templates = [
            ("我的订单{0}到哪了", [("ORDER_ID", order_id)]),
            ("订单{0}发货了吗", [("ORDER_ID", order_id)]),
            ("订单{0}的状态是{1}吗", [("ORDER_ID", order_id), ("STATUS", status)]),
            ("收货人{0}的订单{1}签收了吗", [("RECEIVER", user), ("ORDER_ID", order_id), ("STATUS", "已签收")]),
        ]
        for tpl, parts in templates:
            text, tags = _render(tpl, parts)
            order_records.append({"tokens": list(text), "tags": tags})

    # --- 政策类句子（模板更多，保证 POLICY 实体有足够样本）---
    for pol in policies:
        title = pol["title"]
        templates = [
            ("可以{0}吗", [("POLICY", title)]),
            ("{0}怎么申请", [("POLICY", title)]),
            ("支持{0}吗", [("POLICY", title)]),
            ("我想申请{0}", [("POLICY", title)]),
            ("{0}需要什么条件", [("POLICY", title)]),
            ("请问{0}是什么意思", [("POLICY", title)]),
            ("{0}的规则是什么", [("POLICY", title)]),
            ("{0}划算吗", [("POLICY", title)]),
        ]
        for tpl, parts in templates:
            text, tags = _render(tpl, parts)
            policy_records.append({"tokens": list(text), "tags": tags})

    # 三类均衡抽样：商品/订单/政策 各约占 1/3，避免政策被淹没
    random.seed(config.SEED)
    random.shuffle(product_records)
    random.shuffle(order_records)
    random.shuffle(policy_records)
    if max_sentences:
        third = max(1, max_sentences // 3)
        product_records = product_records[:third]
        order_records = order_records[:third]
        policy_records = policy_records[:third]
    records = product_records + order_records + policy_records
    random.shuffle(records)

    if out_path:
        _write_jsonl(out_path, records)
        logger.info("NER 数据集：%d 句 -> %s", len(records), out_path)
    return records


# --------------------------------------------------------------------------
# 关系数据集
# --------------------------------------------------------------------------

def build_relation_dataset(out_path=None, max_samples: int = None,
                           products=None, orders=None, policies=None) -> list:
    """从结构化数据生成关系抽取样本。

    Returns:
        [{"text", "head", "tail", "relation"}, ...]
    """
    products = products if products is not None else _load("products")
    orders = orders if orders is not None else _load("orders")
    policies = policies if policies is not None else _load("policies")

    records = []

    # --- 商品三元组（生产品牌/属于类别/价格为/库存为/参数为）---
    for p in products:
        model = p["model"]
        triples = [
            ("生产品牌", model, p["brand"], "{0} 是 {1} 生产的"),
            ("属于类别", model, p["category"], "{0} 属于{1}"),
            ("价格为", model, f"{p['price']}元", "{0} 的价格是 {1}"),
            ("库存为", model, f"{p['stock']}件", "{0} 的库存是 {1}"),
        ]
        for rel, head, tail, tpl in triples:
            records.append({"text": tpl.format(head, tail), "head": head, "tail": tail, "relation": rel})
        for param in p["params"]:
            records.append({
                "text": f"{model} 支持{param}",
                "head": model, "tail": param, "relation": "参数为",
            })

    # --- 订单三元组（订单包含/订单状态为）---
    pid2model = {p["id"]: p["model"] for p in products}
    for o in orders:
        model = pid2model.get(o["product_id"], "未知商品")
        records.append({
            "text": f"订单 {o['order_id']} 包含 {model}",
            "head": o["order_id"], "tail": model, "relation": "订单包含",
        })
        records.append({
            "text": f"订单 {o['order_id']} 的状态是 {o['status']}",
            "head": o["order_id"], "tail": o["status"], "relation": "订单状态为",
        })

    # --- 政策三元组（适用政策，作用于部分商品）---
    for p in products[:20]:
        pol = random.choice(policies)
        records.append({
            "text": f"{p['model']} 适用于 {pol['title']}",
            "head": p["model"], "tail": pol["title"], "relation": "适用政策",
        })

    random.seed(config.SEED)
    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    if out_path:
        _write_jsonl(out_path, records)
        logger.info("关系数据集：%d 条 -> %s", len(records), out_path)
    return records


# --------------------------------------------------------------------------
# 意图数据集
# --------------------------------------------------------------------------

def build_intent_dataset(out_path=None, products=None) -> list:
    """从意图问法模板生成意图分类样本。

    Returns:
        [{"text", "intent"}, ...]
    """
    products = products if products is not None else _load("products")
    models = [p["model"] for p in products]
    brands = list({p["brand"] for p in products})

    INTENT_TEMPLATES = {
        "查商品参数": [
            lambda: f"这款{random.choice(models)}支持5G吗",
            lambda: f"{random.choice(brands)}的{random.choice(models)}多少钱",
            lambda: "这个手机内存多大",
            lambda: f"{random.choice(models)}参数有哪些",
            lambda: f"有{random.choice(brands)}的货吗",
            lambda: "这款手机防水吗",
            lambda: f"{random.choice(models)}是什么配置",
            lambda: f"{random.choice(models)}多少钱",
            lambda: f"{random.choice(brands)}的鞋多少钱",
            lambda: "这个手机电池多大",
            lambda: f"{random.choice(models)}有128G的吗",
            lambda: "支持快充吗",
        ],
        "查订单状态": [
            lambda: "我的订单到哪了",
            lambda: "订单发货了吗",
            lambda: "帮我查一下订单状态",
            lambda: f"我买的{random.choice(models)}到哪了",
            lambda: "物流到哪了",
            lambda: "东西发货没有",
            lambda: "我的包裹到哪了",
            lambda: "订单202608130001到哪了",
            lambda: "帮我查查物流",
            lambda: "我的货怎么还没到",
            lambda: "什么时候发货",
            lambda: "签收了吗",
        ],
        "咨询售后": [
            lambda: "可以退货吗",
            lambda: "商品坏了怎么办",
            lambda: "怎么申请售后",
            lambda: "质量问题找谁",
            lambda: "东西有问题怎么办",
            lambda: "可以换货吗",
            lambda: "怎么申请退款",
            lambda: "商品有瑕疵",
            lambda: "售后电话多少",
            lambda: "鞋子开胶了怎么办",
            lambda: "能退货吗",
            lambda: "怎么联系客服",
        ],
        "查退换货政策": [
            lambda: "退货政策是什么",
            lambda: "7天无理由怎么算",
            lambda: "运费谁出",
            lambda: "退换货规则",
            lambda: "可以无理由退货吗",
            lambda: "退货要运费吗",
            lambda: "多久能退货",
            lambda: "换货流程是什么",
            lambda: "退货运费谁承担",
            lambda: "无理由退货的条件",
            lambda: "可以拒收吗",
            lambda: "退货需要什么",
        ],
        "闲聊": [
            lambda: "你好",
            lambda: "谢谢",
            lambda: "你是谁",
            lambda: "在吗",
            lambda: "早上好",
            lambda: "再见",
            lambda: "哈哈",
            lambda: "辛苦了",
            lambda: "你叫什么",
            lambda: "好的",
            lambda: "嗯嗯",
            lambda: "感谢",
        ],
    }

    records = []
    for intent, makers in INTENT_TEMPLATES.items():
        for maker in makers:
            # 每条模板生成 4 个变体（随机抽不同品牌/型号），增加样本量
            for _ in range(4):
                records.append({"text": maker(), "intent": intent})

    random.seed(config.SEED)
    random.shuffle(records)

    if out_path:
        _write_jsonl(out_path, records)
        logger.info("意图数据集：%d 条 -> %s", len(records), out_path)
    return records


# 便捷入口：一步生成全部标注
def build_all() -> None:
    build_ner_dataset(out_path=config.ANNOTATED_DIR / "ner.jsonl", max_sentences=config.NER["max_sentences"])
    build_relation_dataset(out_path=config.ANNOTATED_DIR / "relation.jsonl", max_samples=config.RELATION["max_samples"])
    build_intent_dataset(out_path=config.ANNOTATED_DIR / "intent.jsonl")


if __name__ == "__main__":
    build_all()
    print("自动标注完成。")
