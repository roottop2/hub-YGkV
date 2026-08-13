# -*- coding: utf-8 -*-
"""模拟数据生成器：生成商品 / 订单 / 售后政策 三类电商数据。

为什么要"模拟"：真实电商数据需要数据部门提供，本学习项目先用
程序生成贴近真实的模拟数据，替代"数据部门提供数据"这一环节。
生成时固定随机种子，保证每次生成一致、结果可复现。
"""
import json
import random
from datetime import date

from ecommerce_qa import config
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)

# 品牌池（实体统一演示：后面会把 nike 这类别名并入 耐克）
BRANDS = ["耐克", "阿迪达斯", "华为", "小米", "苹果", "李宁"]

# 每个品牌对应的型号池
BRAND_MODELS = {
    "耐克": ["Air Max 270", "Air Force 1", "Pegasus 40", "Dunk Low"],
    "阿迪达斯": ["Ultraboost", "Superstar", "NMD R1", "Campus"],
    "华为": ["Mate 60 Pro", "P60 Pro", "MateBook 14", "FreeBuds 5"],
    "小米": ["小米15", "Redmi K70", "小米手环9", "小米14"],
    "苹果": ["iPhone 15", "iPhone 15 Pro", "MacBook Air", "AirPods Pro"],
    "李宁": ["飞电3", "赤兔6", "超轻21", "利刃4"],
}

# 类别池（按品牌大致匹配）
CATEGORIES = ["跑鞋", "运动鞋", "手机", "笔记本电脑", "蓝牙耳机", "智能手表"]

# 商品参数池
PARAMS = ["5G网络", "128G存储", "5000毫安电池", "防水", "降噪", "OLED屏幕", "快充"]

# 订单状态池
ORDER_STATUSES = ["待付款", "已发货", "运输中", "已签收", "已退货"]

# 售后政策池
POLICIES = [
    {"title": "7天无理由退货", "content": "自签收之日起7天内，支持无理由退货，商品需完好、吊牌齐全，运费由买家承担。"},
    {"title": "15天质量问题换货", "content": "签收后15天内，如出现质量问题，可免费换货，运费由卖家承担。"},
    {"title": "运费险保障", "content": "购买商品时赠送运费险，退货时退货运费由保险公司赔付。"},
    {"title": "支持无理由退货", "content": "大部分商品支持无理由退货，定制类、生鲜类商品除外。"},
    {"title": "30天包换", "content": "签收后30天内出现非人为损坏，可免费包换同款商品。"},
    {"title": "全国联保", "content": "所有商品支持全国联保，凭发票可在任意授权网点享受售后服务。"},
    {"title": "假一赔三", "content": "如鉴定为假货，商家按商品价格的3倍进行赔偿。"},
    {"title": "极速退款", "content": "符合极速退款条件的订单，商家在2小时内发起退款，无需等待退货。"},
]

# 收货人池
RECEIVERS = ["张三", "李四", "王五", "赵六", "钱七"]


def _new_order_id(seed_products: list) -> str:
    """生成订单号：2026MMDD + 4 位随机数（如 202608130001）。"""
    today = date.today().strftime("%Y%m%d")
    return f"{today}{random.randint(1000, 9999)}"


def generate_products(n: int) -> list:
    """生成 n 条商品数据。"""
    products = []
    for i in range(n):
        brand = random.choice(BRANDS)
        model = random.choice(BRAND_MODELS[brand])
        category = random.choice(CATEGORIES)
        products.append({
            "id": f"P{i+1:04d}",
            "brand": brand,
            "model": model,
            "category": category,
            "price": random.randint(99, 9999),
            "stock": random.randint(0, 500),
            "params": random.sample(PARAMS, k=random.randint(1, 3)),
        })
    return products


def generate_orders(m: int, products: list) -> list:
    """生成 m 条订单数据，引用已有商品。"""
    orders = []
    for i in range(m):
        product = random.choice(products)
        orders.append({
            "id": f"O{i+1:04d}",
            "order_id": _new_order_id(products),
            "user": random.choice(RECEIVERS),
            "product_id": product["id"],
            "status": random.choice(ORDER_STATUSES),
        })
    return orders


def generate_policies() -> list:
    """生成售后政策数据。"""
    return [
        {"id": f"POL{idx+1:03d}", "title": p["title"], "content": p["content"], "category": "售后"}
        for idx, p in enumerate(POLICIES)
    ]


def generate_all(product_n: int = 120, order_m: int = 80) -> None:
    """生成全部数据并写入 RAW_DIR。

    Args:
        product_n: 商品数量
        order_m: 订单数量
    """
    random.seed(config.SEED)
    products = generate_products(product_n)
    orders = generate_orders(order_m, products)
    policies = generate_policies()

    # 保存为 JSON（ensure_ascii=False 让中文可读）
    for name, data in [("products", products), ("orders", orders), ("policies", policies)]:
        path = config.RAW_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("已生成 %s：%s（%d 条）", name, path, len(data))


if __name__ == "__main__":
    generate_all()
    print("模拟数据生成完成。")
