# -*- coding: utf-8 -*-
"""全局配置：路径、标签集合、模型与训练超参。

整个项目所有路径与超参都从这里读取，避免散落各处。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------- 路径 ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                          # 模拟原始数据
ANNOTATED_DIR = DATA_DIR / "annotated"              # 自动标注的训练数据
GRAPH_DIR = DATA_DIR / "graph"                      # 知识图谱导出
MODELS_DIR = ROOT_DIR / "models"                    # 训练产物

for _d in (RAW_DIR, ANNOTATED_DIR, GRAPH_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 运行模式 ----------------
# True = 快速模式（少数据、少 epoch，CPU 几分钟内跑完演示）
QUICK_MODE = True

# 随机种子，保证可复现
SEED = 42

# ---------------- 标签集合 ----------------
# 命名实体类型（不含 "O"；NER 模块会自动补上 "O" 表示非实体）
ENTITIES = [
    "BRAND",            # 品牌，如 耐克
    "PRODUCT_MODEL",    # 商品型号，如 Air Max
    "PRODUCT_CATEGORY", # 商品类别，如 手机
    "PRICE",            # 价格，如 599 元
    "QUANTITY",         # 数量，如 两件
    "ORDER_ID",         # 订单号
    "RECEIVER",         # 收货人
    "STATUS",           # 订单状态
    "POLICY",           # 售后政策关键词
]

# 关系集合（头实体, 关系, 尾实体 中的"关系"）
RELATIONS = [
    "生产品牌", "属于类别", "价格为", "库存为",
    "参数为", "订单包含", "订单状态为", "适用政策",
]

# 意图集合
INTENTS = ["查商品参数", "查订单状态", "咨询售后", "查退换货政策", "闲聊"]

# ---------------- 预训练模型 ----------------
BERT_MODEL_NAME = "bert-base-chinese"

# ---------------- Deepseek ----------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def load_env():
    """加载项目根目录的 .env 文件（若有）。"""
    load_dotenv(ROOT_DIR / ".env")


def has_deepseek_key() -> bool:
    """检查是否配置了 Deepseek API Key。"""
    load_env()
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


# ---------------- 训练超参（快速模式） ----------------
NER = dict(
    max_sentences=300,   # 最大训练句子数（三类均衡抽样）
    epochs=12,           # 训练轮数（CPU 上约 3~5 秒，效果明显好于 3）
    batch_size=32,
    embed_dim=128,       # 词向量维度（随机初始化）
    hidden_dim=128,      # BiLSTM 隐层维度
)

RELATION = dict(
    max_samples=150,     # 最大训练样本数
    epochs=1,            # R-BERT 微调轮数（CPU 上建议 1）
    batch_size=8,
    freeze_bert=True,    # 快速模式下冻结 BERT 主干，只训分类头
)

INTENT = dict(
    max_samples=240,     # 最大训练样本数（每条模板 4 个变体）
    epochs=3,            # 微调轮数（意图分类需真正微调 BERT 才能学好 [CLS]）
    batch_size=16,
    freeze_bert=False,   # 冻结 BERT 时 [CLS] 区分度不足，必须微调
)
