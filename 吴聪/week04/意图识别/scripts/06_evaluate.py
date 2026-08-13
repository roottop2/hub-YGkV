# -*- coding: utf-8 -*-
"""脚本 06：模型评测 与 端到端满意度模拟。

指标：
  1. NER F1（token 级，忽略 O）
  2. 关系抽取准确率
  3. 意图分类准确率
  4. 端到端满意度模拟（检索命中率）

运行：python scripts/06_evaluate.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecommerce_qa import config  # noqa: E402
from ecommerce_qa.pipeline import EcommerceQA  # noqa: E402
from ecommerce_qa.utils.metrics import sequence_f1  # noqa: E402


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_ner(qa):
    """在 ner.jsonl 子集上计算 NER F1。"""
    records = _load_jsonl(config.ANNOTATED_DIR / "ner.jsonl")[:60]
    if qa.ner is None:
        return None
    gold, pred = [], []
    for r in records:
        gold.append(r["tags"])
        pred.append(qa.ner.predict_tags("".join(r["tokens"])))
    return sequence_f1(gold, pred)


def eval_relation(qa):
    """在 relation.jsonl 子集上计算关系准确率。"""
    samples = _load_jsonl(config.ANNOTATED_DIR / "relation.jsonl")[:60]
    if qa.rel is None:
        return None
    correct = 0
    for s in samples:
        try:
            if qa.rel.predict(s["text"], s["head"], s["tail"]) == s["relation"]:
                correct += 1
        except Exception:
            continue
    return correct / len(samples)


def eval_intent(qa):
    """在 intent.jsonl 上计算意图准确率。"""
    samples = _load_jsonl(config.ANNOTATED_DIR / "intent.jsonl")[:120]
    if qa.intent is None:
        return None
    correct = sum(1 for s in samples if qa.intent.predict(s["text"])[0] == s["intent"])
    return correct / len(samples)


def eval_end_to_end(qa):
    """端到端满意度模拟：构造 20 个标准问题，检查回答是否检索到正确答案。"""
    questions = [
        ("查一下 202608130001 的状态", "202608130001", "订单状态为"),
        ("我的订单到哪了", None, "订单状态为"),
        ("耐克 Air Max 270 多少钱", "Air Max 270", "价格为"),
        ("小米15 支持什么", "小米15", "参数为"),
        ("可以7天无理由退货吗", None, "适用政策"),
    ]
    # 用多个随机商品/订单生成更多测试问题
    products = json.load(open(config.RAW_DIR / "products.json", encoding="utf-8"))
    orders = json.load(open(config.RAW_DIR / "orders.json", encoding="utf-8"))
    random.seed(config.SEED)
    for _ in range(8):
        p = random.choice(products)
        questions.append((f"{p['model']} 是什么价格", p["model"], "价格为"))
    for _ in range(7):
        o = random.choice(orders)
        questions.append((f"我的订单{o['order_id']}到哪了", o["order_id"], "订单状态为"))

    satisfied = 0
    total = 0
    for question, entity, expect_rel in questions:
        session = {"slots": {}}
        result = qa.answer(question, session)
        total += 1
        # 判定：检索到的知识中是否命中期望的关系/实体
        hit = False
        if result["knowledge"]:
            for k in result["knowledge"]:
                if expect_rel in k and (entity is None or entity in k):
                    hit = True
                    break
        if hit:
            satisfied += 1
    return satisfied / max(total, 1)


def main():
    print("== 06 模型评测 ==")
    qa = EcommerceQA()
    print("系统状态：")
    for name, ready in qa.status().items():
        print(f"  {name}: {'就绪' if ready else '未就绪'}")

    ner = eval_ner(qa)
    rel = eval_relation(qa)
    intent = eval_intent(qa)
    e2e = eval_end_to_end(qa)

    print("\n== 评测结果 ==")
    if ner is not None:
        print(f"NER F1：{ner['f1']:.4f}（P={ner['precision']:.4f} R={ner['recall']:.4f}）")
    if rel is not None:
        print(f"关系抽取准确率：{rel:.4f}")
    if intent is not None:
        print(f"意图分类准确率：{intent:.4f}")
    if e2e is not None:
        print(f"端到端满意度模拟：{e2e:.4f}")
    print("说明：快速模式在少量模拟数据上训练，指标用于验证流程正确性；"
          "生产环境需扩大数据量并在 GPU 上训练（见设计文档）。")


if __name__ == "__main__":
    main()
