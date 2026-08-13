# -*- coding: utf-8 -*-
"""回答模板：Deepseek 不可用时的降级回答 + 闲聊回复。"""
# 闲聊关键词回复
CHAT_KEYWORDS = [
    ("你好", "您好！我是电商智能客服，可以帮您查询商品参数、订单状态、售后政策等。"),
    ("谢谢", "不客气，很高兴为您服务！还有什么需要帮忙的吗？"),
    ("你是谁", "我是基于知识图谱的电商智能问答助手，会从知识库里检索真实信息回答您。"),
    ("在吗", "在的，请问有什么可以帮您？"),
    ("再见", "再见，祝您生活愉快！"),
    ("辛苦了", "不辛苦，能帮到您就好！"),
    ("你叫什么", "我叫小商，是电商智能问答助手。"),
]


def chat_reply(text: str) -> str:
    """闲聊回复：按关键词匹配。"""
    for keyword, reply in CHAT_KEYWORDS:
        if keyword in text:
            return reply
    return "很高兴为您服务，请问有什么可以帮您？"


def no_knowledge_reply(intent: str) -> str:
    """检索不到知识时的兜底话术。"""
    if intent == "查订单状态":
        return "很抱歉，我没有查到该订单的信息，请确认订单号是否正确，或联系人工客服。"
    if intent == "查商品参数":
        return "很抱歉，暂未找到该商品的相关信息，请尝试提供完整的品牌和型号。"
    if intent in ("咨询售后", "查退换货政策"):
        return "很抱歉，暂未查询到相关售后政策，请稍后重试或联系人工客服。"
    return "抱歉，我暂时无法回答这个问题，请联系人工客服。"


def fallback_answer(intent: str, knowledge: list, slots: dict) -> str:
    """从检索到的三元组知识拼装模板回答（Deepseek 不可用时的降级）。"""
    # 订单状态有明确的单条答案
    if intent == "查订单状态":
        for s, r, o in knowledge:
            if r == "订单状态为":
                return f"根据查询，订单 {s} 当前状态为：{o}。"
        oid = (slots.get("ORDER_ID") or [""])[0]
        return f"很抱歉，未查询到订单 {oid} 的物流状态，请确认订单号是否正确。"

    lines = []
    for s, r, o in knowledge:
        if r == "适用政策":
            lines.append(f"适用政策：{o}")
        elif r == "参数为":
            lines.append(f"参数：{o}")
        else:
            lines.append(f"{s}的{r}：{o}")
    if lines:
        return "根据知识库查询到：\n" + "\n".join(lines)
    return no_knowledge_reply(intent)
