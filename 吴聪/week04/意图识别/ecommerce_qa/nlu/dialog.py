# -*- coding: utf-8 -*-
"""对话管理：槽位填充、对话流程树、回答策略。

"对话流程树"在这里实现为一张表：每个意图 -> 需要的槽位组。
每个组是"或"的关系（任一槽位满足即可），缺了就反问用户补全。
"""
# 每个意图需要的槽位组（每组内任意一个槽位有值即可满足）
# 例：查商品参数 需要 BRAND/PRODUCT_MODEL/PRODUCT_CATEGORY 中至少一个
SLOT_RULES = {
    "查商品参数": [("BRAND", "PRODUCT_MODEL", "PRODUCT_CATEGORY")],
    "查订单状态": [("ORDER_ID",)],
    "咨询售后": [],              # 无硬性要求，有则更好
    "查退换货政策": [],
    "闲聊": [],
}

# 缺槽位时的反问话术
ASK_MESSAGES = {
    "ORDER_ID": "请提供您的订单号（12 位数字），我来帮您查询。",
    "BRAND": "请问您想查询哪个品牌的商品呢？",
    "PRODUCT_MODEL": "请问您想查询哪款商品（型号）呢？",
    "PRODUCT_CATEGORY": "请问您想查询哪一类商品呢？",
}


class DialogManager:
    """维护会话状态（槽位/意图），决定下一步动作。"""

    def update(self, session: dict, intent: str, slots: dict) -> dict:
        """合并新槽位，检查缺失，返回更新后的会话状态。

        Args:
            session: 会话 dict（含 "slots" 等字段）
            intent: 识别到的意图
            slots: 本轮抽取的 {槽位类型: [值, ...]}
        Returns:
            {"slots": 合并后槽位, "missing": 缺失槽位列表,
             "complete": 是否已满足查询条件}
        """
        # 换意图时清空旧槽位，避免跨意图串扰
        # （例如问完订单状态再问商品参数，不应继续携带订单号）
        if session.get("intent") and session["intent"] != intent:
            session["slots"] = {}

        merged = dict(session.get("slots", {}))
        for key, values in slots.items():
            if values:
                merged[key] = values   # 槽位按类型存一组候选值

        session["slots"] = merged
        session["intent"] = intent
        missing = self._missing(intent, merged)
        return {
            "slots": merged,
            "missing": missing,
            "complete": len(missing) == 0,
        }

    def _missing(self, intent: str, slots: dict) -> list:
        """返回缺失的槽位类型（每个不满足的组返回其第一个槽位）。"""
        missing = []
        for group in SLOT_RULES.get(intent, []):
            if not any(slots.get(key) for key in group):
                missing.append(group[0])
        return missing

    def ask_message(self, slot_type: str) -> str:
        """根据缺失槽位生成反问话术。"""
        return ASK_MESSAGES.get(slot_type, "请补充相关信息，谢谢。")

    def response_strategy(self, intent: str, complete: bool,
                          has_knowledge: bool) -> str:
        """决定回答策略：chat / ask_missing / generate / no_knowledge。"""
        if intent == "闲聊":
            return "chat"
        if not complete:
            return "ask_missing"
        if has_knowledge:
            return "generate"
        return "no_knowledge"
