# -*- coding: utf-8 -*-
"""对话管理测试：槽位填充与回答策略。"""
from ecommerce_qa.nlu.dialog import DialogManager


def _session():
    return {"slots": {}}


def test_order_status_asks_for_order_id():
    """查订单状态但没有订单号时，应反问提供订单号。"""
    dm = DialogManager()
    result = dm.update(_session(), "查订单状态", {"BRAND": ["耐克"]})
    assert result["complete"] is False
    assert "ORDER_ID" in result["missing"]
    assert "订单号" in dm.ask_message("ORDER_ID")


def test_order_status_complete_with_order_id():
    """补齐订单号后应 complete。"""
    dm = DialogManager()
    result = dm.update(_session(), "查订单状态", {"ORDER_ID": ["202608130001"]})
    assert result["complete"] is True
    assert result["missing"] == []


def test_product_param_any_one_slot():
    """查商品参数只要品牌或型号任一个就满足（或的关系）。"""
    dm = DialogManager()
    r1 = dm.update(_session(), "查商品参数", {"BRAND": ["耐克"]})
    assert r1["complete"] is True
    r2 = dm.update(_session(), "查商品参数", {"PRODUCT_MODEL": ["Air Max 270"]})
    assert r2["complete"] is True


def test_response_strategy():
    """四种回答策略的分支逻辑。"""
    dm = DialogManager()
    assert dm.response_strategy("闲聊", complete=True, has_knowledge=False) == "chat"
    assert dm.response_strategy("查订单状态", complete=False, has_knowledge=False) == "ask_missing"
    assert dm.response_strategy("查订单状态", complete=True, has_knowledge=True) == "generate"
    assert dm.response_strategy("查订单状态", complete=True, has_knowledge=False) == "no_knowledge"
