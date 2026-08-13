# -*- coding: utf-8 -*-
"""生成器测试：提示词构造、无 key 降级、模板回答。"""
import os

from ecommerce_qa.generator import templates
from ecommerce_qa.generator.deepseek_llm import build_prompt, generate


def test_build_prompt_contains_content():
    """提示词应包含参考资料与用户问题。"""
    prompt = build_prompt("我的订单到哪了", ["202608130001 订单状态为 已发货"])
    assert "我的订单到哪了" in prompt
    assert "已发货" in prompt


def test_generate_without_key_falls_back(monkeypatch):
    """没有 API key 时，应返回模板回答而不是抛异常。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")   # 模拟未配置 key
    text, source = generate(
        "我的订单到哪了",
        [("202608130001", "订单状态为", "已发货")],
        intent="查订单状态",
        slots={"ORDER_ID": ["202608130001"]},
    )
    assert source == "template"
    assert "已发货" in text


def test_generate_no_knowledge(monkeypatch):
    """无知识时返回无知识话术（不走真实网络调用）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    text, source = generate("你好", [], intent="闲聊")
    assert isinstance(text, str) and text
    assert source == "none"


def test_fallback_order_status():
    """订单状态模板回答能提取出状态。"""
    answer = templates.fallback_answer(
        "查订单状态",
        [("202608130001", "订单状态为", "已发货")],
        {"ORDER_ID": ["202608130001"]},
    )
    assert "已发货" in answer


def test_chat_reply():
    """闲聊回复包含礼貌用语。"""
    reply = templates.chat_reply("你好")
    assert "客服" in reply or "您好" in reply
