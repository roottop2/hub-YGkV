# -*- coding: utf-8 -*-
"""Deepseek 大模型生成器（OpenAI 兼容接口）。

核心：把"检索到的知识 + 用户问题"一起交给大模型，
要求它只依据参考资料回答，从而压制"幻觉"。
Deepseek 不可用时（无 key / 网络失败 / 超时）降级为模板回答。
"""
import os

from openai import OpenAI

from ecommerce_qa import config
from ecommerce_qa.generator import templates
from ecommerce_qa.utils.logger import get_logger

logger = get_logger(__name__)

# 系统提示词：约束模型只使用参考资料，不编造
SYSTEM_PROMPT = (
    "你是电商平台的智能客服助手。请只根据用户消息后面的【参考资料】回答，"
    "不要编造参考资料中没有的信息；如果参考资料不足以回答，请如实说明。"
    "回答要简洁、准确、自然。"
)


def build_prompt(question: str, knowledge: list) -> str:
    """构造提示词：把知识列表与问题组合起来。"""
    if not knowledge:
        refs = "（没有检索到相关资料）"
    else:
        refs = "\n".join(f"- {k}" for k in knowledge)
    return (
        f"【参考资料】\n{refs}\n\n"
        f"【用户问题】\n{question}"
    )


def call_api(question: str, knowledge: list, timeout: int = 20) -> str:
    """调用 Deepseek API，返回回答文本。失败时抛出异常由调用方降级。"""
    config.load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL, timeout=timeout)
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, knowledge)},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    text = response.choices[0].message.content
    return text.strip() if text else ""


def generate(question: str, knowledge: list, intent: str = None,
             slots: dict = None, timeout: int = 20) -> tuple:
    """生成最终回答。

    Returns:
        (answer_text, source)
        source: "deepseek" 真实大模型 | "template" 降级模板 | "none" 无知识
    """
    try:
        text = call_api(question, knowledge, timeout=timeout)
        if text:
            return text, "deepseek"
    except Exception as exc:
        logger.warning("Deepseek 调用失败（%s），降级为模板回答", exc)

    intent = intent or "闲聊"
    slots = slots or {}
    if knowledge:
        return templates.fallback_answer(intent, knowledge, slots), "template"
    return templates.no_knowledge_reply(intent), "none"
