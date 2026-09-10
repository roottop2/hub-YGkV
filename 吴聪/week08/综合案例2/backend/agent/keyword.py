"""KeywordAgent —— 规划员：把研究主题拆成一组搜索关键词。

对应研究流程的第一环「规划（拆子问题）」。这一步的产物会写进
ResearchProcess.plan，也是后面每一轮检索的起点。
"""

from __future__ import annotations

import logging

from ..models import KeywordOutput
from .base import BaseAgent

logger = logging.getLogger(__name__)


class KeywordAgent(BaseAgent):
    """主题 → 关键词列表。一次 LLM 调用，无工具。"""

    name = "keyword"
    template = "keyword_agent.jinja2"

    async def generate_keywords(self, topic: str, max_keywords: int = 5) -> list[str]:
        """把一个研究主题拆成若干搜索关键词。

        返回的是清洗过的列表：去掉空白、去掉空串、去重（保持原顺序）。
        去重很有必要——模型偶尔会把同一个词换个写法给两遍。
        """
        topic = (topic or "").strip()
        if not topic:
            return []

        output = await self.call_json(
            system_vars={"topic": topic, "max_keywords": max_keywords},
            user_input=f"请为这个研究主题生成搜索关键词：{topic}",
            output_cls=KeywordOutput,
        )

        keywords = self._clean(output.keywords, limit=max_keywords)
        logger.info("[keyword] 主题「%s」生成 %d 个关键词：%s", topic, len(keywords), keywords)
        return keywords

    @staticmethod
    def _clean(raw: list[str], limit: int) -> list[str]:
        """清洗 + 去重 + 截断。"""
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw:
            keyword = (item or "").strip()
            if not keyword:
                continue
            # 用 casefold 做大小写无关的去重：RAG 和 rag 算同一个
            key = keyword.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(keyword)
            if len(cleaned) >= limit:
                break
        return cleaned


if __name__ == "__main__":
    # 需要网络与密钥，人工验证用：python3 -m backend.agent.keyword
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _demo() -> None:
        topic = "2026 年国内大模型 API 的价格趋势"
        print(f"研究主题：{topic}\n")

        agent = KeywordAgent()
        keywords = await agent.generate_keywords(topic, max_keywords=5)

        print(f"生成 {len(keywords)} 个关键词：")
        for i, kw in enumerate(keywords, 1):
            print(f"  {i}. {kw}")

        assert keywords, "一个关键词都没生成"
        assert len(keywords) <= 5, "超过了数量上限"
        assert len(set(keywords)) == len(keywords), "有重复关键词"
        print("\n检查通过：非空、未超上限、无重复。")

    asyncio.run(_demo())
