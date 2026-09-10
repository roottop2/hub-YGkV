"""JudgeAgent —— 审核员：判断材料够不够，不够就给出补检关键词。

对应研究流程的「判断是否需要补检」这一环。
**这一环是"深度研究"区别于"一次性问答"的关键**：没有它，检索一轮就出报告。
"""

from __future__ import annotations

import logging

from ..models import JudgeDecision, Source
from ..tools import domain_of
from .base import BaseAgent

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """(主题, 草稿, 已有来源, 已搜关键词) → JudgeDecision。一次 LLM 调用，无工具。"""

    name = "judge"
    template = "judge_agent.jinja2"

    async def judge(
        self,
        topic: str,
        draft_text: str,
        sources: list[Source],
        searched: list[str],
        round_no: int = 1,
        max_rounds: int = 3,
        max_new_keywords: int = 3,
    ) -> JudgeDecision:
        """判断现有草稿能否支撑一份完整报告。

        注意：**这里不做"轮数是否到上限"的判断**，那是引擎的职责。
        本 agent 只管"信息够不够"这一件事，职责单一才好调试。
        """
        domains = {domain_of(s.url) for s in sources if domain_of(s.url)}

        decision = await self.call_json(
            system_vars={
                "topic": topic,
                "round_no": round_no,
                "max_rounds": max_rounds,
                "searched_count": len(searched),
                "searched_text": self._format_searched(searched),
                "source_count": len(sources),
                "domain_count": len(domains),
                "draft_text": draft_text or "（草稿还是空的）",
                "max_new_keywords": max_new_keywords,
            },
            user_input="请判断现有材料是否足够写出完整报告，并给出 JSON。",
            output_cls=JudgeDecision,
        )

        self._sanitize(decision, searched, max_new_keywords)

        logger.info(
            "[judge] 第 %d 轮判断：%s（%s）%s",
            round_no,
            "材料已足够" if decision.sufficient else "需要补检",
            decision.reason or "未给理由",
            "" if decision.sufficient else f"→ 补检关键词 {decision.new_keywords}",
        )
        return decision

    @staticmethod
    def _format_searched(searched: list[str]) -> str:
        if not searched:
            return "（无）"
        return "\n".join(f"{i}. {kw}" for i, kw in enumerate(searched, 1))

    @staticmethod
    def _sanitize(decision: JudgeDecision, searched: list[str], max_new_keywords: int) -> None:
        """清洗模型的判断结果，直接把不合理的部分改掉（原地修改）。

        做三件事：
        1. 说不清楚却给了补检词的，以"需要补检"为准（宁可多搜一轮，别漏信息）。
        2. 去掉已经搜过的关键词——重复搜同一批词只会拿到同样的网页，纯浪费。
        3. 截断到数量上限。
        """
        if decision.sufficient:
            # 已经够了，补检词就没有意义了，清空避免引擎误用
            decision.new_keywords = []
            return

        seen = {kw.strip().casefold() for kw in searched}
        fresh: list[str] = []
        for item in decision.new_keywords:
            keyword = (item or "").strip()
            if not keyword or keyword.casefold() in seen:
                continue
            seen.add(keyword.casefold())
            fresh.append(keyword)
            if len(fresh) >= max_new_keywords:
                break

        dropped = len(decision.new_keywords) - len(fresh)
        if dropped > 0:
            logger.info("[judge] 过滤掉 %d 个重复/超额的补检关键词", dropped)
        decision.new_keywords = fresh


if __name__ == "__main__":
    # 需要网络与密钥，人工验证用：python3 -m backend.agent.judge
    import asyncio

    from ..models import DraftBlock, Source

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _demo() -> None:
        agent = JudgeAgent()
        topic = "2026 年国内大模型 API 的价格趋势"

        # 场景一：草稿很单薄，应当被判为"需要补检"
        thin = [
            DraftBlock(
                heading="大模型 API 价格",
                body="目前国内大模型 API 的价格整体呈下降趋势，但缺乏具体的价格对比数据。",
            )
        ]
        thin_sources = [
            Source(url="https://www.example.com/a", title="某篇文章", site_name="example")
        ]

        print("【场景一】草稿单薄")
        d1 = await agent.judge(topic, thin[0].body, thin_sources, ["大模型 API 价格"])
        print(f"  sufficient = {d1.sufficient}")
        print(f"  reason     = {d1.reason}")
        print(f"  补检关键词  = {d1.new_keywords}")
        assert d1.sufficient is False, "这么单薄的草稿不该被判为足够"
        assert d1.new_keywords, "判定需要补检却没给关键词"

        # 场景二：重复的关键词应当被过滤掉
        print("\n【场景二】补检词里混入已搜过的词（验证去重）")
        d2 = JudgeDecision(sufficient=False, reason="测试", new_keywords=["大模型 API 价格", "价格战"])
        JudgeAgent._sanitize(d2, ["大模型 API 价格"], 3)
        print(f"  过滤后 = {d2.new_keywords}")
        assert "大模型 API 价格" not in d2.new_keywords, "重复关键词没被过滤"
        assert "价格战" in d2.new_keywords

        # 场景三：判为足够时必须清空补检词
        d3 = JudgeDecision(sufficient=True, reason="够了", new_keywords=["多余的词"])
        JudgeAgent._sanitize(d3, [], 3)
        assert d3.new_keywords == [], d3.new_keywords
        print("\n【场景三】sufficient=true 时补检词被清空 → 通过")

        print("\n检查通过。")

    asyncio.run(_demo())
