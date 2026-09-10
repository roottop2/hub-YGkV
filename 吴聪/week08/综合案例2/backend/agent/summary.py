"""SummaryAgent —— 资料整理员：把一批搜索结果写成一段正文。

对应研究流程的「阅读抽取」。每个关键词调一次，产物是一段正文，
引擎把它作为一节（DraftBlock）追加进报告草稿。

**本 agent 是全项目唯一的特例：输出纯文字，不走 JSON 解析。**
它的产物本来就是报告正文段落（一个字符串），没有结构化字段，
硬塞进 JSON 反而要处理换行转义，容易出错。
"""

from __future__ import annotations

import logging

from .base import BaseAgent, EmptyOutputError

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """(主题, 关键词, 搜索结果) → 一段正文。一次 LLM 调用，无工具。"""

    name = "summary"
    template = "summary_agent.jinja2"

    async def summarize(self, topic: str, keyword: str, results: list[dict]) -> str:
        """把某个关键词的搜索结果总结成一段正文。

        参数 results 是 tools.web_search 的返回值，
        [{"title","url","snippet","site_name","date"}, ...]
        """
        if not results:
            # 没搜到结果就不浪费一次 LLM 调用，直接给一句说明性正文
            logger.info("[summary] 关键词「%s」没有搜索结果，跳过总结", keyword)
            return f"围绕「{keyword}」的检索没有返回有效结果，本节暂无内容可整理。"

        text = await self._run(
            system_vars={
                "topic": topic,
                "keyword": keyword,
                "source_count": len(results),
                "results_text": self.format_search_results(results),
            },
            user_input=f"请把「{keyword}」的搜索结果整理成一段正文。",
        )

        # 兜底：模板已要求"不要代码块"，但模型偶尔还是会包 ```，这里剥掉
        body = self.strip_code_fence(text)
        if not body:
            raise EmptyOutputError(f"[summary] 关键词「{keyword}」总结结果为空")

        logger.info("[summary] 关键词「%s」总结出 %d 字正文", keyword, len(body))
        return body


if __name__ == "__main__":
    # 需要网络与密钥，人工验证用：python3 -m backend.agent.summary
    import asyncio

    from ..tools import web_search

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _demo() -> None:
        topic = "天空为什么是蓝色的"
        keyword = "瑞利散射"

        results = await web_search(keyword, count=5)
        print(f"先搜到 {len(results)} 条结果，交给总结 agent。\n")

        agent = SummaryAgent()
        body = await agent.summarize(topic, keyword, results)

        print("-" * 60)
        print(body)
        print("-" * 60)
        print(f"\n共 {len(body)} 字。")
        assert body.strip(), "总结结果为空"
        assert "```" not in body, "输出里残留了代码块围栏"

        # 空结果的分支也要能跑
        empty = await agent.summarize(topic, keyword, [])
        print(f"\n空结果分支：{empty}")
        assert empty, "空结果分支不该返回空串"

        print("\n检查通过。")

    asyncio.run(_demo())
