"""ReportAgent —— 主笔：草稿 + 来源 → 结构化报告 + 自包含 HTML。

对应研究流程的最后一环「综合生成报告」。**两次 LLM 调用**：

    第 1 次（report_agent.jinja2）      → 报告元信息 ReportOutline
                                          （标题 / 摘要 / 关键结论 / 遗留问题）

    中间：**正文分节不走 LLM**，直接由草稿段落映射
          （heading = 当初的关键词，body = 那段总结正文）

    第 2 次（report_html_agent.jinja2） → 把组装好的报告渲染成完整 HTML

为什么正文不让 LLM 重新组织？因为那样就断了"正文 → 搜索关键词 → 来源"
这条追溯链，而且模型重写时容易顺手编造。直接映射最忠实也最省一次调用。
"""

from __future__ import annotations

import json
import logging

from ..models import (
    Conclusion,
    ConclusionDraft,
    Confidence,
    DraftBlock,
    ReportContent,
    ReportOutline,
    Source,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """(主题, 草稿, 来源, 置信度) → (ReportContent, HTML)。两次 LLM 调用，无工具。"""

    name = "report"
    outline_template = "report_agent.jinja2"
    html_template = "report_html_agent.jinja2"
    # 基类的 template 留空：本 agent 有两个模板，调用时显式指定
    template = ""

    async def generate(
        self,
        topic: str,
        draft: list[DraftBlock],
        sources: list[Source],
        confidence: Confidence,
    ) -> tuple[ReportContent, str]:
        """生成结构化报告与它的 HTML 版本。

        返回 (ReportContent, html)。
        **副作用**：会把没有来源的结论清单写进 `confidence.inferences`
        （置信度对象是由引擎传进来的，这里只是补上它算不出来的那一项）。
        """
        # ---- 第 1 次调用：报告元信息 ----
        outline = await self._outline(topic, draft, sources)

        # ---- 正文分节：草稿直接映射，不经 LLM ----
        sections = [DraftBlock(heading=b.heading, body=b.body) for b in draft]

        # ---- 关键结论：把 URL 还原成来源对象 ----
        conclusions = self._resolve_conclusions(outline.key_conclusions, sources)

        # ---- 置信度补一项：哪些结论没有来源（产物④的"模型推断"标注）----
        confidence.inferences = [c.text for c in conclusions if c.inferred]
        if confidence.inferences:
            logger.info(
                "[report] %d 条结论没有来源支撑，已标注为「模型推断」", len(confidence.inferences)
            )

        content = ReportContent(
            title=outline.title.strip() or f"关于「{topic}」的研究报告",
            topic=topic,
            summary=outline.summary.strip(),
            sections=sections,
            key_conclusions=conclusions,
            open_questions=[q.strip() for q in outline.open_questions if q and q.strip()],
        )

        logger.info(
            "[report] 组装完成：%d 节正文 / %d 条关键结论 / %d 个遗留问题",
            len(content.sections), len(content.key_conclusions), len(content.open_questions),
        )

        # ---- 第 2 次调用：渲染 HTML ----
        html = await self._render_html(content, confidence)

        return content, html

    # ------------------------------------------------------------------
    # 第 1 次调用
    # ------------------------------------------------------------------
    async def _outline(
        self, topic: str, draft: list[DraftBlock], sources: list[Source]
    ) -> ReportOutline:
        """调主笔拿报告元信息。"""
        # 正文用"第 N 节：标题 + 段落"的形式给出，让模型看清都有哪些内容
        sections_text = "\n\n".join(
            f"第 {i} 节 · {block.heading}\n{block.body}" for i, block in enumerate(draft, 1)
        )

        return await self.call_json(
            template=self.outline_template,
            system_vars={
                "topic": topic,
                "section_count": len(draft),
                "sections_text": sections_text or "（没有正文）",
                "source_count": len(sources),
                "sources_text": self.format_sources(sources),
            },
            user_input="请为这份研究写报告元信息，输出 JSON。",
            output_cls=ReportOutline,
        )

    # ------------------------------------------------------------------
    # 关键结论：URL → Source 对象
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_conclusions(
        drafts: list[ConclusionDraft], sources: list[Source]
    ) -> list[Conclusion]:
        """把主笔给的 URL 字符串，还原成完整的来源对象。

        关键点：**模型给的 URL 如果在真实来源列表里找不到，就丢弃它**。
        模型有可能顺手编一个看起来很像的 URL——留着它等于伪造来源，
        宁可丢掉并记一条 warning。

        还原失败（或模型压根没给来源）的结论会被标成 inferred=True，
        也就是 README 要求的「模型推断」。
        """
        # 建索引：URL 原文 → Source
        by_url = {s.url.strip(): s for s in sources}
        # 再来一份"去掉结尾斜杠"的，模型经常会漏掉或多加一个 /
        by_normalized = {s.url.strip().rstrip("/").casefold(): s for s in sources}

        resolved: list[Conclusion] = []
        unmatched_total = 0

        for draft in drafts:
            matched: list[Source] = []
            matched_urls: set[str] = set()

            for raw_url in draft.sources:
                url = (raw_url or "").strip().rstrip(".,;)")
                if not url:
                    continue
                source = by_url.get(url) or by_normalized.get(url.rstrip("/").casefold())
                if source is None:
                    unmatched_total += 1
                    logger.warning("[report] 结论引用了不在来源列表里的 URL，已丢弃：%s", url)
                    continue
                if source.url in matched_urls:
                    continue
                matched_urls.add(source.url)
                matched.append(source)

            text = (draft.text or "").strip()
            if not text:
                continue

            resolved.append(
                Conclusion(text=text, sources=matched, inferred=not matched)
            )

        if unmatched_total:
            logger.warning("[report] 共丢弃 %d 个无法核实的引用 URL", unmatched_total)
        return resolved

    # ------------------------------------------------------------------
    # 第 2 次调用
    # ------------------------------------------------------------------
    async def _render_html(self, content: ReportContent, confidence: Confidence) -> str:
        """把组装好的报告渲染成完整自包含 HTML。

        这一次的输出是**HTML 原文，不包 JSON**——HTML 里有大量引号和尖括号，
        塞进 JSON 要来回转义，极易出错。
        """
        html = await self._run(
            system_vars={
                "report_json": content.model_dump_json(indent=2),
                "confidence_json": confidence.model_dump_json(indent=2),
            },
            user_input="请把这份报告渲染成完整的自包含 HTML 页面。",
            template=self.html_template,
        )

        html = self.strip_code_fence(html)

        # 粗检：不像 HTML 就记一条 warning（不中断，报告数据本身还是好的）
        lowered = html.lstrip().lower()
        if not lowered.startswith("<!doctype") and not lowered.startswith("<html"):
            logger.warning(
                "[report] HTML 输出开头不是 <!DOCTYPE/<html，可能没按要求渲染：%s...",
                html[:80],
            )
        if "<html" not in lowered:
            logger.warning("[report] HTML 输出里找不到 <html> 标签，页面可能不完整")
        if "<script" in lowered:
            logger.warning("[report] HTML 里出现了 <script>，模板要求不使用 JavaScript")

        logger.info("[report] HTML 渲染完成，共 %d 字符", len(html))
        return html


if __name__ == "__main__":
    # 需要网络与密钥，人工验证用：python3 -m backend.agent.report
    import asyncio
    from pathlib import Path

    from .. import config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _demo() -> None:
        topic = "天空为什么是蓝色的"
        draft = [
            DraftBlock(
                heading="瑞利散射",
                body="太阳光进入大气层后会被空气分子散射。其中波长较短的蓝光散射得最厉害"
                "[1]，因此从各个方向看天空都是蓝的。日落时阳光穿过的大气更厚，蓝光被散射殆尽，"
                "剩下红光，所以晚霞是红的[2]。",
            ),
            DraftBlock(
                heading="人眼感知",
                body="人眼对蓝光比对紫光更敏感，所以尽管紫光散射更强，我们看到的天空仍是蓝色[3]。",
            ),
        ]
        sources = [
            Source(url="https://example.com/rayleigh", title="瑞利散射原理", site_name="示例科普"),
            Source(url="https://example.com/sunset", title="为什么晚霞是红的", site_name="示例科普"),
            Source(url="https://example.com/eye", title="人眼的色彩感知", site_name="另一个站点"),
        ]
        confidence = Confidence(
            level="中",
            score=0.55,
            reason="来源数量中等，站点较集中",
            cutoff_date="2026-09-10",
        )

        agent = ReportAgent()
        content, html = await agent.generate(topic, draft, sources, confidence)

        print("\n" + "=" * 60)
        print(f"标题     : {content.title}")
        print(f"摘要     : {content.summary[:100]}...")
        print(f"正文节数 : {len(content.sections)}")
        print(f"关键结论 : {len(content.key_conclusions)} 条")
        for c in content.key_conclusions:
            flag = "【模型推断】" if c.inferred else f"（{len(c.sources)} 个来源）"
            print(f"    - {c.text[:50]} {flag}")
        print(f"遗留问题 : {content.open_questions}")
        print(f"模型推断 : {confidence.inferences}")
        print(f"HTML     : {len(html)} 字符")
        print("=" * 60)

        assert content.sections, "正文分节为空"
        assert html.strip(), "HTML 为空"
        assert "<html" in html.lower(), "HTML 里没有 <html>"

        out = config.DATA_DIR.parent / "report_demo.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(html, encoding="utf-8")
        print(f"\nHTML 已写出：{out}（可直接浏览器打开看效果）")

    asyncio.run(_demo())
