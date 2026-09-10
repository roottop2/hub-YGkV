"""研究引擎 DeepResearch —— 全流程的编排器。

**它不是 agent。** 这个类里没有任何 LLM 调用，只有确定性控制流：

    规划（KeywordAgent）
      → 逐关键词 web_search（直接函数调用，不是工具调用）
      → 总结成正文并累积进报告草稿（SummaryAgent）
      → 判断是否补检（JudgeAgent）
          不够 → 拿新关键词进下一轮（最多 max_rounds 轮）
      → 统计来源 + 确定性计算置信度
      → 出报告（ReportAgent：结构化报告 + 自包含 HTML）

**为什么不让 agent 自己调工具？**
因为"研究不能搜一轮就出报告"是产品要求（README：迭代是产品特性）。
把控制流攥在引擎手里，轮数、补检条件、什么时候收尾都是确定的代码，
可预测、可调试、不会出现 agent 空转。子 agent 全都无工具，只负责"想"。

流程中逐步累积三样东西，每完成一步都通过 on_progress 回调交给外面落盘：

    process.steps   每一步做了什么（plan / search / summarize / judge）
    draft           每完成一个关键词追加一段正文
    sources         按 URL 去重的来源列表
"""

from __future__ import annotations

import inspect
import logging
from datetime import date
from typing import Any, Awaitable, Callable

from . import config
from .agent import JudgeAgent, KeywordAgent, ReportAgent, SummaryAgent
from .models import (
    Confidence,
    DraftBlock,
    ProcessStep,
    ResearchOutcome,
    ResearchProcess,
    Source,
)
from .tools import SearchError, domain_of, web_search

logger = logging.getLogger(__name__)

#: 进度回调：可以是同步函数，也可以是 async 函数
ProgressCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class ResearchError(RuntimeError):
    """研究无法继续（例如一个关键词都生成不出来）。"""


class DeepResearch:
    """一次深度研究的完整流程。"""

    def __init__(
        self,
        max_rounds: int | None = None,
        max_keywords: int = 5,
        max_new_keywords: int = 3,
        on_progress: ProgressCallback | None = None,
        keyword_agent: KeywordAgent | None = None,
        summary_agent: SummaryAgent | None = None,
        judge_agent: JudgeAgent | None = None,
        report_agent: ReportAgent | None = None,
    ) -> None:
        self.max_rounds = max_rounds or config.RESEARCH_MAX_ROUNDS
        self.max_keywords = max_keywords
        self.max_new_keywords = max_new_keywords
        self.on_progress = on_progress

        # 允许从外面注入 agent（测试时替换成假的，就不必真的调模型）
        self.keyword_agent = keyword_agent or KeywordAgent()
        self.summary_agent = summary_agent or SummaryAgent()
        self.judge_agent = judge_agent or JudgeAgent()
        self.report_agent = report_agent or ReportAgent()

        # ---- 累积中的状态 ----
        self.process = ResearchProcess()
        self.draft: list[DraftBlock] = []
        self.sources: list[Source] = []
        self._seen_urls: set[str] = set()
        # 本轮"关键词 → 搜索结果"的映射。检索阶段填，总结阶段读。
        # 放在实例上而不是当参数传，是为了让 _search_round / _summarize_round
        # 各自保持单一职责，不必互相传一大坨数据。
        self._round_results: dict[str, list[dict]] = {}

    # ==================================================================
    # 对外入口
    # ==================================================================
    async def run(self, topic: str) -> ResearchOutcome:
        """跑完整个研究，返回四类产物。

        参数 topic 为空时直接抛错——与其让后面每个 agent 各自处理空主题，
        不如在这里一次性拦住。
        """
        topic = (topic or "").strip()
        if not topic:
            raise ResearchError("研究主题不能为空")

        logger.info("=" * 60)
        logger.info("开始研究：%s（最多 %d 轮）", topic, self.max_rounds)

        # ---- 第 0 步：规划 ----
        pending = await self._plan(topic)

        # ---- 第 1..N 轮：检索 → 总结 → 判断是否补检 ----
        for round_no in range(1, self.max_rounds + 1):
            self.process.iterations = round_no
            logger.info("-" * 60)
            logger.info("第 %d/%d 轮，本轮关键词：%s", round_no, self.max_rounds, pending)

            await self._search_round(pending, round_no)
            await self._summarize_round(topic, pending, round_no)

            # 到轮数上限就不再问模型"够不够"了，直接收尾出报告
            if round_no >= self.max_rounds:
                self._add_step("judge", round_no, f"已达轮数上限 {self.max_rounds}，进入报告生成")
                await self._progress()
                break

            pending = await self._judge_round(topic, round_no)
            if pending is None:
                break

        # ---- 收尾：置信度 → 报告 ----
        confidence = self.compute_confidence()
        logger.info(
            "置信度：%s（%.2f）—— %s", confidence.level, confidence.score, confidence.reason
        )

        self._add_step("report", self.process.iterations, "综合草稿与来源，生成结构化报告与 HTML")
        content, html = await self.report_agent.generate(
            topic, self.draft, self.sources, confidence
        )

        logger.info("=" * 60)
        logger.info(
            "研究完成：%d 轮 / %d 条来源 / %d 节正文 / %d 条关键结论",
            self.process.iterations, len(self.sources), len(content.sections),
            len(content.key_conclusions),
        )

        return ResearchOutcome(
            report=content,
            report_html=html,
            sources=self.sources,
            process=self.process,
            confidence=confidence,
            draft=self.draft,
        )

    # ==================================================================
    # 各阶段
    # ==================================================================
    async def _plan(self, topic: str) -> list[str]:
        """规划：把主题拆成一批初始关键词。"""
        logger.info("【规划】拆解研究主题……")
        keywords = await self.keyword_agent.generate_keywords(topic, max_keywords=self.max_keywords)

        if not keywords:
            raise ResearchError("规划阶段没有生成任何关键词，无法开始检索")

        self.process.plan = list(keywords)
        self.process.search_queries.extend(keywords)
        self._add_step("plan", 0, f"拆解出 {len(keywords)} 个初始关键词：{'、'.join(keywords)}")
        await self._progress()
        return keywords

    async def _search_round(self, keywords: list[str], round_no: int) -> None:
        """检索：逐个关键词搜，结果去重后并进来源列表。

        单个关键词搜索失败**不会中断研究**——记一笔日志和过程记录就继续，
        毕竟少一个角度的资料，总好过整次研究白跑。

        搜索结果存进 self._round_results，供紧接着的总结阶段取用。
        """
        # 每轮开一张新表：上一轮的结果已经总结完了
        self._round_results = {}

        for keyword in keywords:
            try:
                results = await web_search(keyword)
            except SearchError as exc:
                logger.warning("关键词「%s」搜索失败：%s", keyword, exc)
                self._add_step("search", round_no, f"「{keyword}」搜索失败：{exc}")
                self._round_results[keyword] = []
                await self._progress()
                continue

            self._round_results[keyword] = results

            # 按 URL 去重：不同关键词常常搜到同一个网页，只收第一次
            added = 0
            for item in results:
                url = (item.get("url") or "").strip()
                if not url or url in self._seen_urls:
                    continue
                self._seen_urls.add(url)
                source = Source(
                    url=url,
                    title=(item.get("title") or "").strip(),
                    site_name=(item.get("site_name") or "").strip(),
                    date=(item.get("date") or "").strip(),
                    snippet=(item.get("snippet") or "").strip(),
                )
                self.sources.append(source)
                self.process.reviewed_urls.append(url)
                added += 1

            logger.info("「%s」拿到 %d 条结果，其中 %d 条是新来源", keyword, len(results), added)
            self._add_step(
                "search", round_no, f"「{keyword}」检索到 {len(results)} 条，新增来源 {added} 条"
            )
            await self._progress()

    async def _summarize_round(self, topic: str, keywords: list[str], round_no: int) -> None:
        """阅读抽取：把每个关键词的搜索结果写成一段正文，追加进草稿。

        总结失败也**不中断**——少一节正文，总好过整个研究失败。
        """
        for keyword in keywords:
            results = self._round_results.get(keyword, [])
            if not results:
                self._add_step("summarize", round_no, f"「{keyword}」没有结果，跳过总结")
                continue

            try:
                body = await self.summary_agent.summarize(topic, keyword, results)
            except Exception as exc:  # noqa: BLE001 —— 单个关键词失败不该拖垮整次研究
                logger.warning("关键词「%s」总结失败：%s", keyword, exc)
                self._add_step("summarize", round_no, f"「{keyword}」总结失败：{exc}")
                await self._progress()
                continue

            self.draft.append(DraftBlock(heading=keyword, body=body))
            self._add_step("summarize", round_no, f"「{keyword}」总结出 {len(body)} 字正文")
            await self._progress()

    async def _judge_round(self, topic: str, round_no: int) -> list[str] | None:
        """判断是否需要补检。

        返回：
            新一批关键词，表示要继续下一轮；
            None，表示材料已足够（或补不出新词了），应当收尾出报告。
        """
        decision = await self.judge_agent.judge(
            topic=topic,
            draft_text=self._draft_text(),
            sources=self.sources,
            searched=self.process.search_queries,
            round_no=round_no,
            max_rounds=self.max_rounds,
            max_new_keywords=self.max_new_keywords,
        )

        if decision.sufficient:
            self._add_step("judge", round_no, f"材料已足够：{decision.reason}")
            await self._progress()
            return None

        pending = decision.new_keywords
        if not pending:
            # 判为不足却提不出新关键词——再搜也是重复，收尾
            logger.info("判为需要补检但没有给出新的关键词，转入报告生成")
            self._add_step("judge", round_no, "判为需要补检但无新关键词，转入报告生成")
            await self._progress()
            return None

        self.process.search_queries.extend(pending)
        self._add_step(
            "judge", round_no, f"材料不足（{decision.reason}），补检：{'、'.join(pending)}"
        )
        await self._progress()
        return pending

    # ==================================================================
    # 置信度（确定性计算，不让 LLM 自评）
    # ==================================================================
    def compute_confidence(self) -> Confidence:
        """由三个客观指标算出置信度。

        **为什么不问模型"你有多确定"？**
        让模型给自己的答案打分，结果基本不可靠——它倾向于说"很有信心"。
        这里改成数客观事实：搜到多少来源、分散在多少个站点、正文有多厚。

        三个指标各占权重：来源数量 40% / 站点多样性 35% / 正文厚度 25%。

        每项用 `x / (x + 常数)` 的**软饱和曲线**归一化到 0~1：

            来源数   n / (n + 20)      →  20 条 = 0.50，100 条 = 0.83
            站点数   d / (d + 10)      →  10 个 = 0.50， 55 个 = 0.85
            正文     c / (c + 3000)    → 3000 字 = 0.50，5737 字 = 0.66

        为什么不直接用 `min(x / 阈值, 1)` 的硬封顶？
        **那样会丢掉区分度。** 第一版就是这么写的：一次实测搜到 100 条来源，
        三个指标全部触顶，总分恒等于上限——"搜了 100 条的"和"刚好达标的"
        给出同一个分数，这个指标就废了。
        软曲线单调递增且永远不封顶，资料越多分数越高，但增长越来越慢。

        **总分再乘 0.9，即上限 0.90，永远不给满分。**
        一个由搜索摘要拼出来的报告，不该声称"绝对确定"。
        """
        source_count = len(self.sources)
        domains = {domain_of(s.url) for s in self.sources if domain_of(s.url)}
        domain_count = len(domains)
        body_chars = sum(len(b.body) for b in self.draft)

        def saturate(x: int, half: int) -> float:
            """软饱和：x=half 时得 0.5，x 越大越接近 1 但永不到 1。"""
            return x / (x + half)

        n_score = saturate(source_count, 20)
        d_score = saturate(domain_count, 10)
        c_score = saturate(body_chars, 3000)

        # 0.9 是刻意留的天花板
        score = round(0.9 * (0.40 * n_score + 0.35 * d_score + 0.25 * c_score), 2)

        if score >= 0.65:
            level = "高"
        elif score >= 0.35:
            level = "中"
        else:
            level = "低"

        reason = (
            f"共 {source_count} 条来源，来自 {domain_count} 个站点，"
            f"正文 {body_chars} 字，检索 {self.process.iterations} 轮。"
        )
        if source_count == 0:
            reason += "没有任何来源支撑，全部结论均为模型推断，请谨慎采信。"
        elif domain_count <= 2:
            reason += "来源站点较集中，可能存在单一视角，建议交叉验证。"
        elif score >= 0.65:
            reason += "来源数量与站点覆盖面较好。"
        else:
            reason += "资料量偏少，结论仅作参考。"

        return Confidence(
            level=level,
            score=score,
            reason=reason,
            # 信息截止时间就是本次研究完成的时间——再晚的信息这次没搜到
            cutoff_date=date.today().isoformat(),
            inferences=[],  # 由 ReportAgent 在组装结论时填入
        )

    # ==================================================================
    # 内部小工具
    # ==================================================================
    def _draft_text(self) -> str:
        """把草稿拼成一段文本，给审核 agent 看。"""
        if not self.draft:
            return ""
        return "\n\n".join(f"【{b.heading}】\n{b.body}" for b in self.draft)

    def _add_step(self, step: str, round_no: int, detail: str) -> None:
        """往过程记录里追加一步。"""
        self.process.steps.append(ProcessStep(step=step, round=round_no, detail=detail))

    async def _progress(self) -> None:
        """把当前中间结果交给回调（同步/异步函数都支持）。

        research.py 用它把中间结果写盘，这样前端轮询就能实时看到过程；
        demo 里传个同步的打印函数也行。判断依据是返回值是不是 awaitable。
        """
        if self.on_progress is None:
            return

        snapshot = {
            "process": self.process,
            "draft": self.draft,
            "sources": self.sources,
        }
        result = self.on_progress(snapshot)
        if inspect.isawaitable(result):
            await result


if __name__ == "__main__":
    # 两部分：
    #   第一部分 —— 离线控制流自检（假 agent + 假搜索，不需要网络）
    #   第二部分 —— 端到端跑一次真实研究（需要密钥与网络）
    import asyncio
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # ==================================================================
    # 第一部分：离线控制流自检
    # ==================================================================
    # 「迭代补检」是这个产品的核心特性（README：迭代是产品特性）。
    # 真实运行能不能走到第二轮要看运气——第一次实测时审核员在第一轮
    # 就判了"材料已足够"，补检分支根本没被执行到。
    # 所以这里用假 agent 把控制流**确定性地**跑一遍。
    def _offline_control_flow_check() -> None:
        from .models import Conclusion, ReportContent

        class FakeKeywordAgent:
            async def generate_keywords(self, topic, max_keywords=5):
                return ["关键词A", "关键词B"]

        class FakeSummaryAgent:
            def __init__(self):
                self.calls = []

            async def summarize(self, topic, keyword, results):
                self.calls.append(keyword)
                return f"{keyword} 的正文，共 {len(results)} 条结果。"

        class FakeJudgeAgent:
            """第一轮说不够（补检"关键词C"），第二轮说够了。"""

            def __init__(self):
                self.rounds = []

            async def judge(self, topic, draft_text, sources, searched, **kwargs):
                from .models import JudgeDecision

                self.rounds.append({
                    "round_no": kwargs.get("round_no"),
                    "searched": list(searched),
                    "draft_chars": len(draft_text),
                })
                if kwargs.get("round_no") == 1:
                    return JudgeDecision(
                        sufficient=False, reason="还缺一块", new_keywords=["关键词C"]
                    )
                return JudgeDecision(sufficient=True, reason="够了", new_keywords=[])

        class FakeReportAgent:
            async def generate(self, topic, draft, sources, confidence):
                confidence.inferences = ["一条没有来源的推断"]
                return (
                    ReportContent(
                        title="测试报告",
                        topic=topic,
                        summary="摘要",
                        sections=list(draft),
                        key_conclusions=[
                            Conclusion(text="结论", sources=[], inferred=True)
                        ],
                        open_questions=["遗留"],
                    ),
                    "<!DOCTYPE html><html><body>测试</body></html>",
                )

        async def fake_search(query, count=None, summary=True):
            """假搜索。

            - 关键词C 故意抛错，验证"单个关键词失败不中断研究"
            - 每个关键词返回一条自己独有的 URL + **一条所有关键词共有的 URL**
              （真实搜索就是这样：不同关键词经常搜到同一个网页）。
              共有那条用来验证来源按 URL 去重。
            """
            if query == "关键词C":
                raise SearchError("模拟的搜索失败")
            return [
                {
                    "title": f"{query} 的专属文章",
                    "url": f"https://unique-{query}.com/a",
                    "snippet": "摘要",
                    "site_name": f"站点-{query}",
                    "date": "2026-01-01",
                },
                {
                    "title": "所有关键词都会搜到的同一篇",
                    "url": "https://shared.com/common",
                    "snippet": "摘要",
                    "site_name": "共享站点",
                    "date": "2026-01-01",
                },
            ]

        # 把模块里的 web_search 换成假的（引擎是直接调用它的）
        global web_search
        real_search = web_search
        web_search = fake_search
        try:
            print("=" * 60)
            print("【第一部分】离线控制流自检（假 agent + 假搜索）")
            print("=" * 60)

            summary_agent = FakeSummaryAgent()
            judge_agent = FakeJudgeAgent()

            engine = DeepResearch(
                max_rounds=3,
                keyword_agent=FakeKeywordAgent(),
                summary_agent=summary_agent,
                judge_agent=judge_agent,
                report_agent=FakeReportAgent(),
            )
            outcome = asyncio.run(engine.run("测试主题"))

            # --- 1. 迭代确实跑了两轮 ---
            assert outcome.process.iterations == 2, outcome.process.iterations
            print(f"  迭代轮数           : {outcome.process.iterations}（第一轮判不足 → 补检 → 第二轮判足够）")

            # --- 2. 检索关键词 = 初始两个 + 补检一个 ---
            assert outcome.process.search_queries == ["关键词A", "关键词B", "关键词C"], \
                outcome.process.search_queries
            print(f"  检索关键词         : {outcome.process.search_queries}")

            # --- 3. 审核员看到的"已搜关键词"要随轮次增长 ---
            assert judge_agent.rounds[0]["searched"] == ["关键词A", "关键词B"], judge_agent.rounds[0]
            assert judge_agent.rounds[1]["searched"] == ["关键词A", "关键词B", "关键词C"], judge_agent.rounds[1]
            print(f"  第1轮审核看到      : {judge_agent.rounds[0]['searched']}")
            print(f"  第2轮审核看到      : {judge_agent.rounds[1]['searched']}")

            # --- 4. 单个关键词搜索失败，研究仍继续 ---
            failed = [s for s in outcome.process.steps if "搜索失败" in s.detail]
            assert failed, "搜索失败没有被记录下来"
            print(f"  搜索失败被容忍     : {failed[0].detail}")

            # --- 5. 失败的关键词没有产出正文（不塞空节） ---
            headings = [b.heading for b in outcome.draft]
            assert "关键词C" not in headings, headings
            assert headings == ["关键词A", "关键词B"], headings
            print(f"  草稿分节           : {headings}（失败的关键词不塞空节）")

            # --- 6. 来源按 URL 去重 ---
            # 关键词A、B 各返回 2 条，其中 "https://shared.com/common" 是共有的，
            # 应当只算一次：2 + 2 - 1 = 3 条（关键词C 搜索失败，一条都没加）
            urls = [s.url for s in outcome.sources]
            assert len(urls) == 3, urls
            assert len(set(urls)) == len(urls), f"有重复 URL：{urls}"
            assert urls.count("https://shared.com/common") == 1, urls
            print(f"  来源去重           : 4 条结果去重后剩 {len(urls)} 条（共有 URL 只留一次）")

            # --- 7. 过程记录四种步骤齐全 ---
            kinds = {s.step for s in outcome.process.steps}
            assert {"plan", "search", "summarize", "judge", "report"} <= kinds, kinds
            print(f"  过程步骤类型       : {sorted(kinds)}")

            # --- 8. 置信度被正确计算（3 条来源 / 3 个站点）---
            conf = outcome.confidence
            print(f"  置信度             : {conf.level}（{conf.score}）")
            assert 0.0 < conf.score < 1.0, f"分数不该触底或触顶：{conf.score}"

            # --- 9. 空来源时置信度必须是"低" ---
            empty_engine = DeepResearch()
            conf0 = empty_engine.compute_confidence()
            assert conf0.level == "低" and conf0.score == 0.0, conf0
            assert "模型推断" in conf0.reason
            print(f"  无来源时的置信度   : {conf0.level}（{conf0.score}）——{conf0.reason[:40]}...")

            # --- 10. 置信度必须随资料量单调递增（硬封顶会破坏这条性质）---
            # 第一版用 min(x/阈值, 1) 硬封顶，100 条来源和 20 条来源都得出
            # 同一个数，"搜得多"和"刚够"分不出来，指标就失去意义了。
            def _score_for(n_sources: int, n_domains: int) -> float:
                eng = DeepResearch()
                eng.sources = [
                    Source(url=f"https://site{i % n_domains}.com/a{i}", title="t")
                    for i in range(n_sources)
                ]
                return eng.compute_confidence().score

            samples = [(5, 3), (20, 10), (50, 25), (100, 50), (400, 200)]
            scores = [_score_for(n, d) for n, d in samples]
            readable = " < ".join(f"{s}" for s in scores)
            print(f"  资料量 → 置信分    : {readable}")

            for i in range(len(scores) - 1):
                assert scores[i] < scores[i + 1], (
                    f"资料更多时置信分没有变大（{samples[i]}→{scores[i]}，"
                    f"{samples[i+1]}→{scores[i+1]}），说明又出现封顶了"
                )
            assert scores[-1] < 0.9, f"再多的资料也不该到满分，实际 {scores[-1]}"
            print("  单调递增且永不封顶 : 通过")

            # --- 11. max_rounds 上限兜底：审核员永远说"不够"也不会死循环 ---
            class AlwaysThirstyJudge:
                """每轮都判不足、每次都提新关键词——最坏情况的审核员。"""

                def __init__(self):
                    self.calls = 0

                async def judge(self, topic, draft_text, sources, searched, **kwargs):
                    from .models import JudgeDecision

                    self.calls += 1
                    return JudgeDecision(
                        sufficient=False,
                        reason="永远觉得不够",
                        new_keywords=[f"补检词{self.calls}"],
                    )

            thirsty = AlwaysThirstyJudge()
            capped = DeepResearch(
                max_rounds=2,
                keyword_agent=FakeKeywordAgent(),
                summary_agent=FakeSummaryAgent(),
                judge_agent=thirsty,
                report_agent=FakeReportAgent(),
            )
            capped_outcome = asyncio.run(capped.run("测试主题"))

            assert capped_outcome.process.iterations == 2, capped_outcome.process.iterations
            # 轮数到上限那轮不再问审核员，所以只被调用 1 次（第 1 轮之后）
            assert thirsty.calls == 1, f"到上限后不该再问审核员，实际问了 {thirsty.calls} 次"
            print(
                f"  轮数上限兜底       : max_rounds=2 → 跑满 2 轮收手，"
                f"审核员只被问 {thirsty.calls} 次（第 2 轮直接进报告）"
            )

            print()
            print("  离线控制流自检通过。")
            print()
        finally:
            web_search = real_search

    _offline_control_flow_check()

    # ==================================================================
    # 第二部分：端到端真实研究
    # ==================================================================
    TOPIC = "2026 年国内大模型 API 的价格趋势"

    def on_progress(snapshot: dict) -> None:
        """同步回调：打印当前进度。"""
        process = snapshot["process"]
        print(
            f"    [进度] 轮次={process.iterations}  "
            f"步骤={len(process.steps)}  来源={len(snapshot['sources'])}  "
            f"草稿={len(snapshot['draft'])} 节"
        )

    async def _demo() -> None:
        print("=" * 60)
        print(f"研究主题：{TOPIC}")
        print("=" * 60)

        engine = DeepResearch(on_progress=on_progress)
        outcome = await engine.run(TOPIC)

        print()
        print("=" * 60)
        print("研究完成")
        print("=" * 60)
        print(f"报告标题   : {outcome.report.title}")
        print(f"摘要       : {outcome.report.summary[:120]}...")
        print(f"正文节数   : {len(outcome.report.sections)}")
        print(f"关键结论   : {len(outcome.report.key_conclusions)} 条")
        for c in outcome.report.key_conclusions:
            flag = "【模型推断】" if c.inferred else f"（{len(c.sources)} 个来源）"
            print(f"    - {c.text[:46]}  {flag}")
        print(f"遗留问题   : {outcome.report.open_questions}")
        print(f"来源总数   : {len(outcome.sources)} 条")
        print(f"置信度     : {outcome.confidence.level}（{outcome.confidence.score}）")
        print(f"              {outcome.confidence.reason}")
        print(f"截止时间   : {outcome.confidence.cutoff_date}")
        print(f"模型推断   : {len(outcome.confidence.inferences)} 条")
        print()
        print("研究过程：")
        for step in outcome.process.steps:
            print(f"    [第{step.round}轮/{step.step:9}] {step.detail}")
        print()
        print(f"检索关键词 : {outcome.process.search_queries}")
        print(f"已读页面   : {len(outcome.process.reviewed_urls)} 个 URL")

        # 落盘一份，方便直接打开看报告效果
        out_dir = config.DATA_DIR.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        html_path = out_dir / f"engine_demo_{stamp}.html"
        json_path = out_dir / f"engine_demo_{stamp}.json"
        html_path.write_text(outcome.report_html, encoding="utf-8")
        json_path.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
        print()
        print(f"HTML 报告已写出：{html_path}")
        print(f"完整数据已写出：{json_path}")

        # --- 基本断言 ---
        assert outcome.report.sections, "报告没有正文分节"
        assert outcome.report_html.strip(), "HTML 为空"
        assert outcome.process.iterations >= 1, "轮数至少为 1"
        assert outcome.process.steps, "过程记录为空"
        assert outcome.sources, "一条来源都没有"
        assert outcome.process.plan, "规划记录为空"
        print("\n全部断言通过。")

    asyncio.run(_demo())
