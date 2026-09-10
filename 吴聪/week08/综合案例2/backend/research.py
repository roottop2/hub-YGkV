"""研究任务编排：后台执行 + 中间结果落盘。

这一层夹在 app.py（HTTP）和 engine.py（研究流程）中间，只管"任务管理"：

    POST /api/research → 建一条记录，状态 running，**立即**返回 ID
    asyncio 后台任务    → 跑引擎，每完成一步就把中间结果写盘
    GET  /api/research/{id} → 读到不断变长的记录

**为什么中间结果也要写盘？**
这是 README 的产品要求（研究过程记录），也是"轮询能看到进度"的前提：
前端每隔一两秒 GET 一次，每次读到的 process.steps / draft / sources 都比上次长。

**失败也要保住已有成果**：研究抛异常时，状态改成 failed，
但**之前写进去的中间结果一律保留**——用户至少能看到"搜了哪些词、读了哪些页面"，
而不是面对一条空记录。
"""

from __future__ import annotations

import asyncio
import logging

from . import config, storage
from .engine import DeepResearch
from .models import ResearchRecord, ResearchStatus

logger = logging.getLogger(__name__)

#: 后台任务的强引用集合。
#: asyncio 内部只持任务的**弱引用**，不自己存一份的话，
#: 任务可能在跑完之前就被垃圾回收掉（官方文档明确提醒过这一点）。
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def create_record(topic: str) -> ResearchRecord:
    """建一条待执行的记录并落盘，立刻返回。

    先落盘再返回，是为了让 POST 返回的 ID 马上就能被 GET 到——
    否则前端拿到 ID 立刻轮询可能读不到文件（404）。
    """
    record = ResearchRecord(topic=topic.strip(), status=ResearchStatus.RUNNING)
    storage.save(record)
    logger.info("创建研究任务 %s：%s", record.id, record.topic)
    return record


async def run_research(research_id: str, topic: str) -> None:
    """在后台跑完一次研究，全程更新同一条记录。

    **本函数保证不向外抛异常**——它是 `asyncio.create_task` 起的后台任务，
    抛出去没人接，还会让记录永远停在 running。所有异常都在这里兜住并写进 record.error。
    """
    record = storage.load(research_id)
    if record is None:
        # 记录不见了（被删了？），重新建一条，让结果至少有地方放
        logger.warning("记录 %s 不存在，重新创建", research_id)
        record = ResearchRecord(id=research_id, topic=topic)

    record.topic = topic
    record.status = ResearchStatus.RUNNING
    record.error = ""
    storage.save(record)

    async def on_progress(snapshot: dict) -> None:
        """引擎每完成一步就调这个：把当前中间结果写盘，状态保持 running。

        research.py 传的是**异步**回调（engine 用 inspect.isawaitable 判断），
        所以这里可以放心 await 落盘。
        """
        record.process = snapshot["process"]
        record.draft = snapshot["draft"]
        record.sources = snapshot["sources"]
        record.status = ResearchStatus.RUNNING
        storage.save(record)

    logger.info("开始执行研究任务 %s：%s", research_id, topic)

    try:
        outcome = await DeepResearch(on_progress=on_progress).run(topic)
    except Exception as exc:  # noqa: BLE001 —— 后台任务必须兜住所有异常
        record.status = ResearchStatus.FAILED
        record.error = f"{type(exc).__name__}: {exc}"
        storage.save(record)  # 保留已经写进去的中间结果
        logger.exception("研究任务 %s 失败：%s", research_id, exc)
        return

    # 成功：把四类产物一次性写全
    record.report = outcome.report
    record.report_html = outcome.report_html
    record.sources = outcome.sources
    record.process = outcome.process
    record.confidence = outcome.confidence
    record.draft = outcome.draft
    record.status = ResearchStatus.COMPLETED
    record.error = ""
    storage.save(record)

    logger.info(
        "研究任务 %s 完成：%d 轮 / %d 条来源 / 置信度 %s",
        research_id, outcome.process.iterations, len(outcome.sources), outcome.confidence.level,
    )


async def start(topic: str) -> ResearchRecord:
    """建记录 + 起后台任务，返回刚建的记录（供接口立即响应）。

    注意这里**不 await** 研究本身，只 await 到"任务已排上"。
    调用方拿到 ID 就能返回 202 了。
    """
    record = create_record(topic)
    task = asyncio.create_task(run_research(record.id, record.topic))
    _BACKGROUND_TASKS.add(task)  # 存强引用，防止被 GC 提前回收
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return record


if __name__ == "__main__":
    # 端到端跑一次真实研究并落盘（需要密钥与网络）：python3 -m backend.research
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    TOPIC = "国内新能源汽车换电模式的发展现状"

    async def _demo() -> None:
        print("=" * 60)
        print("研究任务编排自检")
        print("=" * 60)

        record = create_record(TOPIC)
        print(f"已创建任务：{record.id}")
        print(f"落盘文件  ：{config.DATA_DIR / f'{record.id}.json'}")
        print()

        # 模拟前端轮询：一边跑一边打印状态变化
        async def poll(run_task: asyncio.Task) -> None:
            last = ""
            while not run_task.done():
                current = storage.load(record.id)
                if current:
                    line = (
                        f"  [轮询] 状态={current.status.value:9} "
                        f"轮次={current.process.iterations} "
                        f"步骤={len(current.process.steps):2} "
                        f"来源={len(current.sources):3} "
                        f"草稿={len(current.draft)} 节"
                    )
                    if line != last:
                        print(line)
                        last = line
                await asyncio.sleep(2)

        task = asyncio.create_task(run_research(record.id, record.topic))
        await poll(task)

        final = storage.load(record.id)
        assert final is not None, "最终记录读不到"

        print()
        print("=" * 60)
        print(f"最终状态：{final.status.value}")
        print("=" * 60)

        if final.status is ResearchStatus.FAILED:
            print(f"失败原因：{final.error}")
            print("\n（中间结果已保留，可查看上面已记录的步骤）")
            return

        assert final.report is not None, "completed 却没有报告"
        assert final.confidence is not None, "completed 却没有置信度"
        assert final.report_html.strip(), "report_html 为空"

        print(f"报告标题 : {final.report.title}")
        print(f"正文节数 : {len(final.report.sections)}")
        print(f"关键结论 : {len(final.report.key_conclusions)} 条")
        print(f"遗留问题 : {len(final.report.open_questions)} 条")
        print(f"来源数量 : {len(final.sources)} 条")
        print(f"置信度   : {final.confidence.level}（{final.confidence.score}）")
        print(f"过程步骤 : {len(final.process.steps)} 步，共 {final.process.iterations} 轮")
        print(f"HTML     : {len(final.report_html)} 字符")
        print()
        print("四类产物齐全。")

    asyncio.run(_demo())
