"""FastAPI 接口层 —— HTTP 入口。

四个接口：

    GET  /health                   健康检查
    POST /api/research             发起研究，**立即**返回 202 + research_id
    GET  /api/research             列出全部研究记录（轻量摘要）
    GET  /api/research/{id}        查单条研究（含四类产物，可轮询看进度）
    GET  /api/research/{id}/html   直接看报告页面（自包含 HTML）

**为什么 POST 是"立即返回"而不是等研究跑完？**
一次深度研究要跑十几轮 LLM 调用和搜索，耗时几分钟。
HTTP 请求等几分钟早就超时了。所以这里用「提交任务 + 轮询查结果」：

    1. POST 建记录，状态 running，返回 202 和 research_id
    2. 研究在 asyncio 后台任务里跑，每完成一步就把中间结果写盘
    3. 前端每隔一两秒 GET 一次，就能看着步骤一条条长出来

状态机：pending → running → completed / failed
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import config, research, storage
from .models import (
    ResearchCreated,
    ResearchRecord,
    ResearchRequest,
    ResearchSummary,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭时执行一次。

    在这里统一打开日志输出（uvicorn 自带的 logger 不重复配置），
    并把落盘目录建好——否则第一次 POST 时才知道目录不存在。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage.ensure_dir()

    logger.info("=" * 60)
    logger.info("深度研究助手后端启动")
    logger.info(config.describe())
    logger.info("=" * 60)

    yield

    logger.info("深度研究助手后端已停止")


app = FastAPI(
    title="深度研究助手",
    description="输入一个研究主题，自动检索、迭代、综合，产出带来源引用的研究报告。",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
@app.get("/health", summary="健康检查")
async def health() -> dict:
    """给部署/探活用，顺便回显当前模型与轮数配置，方便确认服务读的是哪份 .env。"""
    return {
        "status": "ok",
        "model": config.DEEPSEEK_MODEL,
        "max_rounds": config.RESEARCH_MAX_ROUNDS,
        "deepseek_key": config.key_origin("DEEPSEEK_API_KEY"),
        "bocha_key": config.key_origin("BOCHA_API_KEY"),
    }


# ---------------------------------------------------------------------------
# 发起研究
# ---------------------------------------------------------------------------
@app.post(
    "/api/research",
    response_model=ResearchCreated,
    status_code=202,
    summary="发起一次研究（立即返回，后台执行）",
)
async def create_research(payload: ResearchRequest) -> ResearchCreated:
    """建任务并起后台执行，立刻返回 research_id。

    返回 202 Accepted 而不是 200：任务只是**被接受**了，还没执行完。
    """
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="研究主题不能为空")

    record = await research.start(topic)
    logger.info("已受理研究请求 %s：%s", record.id, topic)

    return ResearchCreated(
        research_id=record.id,
        status=record.status,
        topic=record.topic,
    )


# ---------------------------------------------------------------------------
# 列表 / 详情
# ---------------------------------------------------------------------------
@app.get(
    "/api/research",
    response_model=list[ResearchSummary],
    summary="列出全部研究记录",
)
async def list_research() -> list[ResearchSummary]:
    """返回轻量摘要（不含报告正文），按创建时间从新到旧。"""
    return [record.to_summary() for record in storage.list_records()]


@app.get(
    "/api/research/{research_id}",
    response_model=ResearchRecord,
    summary="查询单条研究（可轮询看进度）",
)
async def get_research(research_id: str) -> ResearchRecord:
    """按 ID 查一条研究记录。

    - 研究进行中时，返回的 process.steps / draft / sources 会不断变长
    - 研究完成后，report / report_html / sources / process / confidence 四类产物齐全
    - 失败时 status=failed，error 里有原因，**已写下的中间结果仍然保留**
    """
    record = storage.load(research_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"研究记录不存在：{research_id}")
    return record


@app.get(
    "/api/research/{research_id}/html",
    response_class=HTMLResponse,
    summary="直接查看报告页面",
)
async def get_research_html(research_id: str) -> HTMLResponse:
    """把报告的自包含 HTML 直接吐出来，浏览器打开就能看。

    这一步不在 README 的接口要求里，纯粹是方便——
    否则 HTML 只能从 GET /api/research/{id} 的 JSON 里拷出来再看。
    """
    record = storage.load(research_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"研究记录不存在：{research_id}")

    if not record.report_html.strip():
        raise HTTPException(
            status_code=409,
            detail=f"报告还没生成（当前状态：{record.status.value}）",
        )
    return HTMLResponse(content=record.report_html)


if __name__ == "__main__":
    # 打印路由表并启动开发服务器：python3 -m backend.app
    import uvicorn

    print("=" * 60)
    print("深度研究助手 —— 路由表")
    print("=" * 60)
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            for method in sorted(methods):
                if method in ("HEAD", "OPTIONS"):
                    continue
                print(f"  {method:6} {route.path}")
    print("=" * 60)
    print(config.describe())
    print("=" * 60)
    print("启动开发服务器：http://127.0.0.1:8001")
    print("（8000 被占用时用 8001，与 start.sh 的约定一致）")
    print()

    uvicorn.run(app, host="127.0.0.1", port=8001)
