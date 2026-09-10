"""全部数据模型（pydantic）。

约定：**整个项目里所有 pydantic 模型都只在本文件定义**。
agent/ 包与 templates/ 里都不允许再定义模型，需要输出结构时从这里 import。

模型分三类，按数据流动的方向排列：

    ① agent 的输出结构（LLM 直接吐出来的 JSON）
         KeywordOutput / JudgeDecision / ConclusionDraft / ReportOutline

    ② 业务产物（README 要求交付的四类东西）
         ReportContent   —— 结构化研究报告
         Source          —— 来源列表
         ResearchProcess —— 研究过程记录
         Confidence      —— 置信度说明

    ③ 接口与落盘（HTTP 层 + 磁盘上那份 JSON）
         ResearchStatus / ResearchRecord / ResearchRequest / ResearchCreated

为什么要用 pydantic？
    它做两件事：① 校验数据（类型不对就报错，而不是悄悄算错）；
    ② 序列化（`record.model_dump_json()` 直接变成 JSON 字符串，不用手写）。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _now() -> str:
    """当前时间的 ISO 字符串。统一走这个函数，格式才不会各处不一致。"""
    return datetime.now().isoformat(timespec="seconds")


# ===========================================================================
# ① agent 的输出结构
# ===========================================================================


class KeywordOutput(BaseModel):
    """KeywordAgent 的输出：一组搜索关键词。"""

    keywords: list[str] = Field(default_factory=list, description="要拿去搜索的关键词")

    @field_validator("keywords", mode="before")
    @classmethod
    def _unwrap(cls, v: Any) -> Any:
        """容忍模型不老实的情况：把 [{"keyword": "x"}] 这类写法压平成 ["x"]。"""
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # 取这个字典里第一个字符串值
                for value in item.values():
                    if isinstance(value, str):
                        out.append(value)
                        break
        return out


class JudgeDecision(BaseModel):
    """JudgeAgent 的输出：信息够不够？不够的话下一轮搜什么？"""

    sufficient: bool = Field(description="现有草稿是否已能支撑一份完整报告")
    reason: str = Field(default="", description="这么判断的理由")
    new_keywords: list[str] = Field(
        default_factory=list, description="若不够，下一轮要补搜的关键词"
    )

    @field_validator("new_keywords", mode="before")
    @classmethod
    def _coerce_list(cls, v: Any) -> Any:
        """模型有时会把关键词写成用顿号/换行分隔的一整个字符串，这里拆开。"""
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace("、", "\n").replace("，", "\n").split("\n")]
            return [p for p in parts if p]
        return v


class ConclusionDraft(BaseModel):
    """报告元信息里的一条关键结论（LLM 输出，来源只给 URL）。"""

    text: str = Field(description="结论本身，一句话")
    sources: list[str] = Field(
        default_factory=list,
        description="支撑这条结论的来源 URL；留空表示这是模型推断",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_string(cls, data: Any) -> Any:
        """模型可能只写了一句字符串而没给 sources，包一层当作"无来源"。"""
        if isinstance(data, str):
            return {"text": data, "sources": []}
        return data

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class ReportOutline(BaseModel):
    """ReportAgent 第一次调用的输出：报告的"元信息"。

    注意：**正文分节不在这里**。分节由草稿段落直接映射（见 ReportContent.sections），
    不让 LLM 重新组织一遍，这样"每条正文都对应某个搜索关键词"，可追溯且不会丢信息。
    """

    title: str = Field(description="报告标题")
    summary: str = Field(description="执行摘要，一段话")
    key_conclusions: list[ConclusionDraft] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("open_questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


# ===========================================================================
# ② 业务产物
# ===========================================================================


class Source(BaseModel):
    """一条来源。README 要求"每条结论关联 URL / 标题 / 来源"。"""

    url: str
    title: str = ""
    site_name: str = Field(default="", description="站点名，如 知乎 / 36氪")
    date: str = Field(default="", description="页面/文章的发布时间，可能为空")
    snippet: str = Field(default="", description="搜索结果的摘要片段")


class DraftBlock(BaseModel):
    """草稿的一段正文：一个关键词 → 一段总结文字。

    heading 就是当初检索用的关键词，所以"正文哪里来的"一眼可查。
    """

    heading: str
    body: str


class Conclusion(BaseModel):
    """最终报告里的一条关键结论（来源已经从 URL 解析成 Source 对象）。"""

    text: str
    sources: list[Source] = Field(default_factory=list)
    inferred: bool = Field(
        default=False,
        description="True 表示这条结论没有任何来源，属于「模型推断」",
    )


class ReportContent(BaseModel):
    """产物①：结构化研究报告。"""

    title: str
    topic: str = Field(description="原始研究主题")
    summary: str = Field(description="摘要")
    sections: list[DraftBlock] = Field(
        default_factory=list, description="分节正文（heading = 关键词，body = 该关键词的总结）"
    )
    key_conclusions: list[Conclusion] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list, description="遗留问题：这次没研究清楚的部分"
    )


class Confidence(BaseModel):
    """产物④：置信度说明。

    这个值由引擎**确定性计算**得出（数来源个数、数站点个数、看轮到几轮），
    不交给 LLM 自评——让模型给自己的答案打分，结果通常不可靠。
    """

    level: str = Field(description="高 / 中 / 低")
    score: float = Field(ge=0.0, le=1.0, description="0~1 的置信分")
    reason: str = Field(default="", description="为什么是这个等级")
    cutoff_date: str = Field(description="信息截止时间（本次研究的完成时间）")
    inferences: list[str] = Field(
        default_factory=list,
        description="没有来源、被标注为「模型推断」的结论",
    )


class ProcessStep(BaseModel):
    """产物③里的一步：研究过程记录中的单个动作。"""

    step: str = Field(description="plan / search / summarize / judge")
    round: int = Field(default=0, description="属于第几轮（规划为第 0 轮）")
    detail: str = Field(default="", description="这一步做了什么")


class ResearchProcess(BaseModel):
    """产物③：研究过程记录。"""

    plan: list[str] = Field(default_factory=list, description="初始关键词")
    search_queries: list[str] = Field(default_factory=list, description="全部检索过的关键词")
    reviewed_urls: list[str] = Field(default_factory=list, description="读过的页面 URL（去重）")
    iterations: int = Field(default=0, description="研究了几轮")
    steps: list[ProcessStep] = Field(default_factory=list, description="逐步动作明细")


# ===========================================================================
# ③ 接口与落盘
# ===========================================================================


class ResearchStatus(str, Enum):
    """状态机：pending → running → completed / failed。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def new_research_id() -> str:
    """生成一个短的研究 ID，用作落盘文件名与接口路径参数。"""
    return uuid.uuid4().hex[:12]


class ResearchRecord(BaseModel):
    """落盘到 backend/data/research/{id}.json 的完整记录。

    它同时是 GET /api/research/{id} 的返回体——
    因为"轮询能看到过程"靠的就是把中间结果也写进这个文件。
    """

    id: str = Field(default_factory=new_research_id)
    topic: str
    status: ResearchStatus = ResearchStatus.PENDING
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    error: str = Field(default="", description="失败原因，仅 status=failed 时有值")

    # ---- 四类产物 ----
    report: ReportContent | None = Field(default=None, description="产物①结构化报告")
    report_html: str = Field(default="", description="产物①的可视化版本：自包含 HTML")
    sources: list[Source] = Field(default_factory=list, description="产物②来源列表")
    process: ResearchProcess = Field(default_factory=ResearchProcess, description="产物③研究过程")
    confidence: Confidence | None = Field(default=None, description="产物④置信度说明")


class ResearchRequest(BaseModel):
    """POST /api/research 的请求体。"""

    topic: str = Field(min_length=1, description="要研究的主题")


class ResearchCreated(BaseModel):
    """POST /api/research 的返回体：立即返回 ID，研究在后台跑。"""

    research_id: str
    status: ResearchStatus
    topic: str


def _demo() -> None:
    """本地自检用的示例数据。"""
    return ResearchRecord(
        topic="2026 年国内大模型 API 价格趋势",
        process=ResearchProcess(
            plan=["大模型 API 价格"],
            search_queries=["大模型 API 价格"],
            reviewed_urls=["https://example.com/a"],
            iterations=1,
            steps=[ProcessStep(step="plan", round=0, detail="生成 1 个关键词")],
        ),
    )


if __name__ == "__main__":
    # 本地自检：python3 -m backend.models
    print("=" * 60)
    print("模型序列化自检")
    print("=" * 60)

    record = _demo()
    print(f"研究 ID      : {record.id}")
    print(f"状态默认值   : {record.status.value}")

    # 1) 序列化成 JSON 字符串
    text = record.model_dump_json(indent=2)
    print(f"JSON 长度    : {len(text)} 字符")

    # 2) 再解析回来，应当与原来完全相等
    back = ResearchRecord.model_validate_json(text)
    assert back == record, "序列化后解析不回来！"
    print("往返序列化   : 通过（dump → load 内容一致）")

    # 3) 验证"容忍模型不老实"的几个兜底
    ko = KeywordOutput.model_validate({"keywords": [{"keyword": "苹果"}, "华为"]})
    assert ko.keywords == ["苹果", "华为"], ko.keywords
    print(f"关键词兜底   : 通过 → {ko.keywords}")

    jd = JudgeDecision.model_validate(
        {"sufficient": False, "reason": "缺数据", "new_keywords": "价格、份额"}
    )
    assert jd.new_keywords == ["价格", "份额"], jd.new_keywords
    print(f"补检词拆分   : 通过 → {jd.new_keywords}")

    cd = ConclusionDraft.model_validate("这是一句没有来源的结论")
    assert cd.text == "这是一句没有来源的结论" and cd.sources == []
    print("纯字符串结论 : 通过 → 自动补空来源")

    ro = ReportOutline.model_validate(
        {
            "title": "标题",
            "summary": "摘要",
            "key_conclusions": ["结论一", {"text": "结论二", "sources": "https://a.com"}],
            "open_questions": "只有一个遗留问题",
        }
    )
    print(f"报告元信息   : 通过 → {len(ro.key_conclusions)} 条结论 / {len(ro.open_questions)} 个遗留问题")

    # 4) 默认值检查：新建的记录应当各字段有合理默认
    fresh = ResearchRecord(topic="测试")
    assert fresh.status is ResearchStatus.PENDING
    assert fresh.process.iterations == 0
    assert fresh.report is None
    print("默认值       : 通过")

    print()
    print("全部检查通过。")
