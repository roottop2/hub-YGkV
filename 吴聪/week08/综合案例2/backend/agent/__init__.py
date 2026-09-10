"""角色 agent 包。

**这里只放"角色"，不放"编排"。**

每个角色 agent 都是**无工具的单次 LLM 调用**（继承 BaseAgent）：
它只负责"想"，不负责"搜"，也不自己循环。
「什么时候搜、搜完干什么、要不要再来一轮」全部由 engine.py 的
DeepResearch 用确定性代码控制。

    BaseAgent     基类（模板渲染 / 调用模型 / 解析 JSON）
    KeywordAgent  规划：主题 → 一组搜索关键词
    SummaryAgent  总结：一批搜索结果 → 一段正文
    JudgeAgent    审核：现有草稿够不够 → 要不要补检
    ReportAgent   主笔：草稿 + 来源 → 结构化报告 + HTML
"""

from .base import BaseAgent, EmptyOutputError, JSONParseError
from .judge import JudgeAgent
from .keyword import KeywordAgent
from .report import ReportAgent
from .summary import SummaryAgent

__all__ = [
    "BaseAgent",
    "EmptyOutputError",
    "JSONParseError",
    "KeywordAgent",
    "SummaryAgent",
    "JudgeAgent",
    "ReportAgent",
]
