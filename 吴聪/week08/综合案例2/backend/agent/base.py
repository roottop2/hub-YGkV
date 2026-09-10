"""BaseAgent：所有角色 agent 的基类。

一个"角色 agent"在本项目里 = **无工具的单次 LLM 调用**：
    渲染提示词（Jinja2 模板） → Agent(model=DeepSeek) → Runner.run() → 拿到文本

它不自己搜网页、不循环、不做工具调用。「什么时候搜、搜完干什么」由
engine.py 确定性控制。这样每个 agent 的行为都是可预期的一次输入一次输出。

三个必须知道的实现约束（来自 CLAUDE.md）：

1. **绝不在 async 上下文里用 run_sync**
   本项目的调用链是 async 的（FastAPI + httpx），用同步版本会把事件循环卡死。
   统一用 `await Runner.run(...)`。

2. **DeepSeek 不支持 `Agent(output_type=...)`**
   那个参数会让 SDK 发出 `response_format: json_schema`，DeepSeek 会直接报错。
   所以改成：提示词里强制模型只输出一个 JSON，我们自己解析（parse_json）。

3. **空输出要重试**
   DeepSeek 偶尔返回 HTTP 200 但 `message.content` 是空的（不是超时——超时会抛异常）。
   拿到空串就重试，每次退避 1 秒。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

from .. import config
from ..models import Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK 全局初始化（模块级，只执行一次）
# ---------------------------------------------------------------------------
# OpenAI Agents SDK 默认要连 OpenAI 官方。这里通过环境变量把它指向 DeepSeek。
# 用 setdefault 而不是直接赋值：万一使用者已经在系统里配了 OPENAI_* 就不覆盖。
os.environ.setdefault("OPENAI_API_KEY", config.DEEPSEEK_API_KEY)
os.environ.setdefault("OPENAI_BASE_URL", config.DEEPSEEK_BASE_URL)

# DeepSeek 只实现了 OpenAI 的 /chat/completions，
# 不支持新的 /responses 接口，所以必须显式声明走 chat_completions。
set_default_openai_api("chat_completions")
# 关掉 SDK 自带的追踪上报（否则它会尝试把数据发到 OpenAI 平台，一直报错）
set_tracing_disabled(True)

# 提示词模板目录：backend/templates/
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# 提示词里禁止出现未定义的变量（写错变量名会立刻报错，而不是渲染成空白）
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

# 泛型变量：让 IDE 知道 call_json(text, JudgeDecision) 返回的是 JudgeDecision
TModel = TypeVar("TModel", bound=BaseModel)


class EmptyOutputError(RuntimeError):
    """重试若干次后模型仍然返回空内容。"""


class JSONParseError(ValueError):
    """模型输出既不是合法 JSON，也不是能认出来的 JSON 片段。"""


class BaseAgent:
    """角色 agent 基类。

    子类只需要指定 `name`（日志里显示的名字）和 `template`（模板文件名），
    再包一层语义化的方法给引擎调用即可，例如：

        class KeywordAgent(BaseAgent):
            name = "keyword"
            template = "keyword_agent.jinja2"
    """

    #: 日志中显示的名字
    name: str = "base"
    #: backend/templates/ 下的模板文件名
    template: str = ""

    def __init__(self, model: str | None = None) -> None:
        # 允许注入别的模型（方便测试），默认用 .env 里配的 DeepSeek 模型
        self.model = model or config.DEEPSEEK_MODEL

    # ------------------------------------------------------------------
    # 渲染提示词
    # ------------------------------------------------------------------
    def render(self, template: str | None = None, **variables: Any) -> str:
        """把模板渲染成最终的提示词字符串。

        模板放在 backend/templates/ 里，好处是：
        改提示词不用动 Python 代码，两者互不干扰。
        """
        name = template or self.template
        if not name:
            raise ValueError(f"{type(self).__name__} 没有指定 template")

        tpl = _TEMPLATE_ENV.get_template(name)
        rendered = tpl.render(**variables)
        logger.debug("[%s] 渲染模板 %s，得到 %d 字符的提示词", self.name, name, len(rendered))
        return rendered

    # ------------------------------------------------------------------
    # 调用模型
    # ------------------------------------------------------------------
    async def _run(
        self,
        system_vars: dict[str, Any],
        user_input: str,
        template: str | None = None,
    ) -> str:
        """渲染提示词 → 调一次模型 → 返回文本（空输出自动重试）。

        system_vars 会填进模板，渲染结果作为 agent 的 instructions（系统提示词）；
        user_input 是这次要处理的具体内容。

        template 用于"一个 agent 有多个模板"的情况（如 ReportAgent 分别有
        元信息模板和 HTML 模板）；不传就用子类声明的 `self.template`。
        """
        instructions = self.render(template, **system_vars)
        agent = Agent(model=self.model, name=self.name, instructions=instructions)

        attempts = max(1, config.LLM_RETRIES)
        for attempt in range(1, attempts + 1):
            logger.debug(
                "[%s] 第 %d/%d 次调用模型 %s\n--- system ---\n%s\n--- user ---\n%s",
                self.name, attempt, attempts, self.model, instructions, user_input,
            )

            result = await Runner.run(agent, user_input)
            text = (result.final_output or "").strip()

            if text:
                logger.info("[%s] 调用成功，输出 %d 字符", self.name, len(text))
                logger.debug("[%s] 输出内容：\n%s", self.name, text)
                return text

            # 空输出：DeepSeek 偶发问题，退避后重试
            logger.warning(
                "[%s] 模型返回空内容（第 %d/%d 次），稍后重试", self.name, attempt, attempts
            )
            if attempt < attempts:
                await asyncio.sleep(1.0)

        raise EmptyOutputError(
            f"{self.name}：连续 {attempts} 次调用都返回空内容（模型 {self.model}）"
        )

    async def call_json(
        self,
        system_vars: dict[str, Any],
        user_input: str,
        output_cls: type[TModel],
        template: str | None = None,
    ) -> TModel:
        """调模型并把输出解析成 pydantic 对象。

        约定：提示词里已经要求模型"只输出一个 JSON 对象"，这里负责把它变成对象。
        """
        text = await self._run(system_vars, user_input, template=template)
        parsed = self.parse_json(text, output_cls)
        logger.info("[%s] 解析为 %s 成功", self.name, output_cls.__name__)
        return parsed

    # ------------------------------------------------------------------
    # 共享小工具（多个角色 agent 都要用）
    # ------------------------------------------------------------------
    @staticmethod
    def strip_code_fence(text: str) -> str:
        """去掉模型自作主张包上的 ``` 代码块。

        用在"期望纯文本输出"的地方（总结正文、HTML）。
        注意：**解析 JSON 时不能先剥围栏**——parse_json 需要保留原文
        才能用正则从围栏里把 JSON 抠出来。
        """
        raw = (text or "").strip()
        # 整个输出就是一个代码块时才剥；只是其中一段的话保持原样
        match = re.fullmatch(r"```[a-zA-Z]*[ \t]*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw

    @staticmethod
    def format_search_results(results: list[dict]) -> str:
        """把搜索结果排成带序号的文本，喂给总结 agent。

        序号很重要：提示词里要求正文用 `[2]` 这样的编号标注引用，
        编号必须和这里的顺序完全对应，来源才追得回去。
        """
        if not results:
            return "（本次没有搜索到任何结果）"

        blocks = []
        for i, item in enumerate(results, 1):
            lines = [f"[{i}] {item.get('title') or '(无标题)'}"]
            meta = []
            if item.get("site_name"):
                meta.append(f"站点：{item['site_name']}")
            if item.get("date"):
                meta.append(f"日期：{item['date']}")
            if meta:
                lines.append("    " + "   ".join(meta))
            lines.append(f"    链接：{item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"    摘要：{item['snippet']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def format_sources(sources: list[Source]) -> str:
        """把已收集的来源排成带序号的清单，喂给主笔 agent。

        这里给出的是**完整 URL 原文**——提示词要求主笔"原样复制"URL，
        引擎之后靠字符串精确匹配把 URL 还原成 Source 对象。
        """
        if not sources:
            return "（没有可用来源）"

        blocks = []
        for i, source in enumerate(sources, 1):
            lines = [f"[{i}] {source.title or '(无标题)'}"]
            meta = []
            if source.site_name:
                meta.append(f"站点：{source.site_name}")
            if source.date:
                meta.append(f"日期：{source.date}")
            if meta:
                lines.append("    " + "   ".join(meta))
            lines.append(f"    URL：{source.url}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # 解析模型输出
    # ------------------------------------------------------------------
    @staticmethod
    def parse_json(text: str, output_cls: type[TModel]) -> TModel:
        """从模型输出里抠出 JSON 并解析成 output_cls。

        模型不一定老老实实只给 JSON，常见三种情况，依次尝试：

            ① 纯 JSON                {"keywords": ["a"]}
            ② 包在代码块里           ```json\\n{"keywords": ["a"]}\\n```
            ③ 前后夹带解释文字       好的，结果如下：{...} 希望有帮助

        三种都试过还不行才报错——报错时会把原文附上，方便定位是哪次调用出的问题。
        """
        raw = (text or "").strip()
        if not raw:
            raise JSONParseError("模型输出为空，无法解析 JSON")

        candidates: list[str] = [raw]

        # ② ```json ... ``` 或 ``` ... ``` 代码块
        for match in re.finditer(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL):
            candidates.append(match.group(1).strip())

        # ③ 第一个 { 到最后一个 } 之间的内容
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])

        last_error: Exception | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return output_cls.model_validate_json(candidate)
            except Exception as exc:  # noqa: BLE001 —— 逐个候选试，失败就换下一个
                last_error = exc
                # 有些模型会输出"合法 JSON 但类型不对"，这时也退化成 dict 再校验一次
                # （例如把数字写成字符串 "3"，pydantic 的 lax 模式能救回来）
                try:
                    return output_cls.model_validate(json.loads(candidate))
                except Exception as exc2:  # noqa: BLE001
                    last_error = exc2

        raise JSONParseError(
            f"无法把模型输出解析成 {output_cls.__name__}：{last_error}\n"
            f"── 原始输出 ──\n{raw[:800]}"
        )


if __name__ == "__main__":
    # 本地自检（不需要网络）：python3 -m backend.agent.base
    # 只测"解析 JSON"和"渲染模板"这两件纯本地的事。
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from ..models import JudgeDecision

    print("=" * 60)
    print("BaseAgent 自检（纯本地，不发网络请求）")
    print("=" * 60)
    print(f"模型          : {config.DEEPSEEK_MODEL}")
    print(f"模板目录      : {TEMPLATE_DIR}  ({'存在' if TEMPLATE_DIR.exists() else '不存在！'})")
    print()

    # --- 1. parse_json 的三种形态 ---
    print("【1】parse_json 三种形态")

    case1 = '{"sufficient": true, "reason": "够了", "new_keywords": []}'
    r1 = BaseAgent.parse_json(case1, JudgeDecision)
    assert r1.sufficient is True and r1.reason == "够了"
    print("  ① 纯 JSON          → 通过")

    case2 = '```json\n{"sufficient": false, "reason": "缺价格", "new_keywords": ["价格"]}\n```'
    r2 = BaseAgent.parse_json(case2, JudgeDecision)
    assert r2.sufficient is False and r2.new_keywords == ["价格"]
    print(f"  ② ```json 代码块   → 通过（补检词 {r2.new_keywords}）")

    case3 = '好的，我的判断如下：\n{"sufficient": true, "reason": "信息充足"}\n希望有帮助！'
    r3 = BaseAgent.parse_json(case3, JudgeDecision)
    assert r3.sufficient is True
    print("  ③ 前后夹带文字     → 通过")

    # --- 2. 类型不对时能否救回来 ---
    #
    # 注意 pydantic v2 比 v1 严格得多：它**不会**把 123 自动变成 "123"，
    # 也不会把 None 变成 ""。所以 models.py 里给模型自由发挥的文字字段
    # 套了 LooseText（见 models._to_text）。这两条就是验证那层兜底。
    print()
    print("【2】类型不严格时兜底")

    case4 = '{"sufficient": "true", "reason": 123, "new_keywords": "价格、份额"}'
    r4 = BaseAgent.parse_json(case4, JudgeDecision)
    assert r4.sufficient is True, r4.sufficient
    assert r4.reason == "123", r4.reason
    assert r4.new_keywords == ["价格", "份额"], r4.new_keywords
    print(f'  字符串布尔 → True；数字理由 → "{r4.reason}"；顿号串 → {r4.new_keywords}')

    case5 = '{"sufficient": false, "reason": null, "new_keywords": null}'
    r5 = BaseAgent.parse_json(case5, JudgeDecision)
    assert r5.reason == "" and r5.new_keywords == [], r5
    print("  null 字段 → 空串 / 空列表")

    # --- 3. 彻底无法解析时必须报错，且报错信息里带原文 ---
    print()
    print("【3】无法解析时应当报错")
    for bad in ["我觉得这个问题很难回答。", '{"sufficient": "也许", "reason": "x"}']:
        try:
            BaseAgent.parse_json(bad, JudgeDecision)
        except JSONParseError as exc:
            assert bad in str(exc), "报错信息里应带上原始输出"
            print(f"  {bad[:24]!r}... → 正确抛出 JSONParseError，且错误信息含原文")
        else:
            raise AssertionError(f"本该报错却成功了：{bad}")

    # --- 4. 渲染真实模板文件 ---
    print()
    print("【4】模板渲染")

    if TEMPLATE_DIR.exists() and any(TEMPLATE_DIR.glob("*.jinja2")):
        for tpl_path in sorted(TEMPLATE_DIR.glob("*.jinja2")):
            source = _TEMPLATE_ENV.get_template(tpl_path.name)
            # 只检查"语法能否编译通过"，不真正渲染（真实渲染需要完整变量）
            print(f"  {tpl_path.name:28} 语法正常")
    else:
        print("  （模板目录还没有 .jinja2 文件，跳过）")

    # 用独立的 DictLoader 验证"漏传变量必须报错"这一行为，
    # 不去动全局的 FileSystemLoader（那是 jinja2 的内部结构，版本升级会变）。
    from jinja2 import DictLoader

    probe_env = Environment(
        loader=DictLoader({"probe.jinja2": "主题={{ topic }}；条数={{ n }}"}),
        undefined=StrictUndefined,
    )
    rendered = probe_env.get_template("probe.jinja2").render(topic="大模型价格", n=3)
    assert rendered == "主题=大模型价格；条数=3", repr(rendered)
    print(f"  正常渲染           → {rendered!r}")

    try:
        probe_env.get_template("probe.jinja2").render(topic="只有主题")
    except Exception as exc:  # jinja2.UndefinedError
        print(f"  漏传变量           → 正确报错（{type(exc).__name__}）")
    else:
        raise AssertionError("漏传变量本该报错")

    print()
    print("全部检查通过。")
