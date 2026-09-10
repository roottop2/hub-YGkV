"""agent 包级自检：python3 -m backend.agent

`__init__.py` 只做导出，不放可执行块；包的测试入口统一放这里。
本文件只检查"包能不能正常导入、类有没有缺方法"，**不发网络请求**。
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def main() -> None:
    from . import (
        BaseAgent,
        EmptyOutputError,
        JSONParseError,
        JudgeAgent,
        KeywordAgent,
        ReportAgent,
        SummaryAgent,
    )

    print("=" * 60)
    print("agent 包导出自检")
    print("=" * 60)

    # 每个角色 agent 的名字、模板、对外方法签名
    expected = [
        (KeywordAgent, "keyword", "keyword_agent.jinja2", "generate_keywords"),
        (SummaryAgent, "summary", "summary_agent.jinja2", "summarize"),
        (JudgeAgent, "judge", "judge_agent.jinja2", "judge"),
        (ReportAgent, "report", None, "generate"),  # ReportAgent 有两个模板，template 留空
    ]

    for cls, name, template, method_name in expected:
        assert issubclass(cls, BaseAgent), f"{cls.__name__} 没继承 BaseAgent"
        assert cls.name == name, f"{cls.__name__}.name 应为 {name}，实为 {cls.name}"

        if template is not None:
            assert cls.template == template, f"{cls.__name__}.template 应为 {template}"
            # 模板文件必须真实存在，否则运行时才报错就太晚了
            path = TEMPLATES_DIR / template
            assert path.exists(), f"模板文件不存在：{path}"

        method = getattr(cls, method_name, None)
        assert callable(method), f"{cls.__name__} 缺少方法 {method_name}()"
        assert inspect.iscoroutinefunction(method), f"{cls.__name__}.{method_name} 必须是 async"

        print(f"  {cls.__name__:14} name={name:8} 方法={method_name}()  ✓")

    # 异常类型可被正常抛出与捕获
    for exc in (EmptyOutputError, JSONParseError):
        assert issubclass(exc, Exception), exc
    print(f"\n  异常类型     : {EmptyOutputError.__name__} / {JSONParseError.__name__}  ✓")

    # 模板目录里应当正好有 5 个模板
    names = sorted(p.name for p in TEMPLATES_DIR.glob("*.jinja2"))
    print(f"\n  模板文件（{len(names)} 个）：")
    for n in names:
        print(f"    - {n}")
    assert len(names) == 5, f"模板应为 5 个，实际 {len(names)} 个：{names}"

    # 每个模板都要能被 Jinja2 编译（查语法错误）
    from .base import _TEMPLATE_ENV

    for n in names:
        _TEMPLATE_ENV.get_template(n)
    print("\n  全部模板语法编译通过 ✓")

    print()
    print("agent 包自检通过。")


if __name__ == "__main__":
    main()
