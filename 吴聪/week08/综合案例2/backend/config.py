"""配置读取。

所有配置都放在**项目根目录**的 `.env` 里（不是 backend/.env）。
本模块负责把它读进来，供其它模块直接 import 使用，例如：

    from . import config
    print(config.DEEPSEEK_MODEL)

为什么用 python-dotenv 而不是 os.environ？
    .env 是一个普通的文本文件，操作系统并不认识它。
    `load_dotenv()` 会把文件里的 `KEY=VALUE` 逐行读出来塞进 os.environ，
    之后 os.getenv() 才能取到。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. 定位项目根目录
# ---------------------------------------------------------------------------
# __file__            → .../综合案例2/backend/config.py
# .resolve()          → 转成绝对路径，去掉 .. 之类的相对写法
# .parent             → .../综合案例2/backend
# .parent.parent      → .../综合案例2         ← 这就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# 记下"读 .env 之前"系统里已经有哪些键。
# 用途：load_dotenv 默认 override=False（系统环境变量优先），
# 所以如果系统里已设了 DEEPSEEK_API_KEY，.env 里那份就会被忽略。
# 这个集合用来告诉使用者"最终生效的密钥到底来自哪里"，省得排查半天。
_PRESET_BY_SYSTEM = {k for k in ("DEEPSEEK_API_KEY", "BOCHA_API_KEY") if os.getenv(k)}

# 把 .env 里的键值对注入 os.environ。
# override=False（默认）表示：如果系统环境变量里已经有同名变量，就不覆盖它。
load_dotenv(ENV_FILE)


def key_origin(name: str) -> str:
    """这个密钥最终是从哪来的？（写日志/自检时很有用）"""
    if not os.getenv(name):
        return "未配置"
    return "系统环境变量" if name in _PRESET_BY_SYSTEM else ".env 文件"


def _get_str(name: str, default: str = "") -> str:
    """读一个字符串配置；没配或配成空串就用默认值。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_int(name: str, default: int) -> int:
    """读一个整数配置；没配、写错格式或小于 1 就用默认值。

    初学者常见坑：.env 里读出来永远是字符串，
    所以 `int("3")` 才能得到数字 3，直接比较会出错。
    """
    raw = _get_str(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


# ---------------------------------------------------------------------------
# 2. 具体配置项
# ---------------------------------------------------------------------------
# -- DeepSeek：负责全部 LLM 调用（规划 / 总结 / 判断 / 写报告）--
DEEPSEEK_API_KEY = _get_str("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _get_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/")
DEEPSEEK_MODEL = _get_str("DEEPSEEK_MODEL", "deepseek-v4-flash")

# -- Bocha：负责网页搜索 --
BOCHA_API_KEY = _get_str("BOCHA_API_KEY")
BOCHA_SEARCH_URL = _get_str("BOCHA_SEARCH_URL", "https://api.bocha.cn/v1/web-search")

# -- 研究行为参数 --
# 最多研究几轮。到上限还没"搜够"也会强制出报告，防止无限循环。
RESEARCH_MAX_ROUNDS = _get_int("RESEARCH_MAX_ROUNDS", 3)
# 每个关键词取多少条搜索结果。
SEARCH_COUNT = _get_int("SEARCH_COUNT", 10)
# DeepSeek 偶发返回 200 但内容为空，此时自动重试几次。
LLM_RETRIES = _get_int("LLM_RETRIES", 3)

# -- 落盘目录：backend/data/research/{研究id}.json --
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "research"


def mask(secret: str) -> str:
    """把密钥打码后再打印，避免完整密钥出现在日志里。

    例：sk-b2987442ad2c4629aa2ed309e340bca3
        → sk-b29...bca3
    """
    if not secret:
        return "(未配置)"
    if len(secret) <= 12:
        return secret[:3] + "***"
    return f"{secret[:6]}...{secret[-4:]}"


def describe() -> str:
    """生成一份人类可读的配置摘要（密钥已打码）。"""
    lines = [
        f"项目根目录     : {PROJECT_ROOT}",
        f".env 文件      : {ENV_FILE}  ({'存在' if ENV_FILE.exists() else '不存在！'})",
        "",
        f"DeepSeek 地址  : {DEEPSEEK_BASE_URL}",
        f"DeepSeek 模型  : {DEEPSEEK_MODEL}",
        f"DeepSeek 密钥  : {mask(DEEPSEEK_API_KEY)}  (来自 {key_origin('DEEPSEEK_API_KEY')})",
        "",
        f"Bocha 搜索地址 : {BOCHA_SEARCH_URL}",
        f"Bocha 密钥     : {mask(BOCHA_API_KEY)}  (来自 {key_origin('BOCHA_API_KEY')})",
        "",
        f"最大研究轮数   : {RESEARCH_MAX_ROUNDS}",
        f"每次搜索条数   : {SEARCH_COUNT}",
        f"LLM 空输出重试 : {LLM_RETRIES}",
        f"落盘目录       : {DATA_DIR}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # 本地自检：python3 -m backend.config
    # 不需要网络，也不打印完整密钥。
    print("=" * 60)
    print("配置自检")
    print("=" * 60)
    print(describe())

    print()
    problems = []
    if not DEEPSEEK_API_KEY:
        problems.append("DEEPSEEK_API_KEY 未配置（所有 LLM 调用会失败）")
    if not BOCHA_API_KEY:
        problems.append("BOCHA_API_KEY 未配置（搜索会失败）")

    if problems:
        print("发现以下问题：")
        for p in problems:
            print(f"  - {p}")
        print("请把 .env.example 复制为 .env 并填写密钥。")
    else:
        print("配置检查通过：DeepSeek 与 Bocha 密钥均已配置。")
