"""搜索工具：调 Bocha 网页搜索 API。

文档：https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
端点：POST https://api.bocha.cn/v1/web-search
鉴权：请求头 Authorization: Bearer <API key>

**这里是一个普通的 async 函数，不是 agent。**
整个项目只有 engine.py 会调它——四个角色 agent 全都"没有工具"，
它们只负责"想"（生成关键词、写总结、做判断），"搜"这个动作由引擎确定性发起。
这样流程完全可控，不会出现 agent 自己空转乱搜的情况。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from . import config

logger = logging.getLogger(__name__)

# 网络请求超时（秒）。搜索接口一般几百毫秒，30 秒足够宽松。
_TIMEOUT = 30.0


class SearchError(RuntimeError):
    """搜索失败。引擎会捕获它，记一笔日志后继续跑，不让整个研究任务崩掉。"""


def _first(item: dict, *keys: str) -> str:
    """从字典里按顺序取第一个非空的字符串值。

    Bocha 的字段名有两种写法（驼峰 siteName / 下划线 site_name 都出现过），
    而且不同接口版本字段略有差异，所以统一用"挨个试"的方式取值。
    """
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_item(item: Any) -> dict | None:
    """把 Bocha 返回的一条原始结果，转成本项目内部统一的结构。

    统一结构（全项目通用，engine 与 summary agent 都按这个读）：
        {"title": ..., "url": ..., "snippet": ..., "site_name": ..., "date": ...}
    """
    if not isinstance(item, dict):
        return None

    url = _first(item, "url", "link")
    if not url:
        # 没有 URL 的结果无法作为"来源"被引用，直接丢弃
        return None

    return {
        "title": _first(item, "name", "title"),
        "url": url,
        # 请求时 summary=true，官方会在 summary 里给更完整的摘要；
        # 万一没有就退回较短的 snippet。
        "snippet": _first(item, "summary", "snippet", "description"),
        "site_name": _first(item, "siteName", "site_name"),
        "date": _first(item, "datePublished", "dateLastCrawled", "date"),
    }


def parse_response(data: Any) -> list[dict]:
    """从 Bocha 的完整响应体里，把搜索结果列表扒出来。

    响应体大致长这样（层层嵌套）：
        {"code": 200,
         "data": {"webPages": {"value": [ {一条结果}, {另一条结果} ]}}}

    这里把每一层都做了容错：少任何一层都返回空列表，而不是抛 KeyError。
    """
    if not isinstance(data, dict):
        return []

    # 官方在业务错误时也返回 200 HTTP，但 body 里 code != 200
    code = data.get("code")
    if code is not None and code != 200:
        msg = data.get("msg") or data.get("message") or "未知错误"
        raise SearchError(f"Bocha 返回业务错误 code={code}：{msg}")

    pages = (data.get("data") or {}).get("webPages") or {}
    values = pages.get("value") or []
    if not isinstance(values, list):
        return []

    results = []
    for item in values:
        parsed = _parse_item(item)
        if parsed is not None:
            results.append(parsed)
    return results


async def web_search(
    query: str,
    count: int | None = None,
    summary: bool = True,
) -> list[dict]:
    """搜一次，返回结构化结果列表。

    参数：
        query   搜索词
        count   要几条，默认取 config.SEARCH_COUNT（.env 可配）
        summary 是否让接口顺带返回摘要（我们需要，因为总结正文靠它）

    返回：
        [{"title","url","snippet","site_name","date"}, ...]

    失败时抛 SearchError（含网络不通、超时、鉴权失败、业务错误码）。
    """
    query = (query or "").strip()
    if not query:
        return []

    if not config.BOCHA_API_KEY:
        raise SearchError("BOCHA_API_KEY 未配置，无法搜索（请检查 .env）")

    payload = {
        "query": query,
        "summary": summary,
        "count": count or config.SEARCH_COUNT,
    }
    headers = {
        "Authorization": f"Bearer {config.BOCHA_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("搜索关键词「%s」（取 %d 条）", query, payload["count"])

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                config.BOCHA_SEARCH_URL, headers=headers, json=payload
            )
    except httpx.HTTPError as exc:
        raise SearchError(f"请求 Bocha 失败（网络问题）：{exc}") from exc

    if response.status_code == 401:
        raise SearchError("Bocha 鉴权失败（401），请检查 BOCHA_API_KEY 是否正确")
    if response.status_code != 200:
        raise SearchError(f"Bocha 返回 HTTP {response.status_code}：{response.text[:200]}")

    try:
        data = response.json()
    except ValueError as exc:
        raise SearchError(f"Bocha 返回的不是合法 JSON：{response.text[:200]}") from exc

    results = parse_response(data)
    logger.info("关键词「%s」拿到 %d 条结果", query, len(results))
    return results


if __name__ == "__main__":
    # 需要网络与 BOCHA_API_KEY，人工验证用：python3 -m backend.tools
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    async def _demo() -> None:
        print("=" * 60)
        print("Bocha 搜索自检")
        print("=" * 60)
        print(config.describe())
        print()

        results = await web_search("天空为什么是蓝色的？", count=3)
        print(f"拿到 {len(results)} 条结果：\n")
        for i, item in enumerate(results, 1):
            print(f"[{i}] {item['title']}")
            print(f"    站点：{item['site_name']}   日期：{item['date']}")
            print(f"    链接：{item['url']}")
            print(f"    摘要：{item['snippet'][:120]}...")
            print()

        assert results, "一条结果都没拿到"
        assert all(item["url"] for item in results), "有结果缺少 URL"
        print("全部结果都带 URL，可作为来源引用。")

    asyncio.run(_demo())
