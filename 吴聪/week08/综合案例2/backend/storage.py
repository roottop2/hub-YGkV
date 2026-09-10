"""落盘读写：backend/data/research/{研究id}.json

一份研究记录从头到尾就是这一个 JSON 文件，状态就写在里面
（pending → running → completed / failed）。

**为什么轮询能看到研究过程？**
因为研究每跑完一轮，research.py 就会调一次 save() 把当前进度写进这个文件；
前端反复 GET 同一个 ID，读到的就是不断变长的 process.steps / draft / sources。

**为什么要加锁？**
FastAPI 是异步的，但落盘用的是同步文件 IO。研究任务在后台写、
接口线程同时在读，两个线程可能同时动同一个文件。
threading.Lock() 保证同一时刻只有一个线程在读写，避免读到写了一半的内容。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from . import config
from .models import ResearchRecord, now_iso

logger = logging.getLogger(__name__)

# 一把全局锁，保护下面所有对文件的读写。
_lock = threading.Lock()


def _path(research_id: str) -> Path:
    """由研究 ID 推出它对应的文件路径。"""
    return config.DATA_DIR / f"{research_id}.json"


def ensure_dir() -> None:
    """确保落盘目录存在（第一次运行时目录还不存在）。"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def save(record: ResearchRecord) -> Path:
    """把一条研究记录写进磁盘，返回写入的文件路径。

    写法是"先写临时文件，再原子替换"：
      1. 先把内容完整写进 {id}.json.tmp
      2. 再用 os.replace 把 .tmp 改名成 {id}.json

    好处：如果写到一半程序崩了，损坏的只是 .tmp，
    正式的 {id}.json 要么是旧的完整内容、要么是新的完整内容，
    **永远不会是写了一半的残缺 JSON**——轮询的接口也就不会因为
    解析半个文件而报错。
    """
    ensure_dir()
    record.updated_at = now_iso()
    target = _path(record.id)
    tmp = target.with_suffix(".json.tmp")

    payload = record.model_dump_json(indent=2)

    with _lock:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)  # 原子替换，Windows 上同样有效

    logger.debug("落盘 %s（%d 字节，状态 %s）", record.id, len(payload), record.status.value)
    return target


def load(research_id: str) -> ResearchRecord | None:
    """按 ID 读回一条记录；不存在或文件损坏时返回 None 并记日志。"""
    target = _path(research_id)
    with _lock:
        if not target.exists():
            return None
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:  # 文件被占用等
            logger.error("读取 %s 失败：%s", research_id, exc)
            return None

    try:
        # 解析放在锁外：JSON 字符串已经在内存里了，不必占着锁做纯计算
        return ResearchRecord.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 —— 损坏的 JSON 不该让整个列表接口挂掉
        logger.error("解析 %s 失败（文件可能损坏）：%s", research_id, exc)
        return None


def list_records(limit: int = 100) -> list[ResearchRecord]:
    """列出全部研究记录，按创建时间**从新到旧**排序。

    文件损坏的记录会被跳过（load 返回 None），不会影响其它记录。
    """
    if not config.DATA_DIR.exists():
        return []

    records = []
    for path in config.DATA_DIR.glob("*.json"):
        record = load(path.stem)
        if record is not None:
            records.append(record)

    # created_at 是 ISO 格式（如 2026-09-10T17:42:01），
    # 这种格式按字符串排序的结果和时间先后一致，不必转成 datetime。
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records[:limit]


def delete(research_id: str) -> bool:
    """删除一条记录，返回是否真的删掉了。"""
    target = _path(research_id)
    with _lock:
        if not target.exists():
            return False
        target.unlink()
    logger.info("已删除研究记录 %s", research_id)
    return True


if __name__ == "__main__":
    # 本地自检：python3 -m backend.storage
    # 自建一条测试记录，跑完在 finally 里删掉，不污染真实数据。
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 60)
    print("落盘读写自检")
    print("=" * 60)
    print(f"落盘目录：{config.DATA_DIR}")

    test_id = "selfcheck_storage"
    try:
        # 1) 写
        rec = ResearchRecord(topic="自检用主题")
        rec.id = test_id
        path = save(rec)
        print(f"写入成功     : {path.name}  ({path.stat().st_size} 字节)")

        # 2) 读
        back = load(test_id)
        assert back is not None, "读回来是 None"
        assert back.topic == "自检用主题", back.topic
        print(f"读回成功     : topic={back.topic!r}  status={back.status.value}")

        # 3) 改一个字段再存，确认会覆盖
        back.topic = "改过的主题"
        save(back)
        again = load(test_id)
        assert again is not None and again.topic == "改过的主题", again
        print(f"覆盖成功     : topic={again.topic!r}")

        # 4) 不存在的 ID
        assert load("绝对不存在的id") is None
        print("不存在返回   : None（符合预期）")

        # 5) 列表里能找到它
        ids = [r.id for r in list_records()]
        assert test_id in ids, f"列表里没有测试记录：{ids[:5]}"
        print(f"列表查询     : 共 {len(ids)} 条，含测试记录")

        # 6) 时间戳应当被 save 刷新
        assert again.updated_at >= again.created_at
        print(f"更新时间戳   : {again.updated_at}")

    finally:
        removed = delete(test_id)
        print(f"清理测试记录 : {'已删除' if removed else '（本来就不存在）'}")

    print()
    print("全部检查通过，且未残留测试数据。")
