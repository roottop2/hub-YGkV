"""角色 agent 包。

**这里只放"角色"，不放"编排"。**

每个角色 agent 都是**无工具的单次 LLM 调用**（继承 BaseAgent）：
它只负责"想"，不负责"搜"，也不自己循环。
「什么时候搜、搜完干什么、要不要再来一轮」全部由 engine.py 的
DeepResearch 用确定性代码控制。

具体角色见 base.py 之上的各模块（keyword / summary / judge / report）。
"""

from .base import BaseAgent, EmptyOutputError, JSONParseError

__all__ = [
    "BaseAgent",
    "EmptyOutputError",
    "JSONParseError",
]
