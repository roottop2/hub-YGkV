"""深度研究助手 —— 后端包。

模块划分（自上而下）：

    app.py       FastAPI 接口层（HTTP 入口）
    research.py  研究任务编排（后台执行 + 中间结果落盘）
    engine.py    研究引擎 DeepResearch（确定性控制流，不含 LLM 调用）
    agent/       角色 agent（无工具的单次 LLM 调用）
    tools.py     Bocha 网页搜索（普通 async 函数）
    storage.py   JSON 落盘读写
    models.py    所有 pydantic 数据模型
    config.py    从项目根 .env 读配置
"""
