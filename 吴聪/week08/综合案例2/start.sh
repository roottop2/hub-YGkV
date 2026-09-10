#!/usr/bin/env bash
# 启动深度研究助手后端。
# 用法：在项目根目录执行  bash start.sh
# 端口默认 8000，若被占用会自动改用 8001。
set -e

PORT="${PORT:-8000}"
# 已经被占用就顺延到 8001
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "[start] 使用解释器：$($PY --version)"
echo "[start] 监听端口：$PORT"

exec "$PY" -m uvicorn backend.app:app --reload --port "$PORT"
