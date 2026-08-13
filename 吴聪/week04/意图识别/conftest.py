# -*- coding: utf-8 -*-
"""pytest 根配置：保证从项目根目录可导入 ecommerce_qa 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
