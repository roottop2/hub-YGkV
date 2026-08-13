# -*- coding: utf-8 -*-
"""统一的日志工具。"""
import logging


def get_logger(name: str) -> logging.Logger:
    """返回一个带控制台输出的 logger（避免重复添加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
