"""项目共用的 logger 工厂，避免每个模块都重复配置 handler。"""

from __future__ import annotations

import logging
import sys


_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """返回带统一 handler/format 的 logger。多次调用不会重复添加 handler。"""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
