from __future__ import annotations

import json
import logging
import os
import sys
import threading
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from cryoseed.runtime.distributed import get_rank, get_world_size, is_rank0


__all__ = [
    "setup_logging",
    "get_logger",
    "info_rank0",
    "log_exception",
    "log_metrics",
    "log_config",
    "log_state",
    "log_block",
]


_LOGGING_INITIALIZED = False


class RankFilter(logging.Filter):
    def __init__(self, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.world_size = self.world_size
        return True


class Rank0OnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return is_rank0()


def _make_formatter() -> logging.Formatter:
    fmt = (
        "[%(asctime)s] "
        "[%(levelname)s] "
        "[rank %(rank)s/%(world_size)s] "
        "[%(name)s] "
        "%(message)s"
    )
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    return repr(obj)


def _install_excepthook(logger: logging.Logger) -> None:
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    def handle_thread_exception(args):
        logger.critical(
            "Uncaught thread exception in %s",
            args.thread.name if args.thread is not None else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = handle_thread_exception


def setup_logging(
    log_dir: str | os.PathLike,
    *,
    filename_prefix: str = "train",
    level: int = logging.INFO,
    capture_warnings: bool = True,
    overwrite_handlers: bool = True,
) -> logging.Logger:
    global _LOGGING_INITIALIZED

    rank = get_rank()
    world_size = get_world_size()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    if overwrite_handlers:
        logger.handlers.clear()

    formatter = _make_formatter()
    rank_filter = RankFilter(rank=rank, world_size=world_size)

    # file handler: every rank writes its own file
    file_path = log_dir / f"{filename_prefix}.rank{rank}.log"
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    fh.addFilter(rank_filter)
    logger.addHandler(fh)

    # console handler: only rank0 prints to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.addFilter(rank_filter)
    ch.addFilter(Rank0OnlyFilter())
    logger.addHandler(ch)

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)

    _install_excepthook(logger)
    _LOGGING_INITIALIZED = True

    logger.info("Logging initialized: rank=%d world_size=%d file=%s", rank, world_size, file_path)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def info_rank0(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    if not is_rank0():
        return
    logger.info(msg, *args, **kwargs)


def log_exception(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    logger.exception(msg, *args, **kwargs)


def log_metrics(
    logger: logging.Logger,
    metrics: dict[str, Any],
    *,
    step: int | None = None,
    prefix: str = "metrics",
) -> None:
    payload = dict(metrics)
    if step is not None:
        payload["step"] = step
    logger.info("%s %s", prefix, json.dumps(payload, default=_json_default, ensure_ascii=False))


def log_config(
    logger: logging.Logger,
    config: Any,
    *,
    title: str = "Configuration",
    rank0_only: bool = True,
) -> None:
    if rank0_only and not is_rank0():
        return

    payload = json.dumps(config, default=_json_default, ensure_ascii=False, indent=2)
    log_block(logger, title=title, lines=payload.splitlines(), rank0_only=False)


def log_state(
    logger: logging.Logger,
    state: Any,
    *,
    title: str = "State",
    rank0_only: bool = True,
) -> None:
    if rank0_only and not is_rank0():
        return

    payload_obj = state.to_dict() if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")) else state
    payload = json.dumps(payload_obj, default=_json_default, ensure_ascii=False, indent=2)
    log_block(logger, title=title, lines=payload.splitlines(), rank0_only=False)


def log_block(logger, title: str, lines: list[str], *, rank0_only=True):
    if rank0_only and not is_rank0():
        return

    width = 60
    header = f"{title}".center(width, "=")
    footer = "=" * width

    msg = "\n".join([header, *lines, footer])
    logger.info("\n" + msg)