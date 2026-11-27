"""Structured JSON logging."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")

_RESERVED: Final = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    """Turn each log record into one JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": run_id_var.get(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, as_json: bool = True) -> None:
    """Install one handler on the root logger. Call only from the entry point."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if as_json else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
