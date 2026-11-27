import json
import logging
from collections.abc import Iterator

import pytest

from ds_prep.logging_setup import JsonFormatter, configure_logging, run_id_var


@pytest.fixture
def restore_root() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_json_line_has_run_id_and_extra() -> None:
    token = run_id_var.set("abc123")
    rec = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "hi", "store_id": 7})
    line = json.loads(JsonFormatter().format(rec))
    run_id_var.reset(token)
    assert line["msg"] == "hi"
    assert line["store_id"] == 7
    assert line["run_id"] == "abc123"


def test_exception_is_captured() -> None:
    with pytest.raises(ValueError, match="boom") as ei:
        raise ValueError("boom")
    rec = logging.makeLogRecord({"msg": "boom", "exc_info": (ei.type, ei.value, ei.tb)})
    line = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in line["exc"]


def test_configure_logging_is_idempotent(restore_root: None) -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1
