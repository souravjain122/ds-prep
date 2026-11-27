import logging
from collections.abc import Iterator

import pytest


@pytest.fixture
def restore_root() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
