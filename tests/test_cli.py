from collections.abc import Iterator

import pytest

from ds_prep.cli import main
from ds_prep.config import get_settings


@pytest.fixture(autouse=True)
def _clear_cache(restore_root: None) -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_happy_path_exits_0() -> None:
    assert main([]) == 0


def test_corrupt_data_exits_65() -> None:
    assert main(["--corrupt"]) == 65


def test_bad_config_exits_78(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSP_WAREHOUSE__PORT", "99999")
    assert main([]) == 78
