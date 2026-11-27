from collections.abc import Iterator

import pytest

from ds_prep.config import get_settings
from ds_prep.errors import ConfigError


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_env_overrides_and_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSP_ENV", "ci")
    monkeypatch.setenv("DSP_WAREHOUSE__PORT", "6543")
    s = get_settings()
    assert s.env == "ci"
    assert s.warehouse.port == 6543


def test_bad_value_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSP_LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigError) as ei:
        get_settings()
    assert ei.value.context["fields"] == ["log_level"]
    assert ei.value.exit_code == 78


def test_password_never_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSP_WAREHOUSE__PASSWORD", "s3cret")
    s = get_settings()
    assert "s3cret" not in repr(s)
    assert s.warehouse.password.get_secret_value() == "s3cret"
