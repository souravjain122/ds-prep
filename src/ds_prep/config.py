"""Typed, validated configuration from env vars / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ds_prep.errors import ConfigError


class WarehouseSettings(BaseModel):
    """Connection settings for the warehouse."""

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "analyst"
    password: SecretStr = SecretStr("")


class Settings(BaseSettings):
    """All runtime settings for ds_prep."""

    model_config = SettingsConfigDict(
        env_prefix="DSP_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
        frozen=True,
    )

    env: Literal["dev", "ci", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    data_dir: Path = Path("data")
    min_orders_per_store: int = Field(default=30, ge=1)
    warehouse: WarehouseSettings = WarehouseSettings()

    @field_validator("data_dir")
    @classmethod
    def _not_in_tmp(cls, v: Path) -> Path:
        if v.is_absolute() and v.parts[1:2] == ("tmp",):
            raise ValueError("data_dir must not live in /tmp")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process; tests call get_settings.cache_clear()."""
    try:
        return Settings()
    except ValidationError as exc:
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
        raise ConfigError("invalid settings", fields=fields) from exc
