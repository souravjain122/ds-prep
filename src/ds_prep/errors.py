"""Error exception hierarchy with exit codes on each subclass."""

from typing import ClassVar


class DsPrepError(Exception):
    """Base Class: every error this package raises on purpose."""

    exit_code: ClassVar[int] = 1
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.context: dict[str, object] = context


class ConfigError(DsPrepError):
    """Wrong settings; retrying won't help"""

    exit_code = 78


class DataValidationError(DsPrepError):
    """Input data broke a contract"""

    exit_code = 65


class ExternalServiceError(DsPrepError):
    """Warehouse or API call failed; worth retrying"""

    exit_code = 69
    retryable = True
