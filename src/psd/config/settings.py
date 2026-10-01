"""Environment-backed configuration settings.

PSD deliberately keeps machine-specific configuration to a single typed
boundary: ``PSD_DATA_ROOT``. Settings are cached so that repeated CLI and test
access does not re-read the environment, and the cache can be reset in tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from psd.paths import DATA_ROOT_ENV_VAR

__all__ = ("PsdSettings", "get_settings", "reset_settings_cache")


class PsdSettings(BaseSettings):
    """Validated PSD environment settings.

    Attributes:
        data_root: Absolute path to the external PSD data root. All datasets
            and run artifacts resolve beneath this path; the Git repository
            never stores them.
    """

    model_config = SettingsConfigDict(
        env_prefix="PSD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    data_root: Path = Field(
        validation_alias=DATA_ROOT_ENV_VAR,
        description="Absolute path to the external PSD data root.",
    )


@lru_cache(maxsize=1)
def _cached_settings() -> PsdSettings:
    return PsdSettings()  # type: ignore[call-arg]


def get_settings() -> PsdSettings:
    """Return the validated settings, reading the environment once."""
    return _cached_settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that mutate the environment)."""
    _cached_settings.cache_clear()
