"""Runtime settings from environment variables (with an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """A setting has a value that the package cannot use."""


def load_dotenv(path: str | os.PathLike = ".env") -> int:
    """Read KEY=VALUE lines into os.environ without replacing variables that are already set.

    Returns the number of variables that were added. A missing file is not an error.
    """
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ and value:
            os.environ[key] = value
            added += 1
    return added


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifact_dir: Path
    seed: int
    n_splits: int
    reference_year: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            data_dir=Path((env.get("SALESCOPE_DATA_DIR") or "data").strip()),
            artifact_dir=Path((env.get("SALESCOPE_ARTIFACT_DIR") or "artifacts").strip()),
            seed=_int(env, "SALESCOPE_SEED", 42, 0, 2**31 - 1),
            n_splits=_int(env, "SALESCOPE_N_SPLITS", 5, 2, 50),
            reference_year=_int(env, "SALESCOPE_REFERENCE_YEAR", 2013, 1950, 2100),
        )

    @property
    def train_file(self) -> Path:
        return self.data_dir / "retail_mart_train.csv"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "retail_mart_test.csv"
