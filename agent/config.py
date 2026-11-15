"""Runtime configuration.

Secrets (API keys, Airflow password, kubeconfig) only ever come from the environment / .env file,
never from code or the policy file. Non-secret policy lives in ``config/agent_policy.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # optional in production images
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = "auto"  # auto | anthropic | offline
    anthropic_model: str = "claude-opus-5-5"
    agent_effort: str = "medium"
    anthropic_fallbacks: bool = True
    max_agent_steps: int = 12

    # Integrations
    repo_root: Path = PROJECT_ROOT
    mlflow_tracking_uri: str = "http://localhost:5000"
    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = "airflow"
    airflow_password: str = field(default="", repr=False)
    kube_context: str = ""
    tool_timeout_s: float = 20.0

    # Logging
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = "INFO"

    policy_path: Path = PROJECT_ROOT / "config" / "agent_policy.yaml"

    @property
    def has_anthropic_credentials(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def use_llm(self) -> bool:
        if self.llm_provider == "offline":
            return False
        if self.llm_provider == "anthropic":
            return True
        return self.has_anthropic_credentials


def load_settings() -> Settings:
    return Settings(
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-opus-5-5"),
        agent_effort=_env("AGENT_EFFORT", "medium"),
        anthropic_fallbacks=_env("ANTHROPIC_FALLBACKS", "true").lower() in {"1", "true", "yes"},
        max_agent_steps=int(_env("MAX_AGENT_STEPS", "12")),
        repo_root=Path(_env("REPO_ROOT", str(PROJECT_ROOT))),
        mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        airflow_base_url=_env("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/"),
        airflow_username=_env("AIRFLOW_USERNAME", "airflow"),
        airflow_password=_env("AIRFLOW_PASSWORD", ""),
        kube_context=_env("KUBE_CONTEXT", ""),
        tool_timeout_s=float(_env("TOOL_TIMEOUT_S", "20")),
        log_dir=Path(_env("LOG_DIR", str(PROJECT_ROOT / "logs"))),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        policy_path=Path(_env("AGENT_POLICY_PATH", str(PROJECT_ROOT / "config" / "agent_policy.yaml"))),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=4)
def _load_policy(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_policy(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return _load_policy(str(settings.policy_path))
