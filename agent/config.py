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


# OpenAI-compatible providers and the env vars that hold their keys (auto-detection order).
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = "auto"  # auto | anthropic | gemini | mistral | cerebras | groq | openrouter | offline
    anthropic_model: str = "claude-opus-5-5"
    llm_model: str = ""  # model for OpenAI-compatible providers (blank = provider default)
    llm_base_url: str = ""  # override the provider's endpoint (blank = preset)
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
    def resolved_provider(self) -> str:
        """auto -> Claude if an Anthropic key is set, else the first OpenAI-compatible provider with a key."""
        if self.llm_provider == "offline" or self.llm_provider == "anthropic" or self.llm_provider in PROVIDER_KEYS:
            return self.llm_provider
        if self.has_anthropic_credentials:
            return "anthropic"
        for provider, envs in PROVIDER_KEYS.items():
            if any(os.environ.get(e) for e in envs):
                return provider
        return "offline"

    @property
    def use_llm(self) -> bool:
        return self.resolved_provider != "offline"


def load_settings() -> Settings:
    return Settings(
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-opus-5-5"),
        llm_model=_env("LLM_MODEL") or _env("GEMINI_MODEL"),
        llm_base_url=_env("LLM_BASE_URL"),
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
