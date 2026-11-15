from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

# Hermetic test environment: no LLM, no real services, no writes to the repo's logs/.
os.environ["LLM_PROVIDER"] = "offline"
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("AGENT_API_KEY", None)
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"

from agent.config import PROJECT_ROOT, PROVIDER_KEYS, get_policy, load_settings

# agent.config loads .env on import - strip every LLM credential so tests never depend on the developer's keys.
for _envs in PROVIDER_KEYS.values():
    for _var in _envs:
        os.environ.pop(_var, None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
from agent.safety import ServiceUnavailable
from agent.tools import build_executor
from tests.fakes import synthetic_credit_frame


@pytest.fixture
def settings(tmp_path):
    return replace(
        load_settings(),
        log_dir=tmp_path / "logs",
        repo_root=PROJECT_ROOT,
        tool_timeout_s=5.0,
        airflow_password="pw",
        airflow_base_url="http://airflow.test",
    )


@pytest.fixture
def policy(settings):
    return get_policy(settings)


def _unavailable(name):
    def factory(ctx):
        raise ServiceUnavailable(f"{name} is not reachable (test)")

    return factory


@pytest.fixture
def make_executor(settings, policy):
    """Executor whose clients are fakes; any client not supplied is 'unavailable'."""

    def _make(clients=None, settings_override=None, **kwargs):
        s = settings_override or settings
        factories = {n: _unavailable(n) for n in ("dvc", "mlflow", "airflow", "kubernetes")}
        return build_executor(s, clients=clients or {}, policy=policy, factories=factories, **kwargs)

    return _make


@pytest.fixture
def credit_csv(tmp_path):
    path = tmp_path / "credit.csv"
    synthetic_credit_frame(400, seed=1).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- real git + DVC repository


def _run(cwd: Path, *cmd: str) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.fixture(scope="session")
def dvc_repo(tmp_path_factory):
    """A real git repo with DVC, a local remote and two tagged dataset versions (v1.0, v2.0)."""
    repo = tmp_path_factory.mktemp("dvcrepo")
    git = lambda *a: _run(repo, "git", *a)  # noqa: E731
    dvc = lambda *a: _run(repo, sys.executable, "-m", "dvc", *a)  # noqa: E731
    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    dvc("init", "-q")
    dvc("remote", "add", "-d", "local", str(tmp_path_factory.mktemp("remote")))
    (repo / "params.yaml").write_text("dataset:\n  version: v1.0\n", encoding="utf-8")
    data = repo / "data" / "raw"
    data.mkdir(parents=True)
    full = synthetic_credit_frame(300, seed=3)
    full.head(200).to_csv(data / "credit_risk.csv", index=False, lineterminator="\n")
    dvc("add", "-q", "data/raw/credit_risk.csv")
    git("add", "-A")
    git("commit", "-q", "-m", "data v1.0")
    git("tag", "-a", "data-v1.0", "-m", "v1.0")
    dvc("push", "-q")
    full.to_csv(data / "credit_risk.csv", index=False, lineterminator="\n")
    (repo / "params.yaml").write_text("dataset:\n  version: v2.0\n", encoding="utf-8")
    dvc("add", "-q", "data/raw/credit_risk.csv")
    git("add", "-A")
    git("commit", "-q", "-m", "data v2.0")
    git("tag", "-a", "data-v2.0", "-m", "v2.0")
    dvc("push", "-q")
    return repo
