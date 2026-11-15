"""Tool modules. ``build_registry`` assembles every tool; ``build_executor`` wires clients + policy."""

from __future__ import annotations

from agent.config import Settings, get_policy, get_settings
from agent.safety import ConfirmationStore
from agent.tools import airflow_tools, dvc_tools, insight_tools, kubernetes_tools, mlflow_tools
from agent.tools.base import ToolContext, ToolExecutor, ToolRegistry

SYSTEMS = ("dvc", "mlflow", "airflow", "kubernetes")


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for module in (dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools, insight_tools):
        module.register(reg)
    return reg


def default_factories() -> dict:
    return {
        "dvc": dvc_tools.dvc_factory,
        "mlflow": mlflow_tools.mlflow_factory,
        "airflow": airflow_tools.airflow_factory,
        "kubernetes": kubernetes_tools.k8s_factory,
    }


def build_executor(
    settings: Settings | None = None,
    clients: dict | None = None,
    policy: dict | None = None,
    confirmations: ConfirmationStore | None = None,
    registry: ToolRegistry | None = None,
    factories: dict | None = None,
) -> ToolExecutor:
    settings = settings or get_settings()
    ctx = ToolContext(
        settings=settings,
        policy=policy if policy is not None else get_policy(settings),
        clients=dict(clients or {}),
        factories=default_factories() if factories is None else factories,
    )
    return ToolExecutor(registry or build_registry(), ctx, confirmations)
