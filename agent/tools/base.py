"""Tool registry + executor.

The executor is the single choke point every tool call passes through, whether it comes from the
LLM, the offline planner or a human confirmation. It is responsible for:

1. argument validation against the tool's Pydantic schema   -> ``invalid_request``
2. policy pre-checks (namespace / allowlists / bounds)        -> ``denied``
3. human-confirmation gating for operate/mutate tools         -> ``requires_confirmation``
4. hard timeouts                                              -> ``timeout``
5. error classification (service down vs. operation failed)   -> ``unavailable`` / ``error``
6. structured logging of tool name, status, duration, request id (secrets redacted)
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.config import Settings, get_policy, get_settings
from agent.logging_utils import get_logger, log_event, redact, request_id_var
from agent.safety import (
    AuditLog,
    ConfirmationStore,
    PolicyError,
    ServiceUnavailable,
    ToolInputError,
)
from agent.schemas import PendingAction, PermissionLevel, ToolResult, ToolStatus

logger = get_logger("tools")

_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


@dataclass
class ToolOutput:
    summary: str
    data: Any = None


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[[Any, ToolContext], ToolOutput]
    input_model: type[BaseModel]
    system: str  # dvc | mlflow | airflow | kubernetes | insight
    permission: PermissionLevel = PermissionLevel.READ
    timeout_s: float | None = None
    precheck: Callable[[Any, ToolContext], None] | None = None
    describe_action: Callable[[Any], str] | None = None

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        schema["type"] = "object"
        return schema

    def anthropic_definition(self) -> dict[str, Any]:
        perm = ""
        if self.permission != PermissionLevel.READ:
            perm = (
                f" [PERMISSION: {self.permission.value} - this tool does NOT run immediately; it creates a "
                "pending action that a human must approve.]"
            )
        return {"name": self.name, "description": self.description + perm, "input_schema": self.json_schema()}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        system: str,
        permission: PermissionLevel = PermissionLevel.READ,
        timeout_s: float | None = None,
        precheck: Callable[[Any, ToolContext], None] | None = None,
        describe_action: Callable[[Any], str] | None = None,
    ) -> Callable[[Callable[..., ToolOutput]], Callable[..., ToolOutput]]:
        def deco(func: Callable[..., ToolOutput]) -> Callable[..., ToolOutput]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    func=func,
                    input_model=input_model,
                    system=system,
                    permission=permission,
                    timeout_s=timeout_s,
                    precheck=precheck,
                    describe_action=describe_action,
                )
            )
            return func

        return deco

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [spec.anthropic_definition() for spec in self._tools.values()]

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "system": s.system,
                "permission": s.permission.value,
                "description": s.description,
                "input_schema": s.json_schema(),
            }
            for s in self._tools.values()
        ]


@dataclass
class ToolContext:
    """Dependencies available to tools. Clients are created lazily and can be injected in tests."""

    settings: Settings
    policy: dict[str, Any]
    clients: dict[str, Any] = field(default_factory=dict)
    factories: dict[str, Callable[[ToolContext], Any]] = field(default_factory=dict)
    executor: ToolExecutor | None = None

    def client(self, name: str) -> Any:
        if name not in self.clients:
            factory = self.factories.get(name)
            if factory is None:
                raise ServiceUnavailable(f"No client configured for '{name}'.")
            self.clients[name] = factory(self)
        return self.clients[name]


@dataclass
class RequestTrace:
    request_id: str
    session_id: str | None = None
    results: list[ToolResult] = field(default_factory=list)
    _counter: int = 0

    def next_evidence_id(self) -> str:
        self._counter += 1
        return f"E{self._counter}"

    @property
    def systems(self) -> list[str]:
        return sorted({r.system for r in self.results if r.system and r.system != "unknown"})


_UNAVAILABLE_HINTS = (
    "connection refused",
    "max retries exceeded",
    "failed to establish a new connection",
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo failed",
    "no connection could be made",
    "connecterror",
    "service unavailable",
    "invalid kube-config",
    "kubeconfig",
    "connection aborted",
    "connection reset",
    "remote end closed",
)


def classify_exception(exc: BaseException) -> ToolStatus:
    if isinstance(exc, ValidationError | ToolInputError):
        return ToolStatus.INVALID_REQUEST
    if isinstance(exc, PolicyError):
        return ToolStatus.DENIED
    if isinstance(exc, ServiceUnavailable | ConnectionError):
        return ToolStatus.UNAVAILABLE
    if isinstance(exc, TimeoutError | cf.TimeoutError):
        return ToolStatus.TIMEOUT
    name = type(exc).__name__.lower()
    text = f"{name}: {exc}".lower()
    if "timeout" in name or "timed out" in text:
        return ToolStatus.TIMEOUT
    if any(h in text for h in _UNAVAILABLE_HINTS):
        return ToolStatus.UNAVAILABLE
    return ToolStatus.ERROR


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        confirmations: ConfirmationStore | None = None,
    ) -> None:
        self.registry = registry
        self.context = context
        context.executor = self
        policy = context.policy
        perms = policy.get("permissions", {})
        self.confirmation_levels = set(perms.get("confirmation_required_levels", ["operate", "mutate"]))
        self.confirmations = confirmations or ConfirmationStore(
            ttl_seconds=int(perms.get("confirmation_ttl_seconds", 600)),
            audit=AuditLog(context.settings.log_dir / "audit.jsonl"),
        )

    # ------------------------------------------------------------ public API

    def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | None,
        trace: RequestTrace | None = None,
        *,
        confirmed: bool = False,
    ) -> ToolResult:
        trace = trace or RequestTrace(request_id=request_id_var.get())
        evidence_id = trace.next_evidence_id()
        raw_args = dict(raw_args or {})
        started = time.perf_counter()
        spec = self.registry.get(name)

        if spec is None:
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=ToolStatus.INVALID_REQUEST,
                error=f"Unknown tool '{name}'. Available: {', '.join(self.registry.names())}",
                arguments=raw_args,
            )
            return self._finish(result, trace, started, system="unknown")

        try:
            args = spec.input_model.model_validate(raw_args)
        except ValidationError as exc:
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=ToolStatus.INVALID_REQUEST,
                error=_short_validation_error(exc),
                arguments=raw_args,
            )
            return self._finish(result, trace, started, system=spec.system)

        clean_args = args.model_dump(mode="json")

        # Policy pre-checks run before anything else so forbidden actions are denied outright,
        # never offered for confirmation.
        if spec.precheck is not None:
            try:
                spec.precheck(args, self.context)
            except Exception as exc:
                status = classify_exception(exc)
                result = ToolResult(
                    evidence_id=evidence_id, tool=name, status=status, error=str(exc), arguments=clean_args
                )
                return self._finish(result, trace, started, system=spec.system)

        if spec.permission.value in self.confirmation_levels and not confirmed:
            description = spec.describe_action(args) if spec.describe_action else f"Run {name}({clean_args})"
            pending = self.confirmations.create(
                name, clean_args, description, spec.permission, session_id=trace.session_id
            )
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=ToolStatus.REQUIRES_CONFIRMATION,
                summary=(
                    f"NOT EXECUTED. '{description}' requires explicit human approval "
                    f"(confirmation id {pending.confirmation_id}, expires {pending.expires_at:%H:%M:%S} UTC)."
                ),
                arguments=clean_args,
                confirmation=pending,
            )
            return self._finish(result, trace, started, system=spec.system)

        timeout = spec.timeout_s or self.context.settings.tool_timeout_s
        future = _POOL.submit(_run_with_request_id, request_id_var.get(), spec.func, args, self.context)
        try:
            output = future.result(timeout=timeout)
            if not isinstance(output, ToolOutput):
                raise TypeError(f"tool {name} returned {type(output).__name__}, expected ToolOutput")
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=ToolStatus.OK,
                summary=output.summary,
                data=redact(output.data),
                arguments=clean_args,
            )
        except cf.TimeoutError:
            future.cancel()
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=ToolStatus.TIMEOUT,
                error=f"{name} did not respond within {timeout:.0f}s.",
                arguments=clean_args,
            )
        except Exception as exc:
            status = classify_exception(exc)
            result = ToolResult(
                evidence_id=evidence_id,
                tool=name,
                status=status,
                error=redact(f"{type(exc).__name__}: {exc}")[:1500],
                arguments=clean_args,
            )
        return self._finish(result, trace, started, system=spec.system)

    def execute_confirmed(
        self, confirmation_id: str, trace: RequestTrace, *, approve: bool = True
    ) -> tuple[PendingAction, ToolResult | None]:
        """Execute (or reject) a pending action after a human decision."""
        action = self.confirmations.consume(confirmation_id, session_id=trace.session_id)
        if not approve:
            self.confirmations.audit.record(
                "action_rejected", confirmation_id=confirmation_id, tool=action.tool, arguments=action.arguments
            )
            return action, None
        self.confirmations.audit.record(
            "action_approved", confirmation_id=confirmation_id, tool=action.tool, arguments=action.arguments
        )
        result = self.execute(action.tool, action.arguments, trace, confirmed=True)
        self.confirmations.audit.record(
            "action_executed",
            confirmation_id=confirmation_id,
            tool=action.tool,
            status=result.status,
            error=result.error,
        )
        return action, result

    # ------------------------------------------------------------ helpers

    def _finish(self, result: ToolResult, trace: RequestTrace, started: float, system: str) -> ToolResult:
        result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        result.system = system
        trace.results.append(result)
        log_event(
            logger,
            "tool_call",
            tool=result.tool,
            system=system,
            status=result.status,
            duration_ms=result.duration_ms,
            evidence_id=result.evidence_id,
            arguments=result.arguments,
            error=result.error,
            session_id=trace.session_id,
        )
        return result


def _run_with_request_id(request_id: str, func: Callable, args: Any, ctx: ToolContext) -> Any:
    token = request_id_var.set(request_id)
    try:
        return func(args, ctx)
    finally:
        request_id_var.reset(token)


def _short_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg')}")
    return "Invalid arguments - " + "; ".join(parts)


def default_context(settings: Settings | None = None) -> ToolContext:
    settings = settings or get_settings()
    return ToolContext(settings=settings, policy=get_policy(settings))
