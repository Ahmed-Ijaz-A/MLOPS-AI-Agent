"""Safety controls enforced in code (never delegated to the LLM).

* ``PolicyError``            - raised by tools when a request violates policy -> status ``denied``.
* ``ConfirmationStore``      - one-time, expiring confirmation tokens bound to a SHA-256 fingerprint of
                               (tool, canonical arguments). The model can *request* an action but only a
                               human decision through the API/CLI can execute it, and the arguments that
                               get executed are exactly the ones the human saw.
* ``AuditLog``               - append-only JSONL record of every requested / approved / rejected action.
* ``check_*`` helpers        - namespace, DAG, conf-key and command allowlists.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent.logging_utils import get_logger, log_event, redact
from agent.schemas import PendingAction, PermissionLevel

logger = get_logger("safety")


class PolicyError(Exception):
    """A request was refused by policy. Mapped to ToolStatus.DENIED."""


class ServiceUnavailable(Exception):
    """Backing service is unreachable or not configured. Mapped to ToolStatus.UNAVAILABLE."""


class ToolInputError(Exception):
    """Arguments are syntactically valid but semantically wrong. Mapped to INVALID_REQUEST."""


# ---------------------------------------------------------------- allowlists


def check_namespace(namespace: str, policy: dict[str, Any]) -> str:
    allowed = policy.get("kubernetes", {}).get("allowed_namespaces", [])
    if namespace not in allowed:
        raise PolicyError(f"Namespace '{namespace}' is not approved for agent access. Approved namespaces: {allowed}.")
    return namespace


def check_mutable_deployment(name: str, policy: dict[str, Any]) -> str:
    allowed = policy.get("kubernetes", {}).get("mutable_deployments", [])
    if name not in allowed:
        raise PolicyError(f"Deployment '{name}' is not on the agent's mutable list {allowed}.")
    return name


def check_replicas(replicas: int, policy: dict[str, Any]) -> int:
    bounds = policy.get("kubernetes", {}).get("scale_bounds", {"min_replicas": 1, "max_replicas": 5})
    lo, hi = int(bounds["min_replicas"]), int(bounds["max_replicas"])
    if not lo <= replicas <= hi:
        raise PolicyError(f"Replica count {replicas} is outside the approved range [{lo}, {hi}].")
    return replicas


def check_triggerable_dag(dag_id: str, policy: dict[str, Any]) -> str:
    allowed = policy.get("airflow", {}).get("triggerable_dags", [])
    if dag_id not in allowed:
        raise PolicyError(f"DAG '{dag_id}' is not approved for triggering. Approved: {allowed}.")
    return dag_id


def check_conf(conf: dict[str, Any] | None, policy: dict[str, Any]) -> dict[str, Any]:
    conf = conf or {}
    allowed = set(policy.get("airflow", {}).get("allowed_conf_keys", []))
    extra = sorted(set(conf) - allowed)
    if extra:
        raise PolicyError(f"DAG conf keys {extra} are not allowed. Allowed keys: {sorted(allowed)}.")
    for key, value in conf.items():
        if not isinstance(value, str | int | float | bool) or (isinstance(value, str) and len(value) > 200):
            raise PolicyError(f"DAG conf value for '{key}' must be a short scalar.")
    return conf


def check_command(tool: str, args: list[str], policy: dict[str, Any]) -> list[str]:
    """Allow-list sub-commands for the dvc / git CLIs and reject shell metacharacters."""
    if not args:
        raise PolicyError(f"Empty {tool} command.")
    key = {"dvc": "allowed_dvc_commands", "git": "allowed_git_commands"}.get(tool)
    if key is None:
        raise PolicyError(f"Executable '{tool}' is not allowed.")
    allowed = policy.get("dvc", {}).get(key, [])
    if args[0] not in allowed:
        raise PolicyError(f"'{tool} {args[0]}' is not an approved operation. Approved: {allowed}.")
    for a in args:
        if any(ch in a for ch in [";", "|", "&", "`", "$(", ">", "<", "\n"]):
            raise PolicyError(f"Argument {a!r} contains disallowed characters.")
    return args


# ---------------------------------------------------------------- confirmations


def fingerprint(tool: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": tool, "args": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> None:
        entry = {"ts": datetime.now(UTC).isoformat(), "event": event, **redact(fields)}
        with self._lock:
            self.events.append(entry)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, default=str) + "\n")
                except OSError:
                    pass
        log_event(logger, f"audit.{event}", **fields)


class ConfirmationStore:
    def __init__(self, ttl_seconds: int = 600, audit: AuditLog | None = None) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self.audit = audit or AuditLog(None)
        self._pending: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def create(
        self,
        tool: str,
        arguments: dict[str, Any],
        description: str,
        permission: PermissionLevel,
        session_id: str | None = None,
    ) -> PendingAction:
        action = PendingAction(
            confirmation_id=f"cfm_{secrets.token_urlsafe(9)}",
            tool=tool,
            arguments=arguments,
            description=description,
            permission=permission,
            fingerprint=fingerprint(tool, arguments),
            expires_at=datetime.now(UTC) + self.ttl,
            session_id=session_id,
        )
        with self._lock:
            self._purge()
            self._pending[action.confirmation_id] = action
        self.audit.record(
            "action_requested",
            confirmation_id=action.confirmation_id,
            tool=tool,
            arguments=arguments,
            session_id=session_id,
        )
        return action

    def get(self, confirmation_id: str) -> PendingAction | None:
        with self._lock:
            self._purge()
            return self._pending.get(confirmation_id)

    def list(self, session_id: str | None = None) -> list[PendingAction]:
        with self._lock:
            self._purge()
            return [a for a in self._pending.values() if session_id is None or a.session_id == session_id]

    def consume(self, confirmation_id: str, session_id: str | None = None) -> PendingAction:
        """Pop a pending action exactly once. Raises PolicyError if missing/expired/tampered."""
        with self._lock:
            self._purge()
            action = self._pending.pop(confirmation_id, None)
        if action is None:
            self.audit.record("action_confirm_failed", confirmation_id=confirmation_id, reason="unknown_or_expired")
            raise PolicyError("Confirmation id is unknown, already used, or expired.")
        if action.session_id and session_id and action.session_id != session_id:
            self.audit.record("action_confirm_failed", confirmation_id=confirmation_id, reason="session_mismatch")
            raise PolicyError("Confirmation belongs to a different session.")
        if fingerprint(action.tool, action.arguments) != action.fingerprint:
            raise PolicyError("Pending action was modified after it was requested.")
        return action

    def _purge(self) -> None:
        now = datetime.now(UTC)
        for cid in [c for c, a in self._pending.items() if a.expires_at <= now]:
            expired = self._pending.pop(cid)
            self.audit.record("action_expired", confirmation_id=cid, tool=expired.tool)
