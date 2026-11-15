"""Typed contracts shared by tools, the orchestrator and the HTTP API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # the tool ran but the operation failed
    UNAVAILABLE = "unavailable"  # backing service unreachable / not configured
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"  # bad / unknown arguments or unknown tool
    DENIED = "denied"  # blocked by policy (namespace, allowlist, bounds)
    REQUIRES_CONFIRMATION = "requires_confirmation"  # not executed; waiting for a human


class PermissionLevel(StrEnum):
    READ = "read"  # no side effects
    OPERATE = "operate"  # starts workflows (e.g. trigger a DAG)
    MUTATE = "mutate"  # changes cluster state


class ToolResult(BaseModel):
    """Envelope returned by every tool call. ``evidence_id`` lets the answer cite its sources."""

    model_config = ConfigDict(use_enum_values=True)

    evidence_id: str
    tool: str
    status: ToolStatus
    system: str = ""
    summary: str = ""
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmation: PendingAction | None = None

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK

    def for_llm(self, max_chars: int = 9000) -> str:
        """Compact JSON for the model; truncated so one huge log cannot blow the context."""
        body = self.model_dump(exclude={"duration_ms", "arguments"}, exclude_none=True)
        text = _json(body)
        if len(text) > max_chars:
            text = text[: max_chars - 60] + f'... [truncated {len(text) - max_chars} chars]"}}'
        return text


class PendingAction(BaseModel):
    confirmation_id: str
    tool: str
    arguments: dict[str, Any]
    description: str
    permission: PermissionLevel
    fingerprint: str
    expires_at: datetime
    session_id: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class ToolCallTrace(BaseModel):
    evidence_id: str
    tool: str
    status: str
    duration_ms: float
    arguments: dict[str, Any]
    summary: str
    error: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


class ConfirmationDecision(BaseModel):
    approve: bool
    session_id: str | None = None


class GroundingReport(BaseModel):
    grounded: bool
    checked_values: int = 0
    unsupported_values: list[str] = Field(default_factory=list)
    note: str = ""


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    mode: str  # "llm" | "offline"
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    pending_confirmations: list[PendingAction] = Field(default_factory=list)
    systems_consulted: list[str] = Field(default_factory=list)
    grounding: GroundingReport | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str, ensure_ascii=False)


ToolResult.model_rebuild()
