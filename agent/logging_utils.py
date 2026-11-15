"""Structured JSON logging with request correlation and secret redaction.

Every tool invocation emits one ``tool_call`` record containing: request_id, tool name, status,
duration_ms and redacted arguments. Records go to stdout and to ``logs/agent.jsonl``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SENSITIVE_KEY = re.compile(
    r"(passw(or)?d|^pass$|secret|(?<![a-z])token(?!s)|api[_-]?key|authorization|credential|kubeconfig)", re.I
)
_SENSITIVE_VALUE = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),  # bearer tokens
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWTs
    re.compile(r"(?i)(password|passwd|pwd)=\S+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
]
REDACTED = "***REDACTED***"


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively mask secrets in dicts/lists/strings. Keys that look sensitive are fully masked."""
    if _depth > 8:
        return value
    if isinstance(value, dict):
        return {
            k: (REDACTED if isinstance(k, str) and _SENSITIVE_KEY.search(k) else redact(v, _depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str):
        out = value
        for pattern in _SENSITIVE_VALUE:
            out = pattern.sub(REDACTED, out)
        return out
    return value


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "event": getattr(record, "event", None) or record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(redact(fields))
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("mlops_agent")
    root.setLevel(level)
    root.propagate = False
    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "agent.jsonl", encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError:  # read-only filesystem in a container: stdout only
            pass
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mlops_agent.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"event": event, "fields": fields})
