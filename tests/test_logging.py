"""Structured logs: tool name, status, duration, request id - and never secrets."""

from __future__ import annotations

import json
import logging

import pytest

from agent.logging_utils import REDACTED, JsonFormatter, log_event, redact, request_id_var


@pytest.mark.parametrize(
    "key", ["password", "AIRFLOW_PASSWORD", "api_key", "apiKey", "Authorization", "token", "secret"]
)
def test_sensitive_keys_masked(key):
    assert redact({key: "value"})[key] == REDACTED


@pytest.mark.parametrize(
    "value",
    [
        "key sk-ant-api03-abcdefghijklmnop",
        "Authorization: Bearer abcdefghijklmnop123",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhaXJmbG93In0.c2lnbmF0dXJlMTIz",
        "connecting with password=hunter2",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
    ],
)
def test_secret_values_masked_in_strings(value):
    assert REDACTED in redact(value)


def test_redaction_is_recursive_and_preserves_safe_values():
    data = {"outer": [{"token": "x", "replicas": 3}], "msg": "ok"}
    assert redact(data) == {"outer": [{"token": REDACTED, "replicas": 3}], "msg": "ok"}


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.setFormatter(JsonFormatter())

    def emit(self, record):
        self.records.append(json.loads(self.format(record)))


@pytest.fixture
def captured():
    handler = _Capture()
    logger = logging.getLogger("mlops_agent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.records
    logger.removeHandler(handler)


def test_tool_call_log_record_has_required_fields(make_executor, captured, healthy_k8s):
    token = request_id_var.set("req_test123")
    try:
        make_executor({"kubernetes": healthy_k8s}).execute("k8s_list_workloads", {"namespace": "mlops-agent"})
    finally:
        request_id_var.reset(token)
    rec = next(r for r in captured if r["event"] == "tool_call")
    assert rec["tool"] == "k8s_list_workloads"
    assert rec["status"] == "ok"
    assert rec["request_id"] == "req_test123"
    assert isinstance(rec["duration_ms"], float)


def test_logs_never_contain_secrets(captured):
    logger = logging.getLogger("mlops_agent.test")
    log_event(logger, "connect", password="hunter2", url="http://x", header="Bearer abcdefghijklmnopqrst")
    dumped = json.dumps(captured)
    assert "hunter2" not in dumped and "abcdefghijklmnopqrst" not in dumped


@pytest.fixture
def healthy_k8s():
    from tests.fakes import healthy_cluster

    return healthy_cluster()
