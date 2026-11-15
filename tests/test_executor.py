"""The executor: validation, timeouts, error taxonomy, confirmation gating, tracing."""

from __future__ import annotations

import re
import time

import pytest
from pydantic import BaseModel, ConfigDict

from agent.safety import PolicyError, ServiceUnavailable
from agent.schemas import PermissionLevel
from agent.tools import build_registry
from agent.tools.base import RequestTrace, ToolOutput, ToolRegistry, classify_exception


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = 1


def _registry(func, permission=PermissionLevel.READ, timeout_s=None, precheck=None):
    reg = ToolRegistry()
    reg.tool("probe", "test tool", Args, system="test", permission=permission, timeout_s=timeout_s, precheck=precheck)(
        func
    )
    return reg


def test_unknown_tool_is_invalid_request(make_executor):
    r = make_executor().execute("drop_database", {})
    assert r.status == "invalid_request" and "Unknown tool" in r.error


@pytest.mark.parametrize("args", [{"n": "abc"}, {"n": 1, "evil": True}, {"n": [1]}])
def test_invalid_arguments_rejected_before_execution(make_executor, args):
    called = []
    ex = make_executor(registry=_registry(lambda a, c: called.append(1) or ToolOutput("x")))
    r = ex.execute("probe", args)
    assert r.status == "invalid_request" and not called


def test_tool_timeout_reported(make_executor):
    ex = make_executor(registry=_registry(lambda a, c: time.sleep(2) or ToolOutput("late"), timeout_s=0.2))
    started = time.perf_counter()
    r = ex.execute("probe", {})
    assert r.status == "timeout" and "did not respond" in r.error
    assert time.perf_counter() - started < 1.5


@pytest.mark.parametrize(
    "exc,status",
    [
        (ServiceUnavailable("down"), "unavailable"),
        (ConnectionError("refused"), "unavailable"),
        (RuntimeError("Max retries exceeded with url: /api"), "unavailable"),
        (TimeoutError("slow"), "timeout"),
        (PolicyError("nope"), "denied"),
        (ValueError("bad math"), "error"),
    ],
)
def test_exceptions_mapped_to_statuses(make_executor, exc, status):
    def boom(a, c):
        raise exc

    r = make_executor(registry=_registry(boom)).execute("probe", {})
    assert r.status == status and r.data is None


@pytest.mark.parametrize(
    "message,status",
    [
        ("HTTPConnectionPool: Failed to establish a new connection", "unavailable"),
        ("[WinError 10061] No connection could be made", "unavailable"),
        ("Read timed out", "timeout"),
        ("KeyError: 'f1'", "error"),
    ],
)
def test_classify_exception_by_message(message, status):
    assert classify_exception(RuntimeError(message)) == status


def test_mutating_tool_requires_confirmation_and_does_not_run(make_executor):
    called = []
    ex = make_executor(
        registry=_registry(lambda a, c: called.append(1) or ToolOutput("done"), permission=PermissionLevel.MUTATE)
    )
    r = ex.execute("probe", {"n": 2})
    assert r.status == "requires_confirmation" and not called
    assert r.confirmation is not None and r.confirmation.arguments == {"n": 2}
    assert "NOT EXECUTED" in r.summary


def test_confirmed_action_executes_with_exact_arguments(make_executor):
    seen = []
    ex = make_executor(
        registry=_registry(lambda a, c: seen.append(a.n) or ToolOutput("done"), permission=PermissionLevel.MUTATE)
    )
    pending = ex.execute("probe", {"n": 3}).confirmation
    _, result = ex.execute_confirmed(pending.confirmation_id, RequestTrace("req_x"))
    assert result.status == "ok" and seen == [3]


def test_declined_action_never_executes(make_executor):
    called = []
    ex = make_executor(
        registry=_registry(lambda a, c: called.append(1) or ToolOutput("done"), permission=PermissionLevel.MUTATE)
    )
    pending = ex.execute("probe", {}).confirmation
    _, result = ex.execute_confirmed(pending.confirmation_id, RequestTrace("req_x"), approve=False)
    assert result is None and not called
    assert any(e["event"] == "action_rejected" for e in ex.confirmations.audit.events)


def test_policy_precheck_denies_without_offering_confirmation(make_executor):
    def deny(a, c):
        raise PolicyError("namespace not approved")

    ex = make_executor(
        registry=_registry(lambda a, c: ToolOutput("x"), permission=PermissionLevel.MUTATE, precheck=deny)
    )
    r = ex.execute("probe", {})
    assert r.status == "denied" and r.confirmation is None and not ex.confirmations.list()


def test_evidence_ids_increment_per_request(make_executor):
    ex = make_executor(registry=_registry(lambda a, c: ToolOutput("x")))
    trace = RequestTrace("req_1")
    ids = [ex.execute("probe", {}, trace).evidence_id for _ in range(3)]
    assert ids == ["E1", "E2", "E3"] and len(trace.results) == 3


def test_tool_output_is_redacted(make_executor):
    ex = make_executor(registry=_registry(lambda a, c: ToolOutput("x", {"password": "hunter2", "ok": 1})))
    assert ex.execute("probe", {}).data == {"password": "***REDACTED***", "ok": 1}


def test_llm_payload_is_truncated():
    from agent.schemas import ToolResult

    r = ToolResult(evidence_id="E1", tool="t", status="ok", data={"blob": "x" * 50_000})
    assert len(r.for_llm(max_chars=2000)) <= 2000


def test_all_tool_definitions_are_valid_for_the_api():
    reg = build_registry()
    defs = reg.anthropic_tools()
    assert len(defs) >= 30
    for d in defs:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", d["name"])
        assert d["input_schema"]["type"] == "object"
        assert len(d["description"]) > 20


def test_each_module_offers_at_least_three_tools():
    reg = build_registry()
    for system in ("dvc", "mlflow", "airflow", "kubernetes"):
        assert sum(1 for s in reg.specs() if s.system == system) >= 3


def test_write_tools_are_flagged_in_descriptions():
    reg = build_registry()
    writes = {s.name for s in reg.specs() if s.permission != PermissionLevel.READ}
    assert writes == {
        "airflow_trigger_dag",
        "k8s_restart_deployment",
        "k8s_scale_deployment",
        "k8s_rollback_deployment",
    }
    for d in reg.anthropic_tools():
        assert ("PERMISSION" in d["description"]) == (d["name"] in writes)
