"""Airflow 3 tools against an in-memory REST API (JWT auth, runs, tasks, logs, trigger)."""

from __future__ import annotations

import json

import httpx
import pytest

from agent.tools.airflow_tools import AirflowClient
from agent.tools.base import RequestTrace
from tests.fakes import AirflowMock, pipeline_runs


@pytest.fixture
def mock():
    runs, tasks = pipeline_runs()
    logs = {
        (runs[0]["dag_run_id"], "quality_gate"): [
            {"timestamp": "2026-11-15T09:02:59Z", "level": "info", "event": "evaluating thresholds"},
            {
                "timestamp": "2026-11-15T09:03:00Z",
                "level": "error",
                "event": "QUALITY GATE FAILED for run abc: ['recall=0.5333 < required 0.65']",
            },
            {
                "timestamp": "2026-11-15T09:03:00Z",
                "level": "error",
                "event": "Task failed with exception",
                "error_detail": "AirflowFailException: Quality gate failed: recall=0.5333 < required 0.65",
            },
        ]
    }
    return AirflowMock(runs=runs, tasks=tasks, logs=logs)


def executor(make_executor, mock):
    client = AirflowClient("http://airflow.test", "airflow", "pw", transport=mock.transport())
    return make_executor({"airflow": client})


def test_list_dags(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_list_dags", {})
    assert r.ok and r.data["dags"][0]["dag_id"] == "ml_training_pipeline"


def test_auth_uses_jwt_bearer_token(make_executor, mock):
    executor(make_executor, mock).execute("airflow_list_dags", {})
    token_req, api_req = mock.requests[0], mock.requests[1]
    assert token_req.url.path == "/auth/token" and json.loads(token_req.content)["username"] == "airflow"
    assert api_req.headers["authorization"] == "Bearer tok1"


def test_expired_token_is_refreshed_once(make_executor, mock):
    mock.expire_first_token = True
    r = executor(make_executor, mock).execute("airflow_list_dags", {})
    assert r.ok and mock.tokens_issued == 2


def test_list_runs_counts_states(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_list_dag_runs", {"limit": 5})
    assert r.data["state_counts"] == {"failed": 1, "success": 1}
    assert r.data["runs"][0]["state"] == "failed"


def test_get_latest_run_with_task_states(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_get_dag_run", {"run_id": "latest"})
    states = {t["task_id"]: t["state"] for t in r.data["tasks"]}
    assert states["quality_gate"] == "failed" and states["register_model"] == "upstream_failed"


def test_failed_tasks_identified_and_promotion_blocked(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_get_failed_tasks", {})
    assert [t["task_id"] for t in r.data["failed_tasks"]] == ["quality_gate"]
    assert [t["task_id"] for t in r.data["blocked_tasks"]] == ["register_model"]


def test_task_logs_extract_root_cause(make_executor, mock):
    run_id = mock.runs[0]["dag_run_id"]
    r = executor(make_executor, mock).execute("airflow_get_task_logs", {"run_id": run_id, "task_id": "quality_gate"})
    assert r.ok and "recall=0.5333 < required 0.65" in r.data["root_cause_line"]
    assert any("QUALITY GATE FAILED" in ln for ln in r.data["error_lines"])


def test_trigger_requires_confirmation(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_trigger_dag", {"conf": {"reason": "new data"}})
    assert r.status == "requires_confirmation" and mock.triggered == []


@pytest.mark.parametrize(
    "args",
    [{"dag_id": "some_other_dag"}, {"conf": {"bash_command": "rm -rf /"}}],
)
def test_trigger_policy_violations_denied(make_executor, mock, args):
    r = executor(make_executor, mock).execute("airflow_trigger_dag", args)
    assert r.status == "denied" and mock.triggered == []


def test_approved_trigger_unpauses_and_posts_run(make_executor, mock):
    mock.paused = True
    ex = executor(make_executor, mock)
    pending = ex.execute("airflow_trigger_dag", {"conf": {"reason": "demo"}}).confirmation
    _, result = ex.execute_confirmed(pending.confirmation_id, RequestTrace("req"))
    assert result.ok and result.data["unpaused"] is True and mock.paused is False
    assert mock.triggered[0]["logical_date"] is None and mock.triggered[0]["conf"] == {"reason": "demo"}


def test_wait_reports_final_state_and_retries(make_executor, mock):
    run_id = mock.runs[1]["dag_run_id"]
    r = executor(make_executor, mock).execute(
        "airflow_wait_for_dag_run", {"run_id": run_id, "timeout_s": 10, "poll_s": 2}
    )
    assert r.data["finished"] and r.data["state"] == "success"
    assert r.data["retried_tasks"] == ["train_model"]


def test_wrong_credentials_unavailable(make_executor, mock):
    client = AirflowClient("http://airflow.test", "airflow", "WRONG", transport=mock.transport())
    r = make_executor({"airflow": client}).execute("airflow_list_dags", {})
    assert r.status == "unavailable" and "credentials" in r.error


def test_server_error_unavailable(make_executor, mock):
    mock.fail_with = 503
    assert executor(make_executor, mock).execute("airflow_list_dags", {}).status == "unavailable"


def test_connection_refused_unavailable(make_executor):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    client = AirflowClient("http://airflow.test", "airflow", "pw", transport=httpx.MockTransport(refuse))
    r = make_executor({"airflow": client}).execute("airflow_list_dags", {})
    assert r.status == "unavailable" and r.data is None


def test_http_timeout_reported_as_timeout(make_executor):
    def slow(request):
        if request.url.path == "/auth/token":
            return httpx.Response(201, json={"access_token": "tok1"})
        raise httpx.ReadTimeout("timed out")

    client = AirflowClient("http://airflow.test", "airflow", "pw", transport=httpx.MockTransport(slow))
    assert make_executor({"airflow": client}).execute("airflow_list_dags", {}).status == "timeout"


def test_missing_password_unavailable(make_executor, settings):
    from dataclasses import replace

    from agent.tools.airflow_tools import airflow_factory
    from agent.tools.base import ToolContext

    ctx = ToolContext(settings=replace(settings, airflow_password=""), policy={})
    with pytest.raises(Exception, match="credentials not configured"):
        airflow_factory(ctx)


def test_unknown_run_is_invalid_request(make_executor, mock):
    r = executor(make_executor, mock).execute("airflow_get_dag_run", {"run_id": "nope"})
    assert r.status == "invalid_request"


@pytest.mark.parametrize("bad", ["../../etc/passwd", "a;b", "x" * 300])
def test_malformed_ids_rejected(make_executor, mock, bad):
    r = executor(make_executor, mock).execute("airflow_get_dag_run", {"dag_id": bad})
    assert r.status == "invalid_request"


def test_structured_airflow3_exceptions_rendered_readably(make_executor, mock):
    run_id = mock.runs[0]["dag_run_id"]
    mock.logs[(run_id, "evaluate_and_log")] = [
        {
            "timestamp": "t",
            "level": "error",
            "event": "Task failed with exception",
            "error_detail": [
                {
                    "exc_type": "ModuleNotFoundError",
                    "exc_value": "No module named 'skops'",
                    "frames": [{"filename": "/x/mlflow/sklearn/__init__.py", "lineno": 701}],
                }
            ],
        },
    ]
    r = executor(make_executor, mock).execute(
        "airflow_get_task_logs", {"run_id": run_id, "task_id": "evaluate_and_log"}
    )
    assert r.data["root_cause_line"] == "Exception: ModuleNotFoundError: No module named 'skops' (at __init__.py:701)"
