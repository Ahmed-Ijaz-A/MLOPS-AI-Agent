"""FastAPI agent interface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.orchestrator import Agent
from api.main import create_app
from tests.fakes import healthy_cluster


@pytest.fixture
def k8s():
    return healthy_cluster()


@pytest.fixture
def client(make_executor, settings, k8s):
    agent = Agent(settings, executor=make_executor({"kubernetes": k8s}), llm=None)
    return TestClient(create_app(agent))


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["mode"] == "offline" and ready["tools"] >= 30


def test_chat_returns_answer_trace_and_request_id(client):
    res = client.post("/api/chat", json={"message": "Why is my prediction API unavailable?"})
    body = res.json()
    assert res.status_code == 200 and res.headers["x-request-id"] == body["request_id"]
    assert body["tool_calls"][0]["tool"] == "k8s_diagnose_deployment"
    assert body["systems_consulted"] == ["kubernetes"] and body["grounding"]["grounded"] is True


def test_client_supplied_request_id_propagates(client):
    res = client.post("/api/chat", json={"message": "list pods"}, headers={"X-Request-ID": "req_from_client1"})
    assert res.json()["request_id"] == "req_from_client1"


@pytest.mark.parametrize(
    "payload", [{}, {"message": ""}, {"message": "x" * 5000}, {"message": "hi", "session_id": "../etc"}]
)
def test_invalid_requests_rejected(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_confirmation_flow_over_http(client, k8s):
    body = client.post("/api/chat", json={"message": "restart the inference api"}).json()
    pending = body["pending_confirmations"][0]
    assert client.get("/api/confirmations").json()[0]["confirmation_id"] == pending["confirmation_id"]
    res = client.post(
        f"/api/confirmations/{pending['confirmation_id']}", json={"approve": True, "session_id": body["session_id"]}
    )
    assert res.status_code == 200 and res.json()["result"]["status"] == "ok" and len(k8s.apps.patches) == 1
    replay = client.post(f"/api/confirmations/{pending['confirmation_id']}", json={"approve": True})
    assert replay.status_code == 409


def test_malformed_confirmation_id(client):
    assert client.post("/api/confirmations/DROP TABLE", json={"approve": True}).status_code in (400, 404)


def test_tool_catalog_lists_permissions(client):
    tools = client.get("/api/tools").json()["tools"]
    perms = {t["name"]: t["permission"] for t in tools}
    assert perms["k8s_scale_deployment"] == "mutate" and perms["airflow_trigger_dag"] == "operate"
    assert perms["mlflow_list_runs"] == "read"


def test_audit_log_endpoint(client):
    client.post("/api/chat", json={"message": "restart the inference api"})
    events = client.get("/api/audit").json()
    assert events[-1]["event"] == "action_requested"


def test_integrations_reports_each_system(client):
    systems = client.get("/api/integrations").json()["systems"]
    assert systems["kubernetes"]["status"] == "ok" and systems["mlflow"]["status"] == "unavailable"


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "s3cret-key")
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    ok = client.post("/api/chat", json={"message": "list pods"}, headers={"X-API-Key": "s3cret-key"})
    assert ok.status_code == 200


def test_web_console_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "MLOps Agent Console" in res.text
    assert client.get("/static/console.js").status_code == 200
