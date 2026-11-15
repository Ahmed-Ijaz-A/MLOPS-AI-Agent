"""Test doubles: scripted LLM, fake Kubernetes / MLflow clients, Airflow mock transport, synthetic data."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from typing import Any

import httpx
import numpy as np
import pandas as pd

from agent.llm import LLMTurn
from training.data import CATEGORICAL, NUMERIC

# --------------------------------------------------------------------------- synthetic data

_CATS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": ["existing paid", "critical/other existing credit", "delayed previously", "all paid"],
    "purpose": ["new car", "used car", "radio/tv", "furniture/equipment", "business", "education"],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": ["male single", "female div/dep/mar", "male mar/wid", "male div/sep"],
    "other_parties": ["none", "guarantor", "co applicant"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["none", "bank", "stores"],
    "housing": ["own", "rent", "for free"],
    "job": ["skilled", "unskilled resident", "high qualif/self emp/mgmt", "unemp/unskilled non res"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}


def synthetic_credit_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "duration": rng.integers(4, 72, n),
            "credit_amount": rng.integers(250, 18000, n),
            "installment_commitment": rng.integers(1, 5, n),
            "residence_since": rng.integers(1, 5, n),
            "age": rng.integers(19, 75, n),
            "existing_credits": rng.integers(1, 4, n),
            "num_dependents": rng.integers(1, 3, n),
        }
    )
    for col in CATEGORICAL:
        df[col] = rng.choice(_CATS[col], n)
    logit = (
        0.05 * (df.duration - 20)
        + 0.0001 * (df.credit_amount - 3000)
        - 0.03 * (df.age - 35)
        + 1.5 * (df.checking_status == "<0")
        - 1.0
    )
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df[[*NUMERIC, *CATEGORICAL, "default"]]


# --------------------------------------------------------------------------- scripted LLM


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_block(name: str, args: dict[str, Any] | None = None, block_id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id or f"toolu_{name}", "name": name, "input": args or {}}


class ScriptedLLM:
    """Replays a fixed list of turns; records every request it received."""

    name = "scripted"

    def __init__(self, turns: list[list[dict[str, Any]]], stop_reasons: list[str] | None = None) -> None:
        self.turns = list(turns)
        self.stop_reasons = list(stop_reasons or [])
        self.calls: list[list[dict[str, Any]]] = []

    def respond(self, system, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        if not self.turns:
            return LLMTurn(content=[text_block("**Summary** - done.")], stop_reason="end_turn")
        content = self.turns.pop(0)
        has_tool = any(b["type"] == "tool_use" for b in content)
        default = "tool_use" if has_tool else "end_turn"
        stop = self.stop_reasons.pop(0) if self.stop_reasons else default
        return LLMTurn(content=content, stop_reason=stop, usage={"input_tokens": 10, "output_tokens": 5})


# --------------------------------------------------------------------------- Kubernetes fakes


def _ts(minutes_ago: int = 5) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes_ago)


def make_container_status(
    name: str = "inference-api",
    ready: bool = True,
    restarts: int = 0,
    waiting: str | None = None,
    waiting_msg: str | None = None,
    last_term: str | None = None,
    exit_code: int | None = None,
    image: str = "mlops-inference:model-v1",
):
    return NS(
        name=name,
        image=image,
        ready=ready,
        restart_count=restarts,
        state=NS(
            waiting=NS(reason=waiting, message=waiting_msg) if waiting else None,
            terminated=None,
            running=None if waiting else NS(started_at=_ts()),
        ),
        last_state=NS(terminated=NS(reason=last_term, exit_code=exit_code) if last_term else None),
    )


def make_pod(name: str, phase: str = "Running", ready: bool = True, labels: dict | None = None, **cs_kwargs):
    cs = make_container_status(ready=ready, **cs_kwargs)
    return NS(
        metadata=NS(
            name=name,
            creation_timestamp=_ts(30),
            labels=labels or {"app.kubernetes.io/name": "inference-api"},
            deletion_timestamp=None,
        ),
        spec=NS(node_name="docker-desktop", containers=[NS(name="inference-api")]),
        status=NS(
            phase=phase,
            container_statuses=[cs] if phase != "Pending" else [],
            conditions=[NS(type="Ready", status="True" if ready else "False")],
        ),
    )


def make_deployment(
    name: str = "inference-api",
    replicas: int = 2,
    ready: int = 2,
    image: str = "mlops-inference:model-v3",
    model_version: str = "3",
    revision: str = "4",
    cm_name: str = "inference-config",
):
    container = NS(image=image, env_from=[NS(config_map_ref=NS(name=cm_name))])
    return NS(
        metadata=NS(
            name=name,
            labels={"app.kubernetes.io/version": f"model-v{model_version}"},
            annotations={"deployment.kubernetes.io/revision": revision},
        ),
        spec=NS(
            replicas=replicas,
            selector=NS(match_labels={"app.kubernetes.io/name": name}),
            template=NS(
                metadata=NS(
                    annotations={
                        "mlops.agent/model-name": "credit-risk-classifier",
                        "mlops.agent/model-version": model_version,
                        "mlops.agent/model-run-id": "a" * 32,
                        "mlops.agent/dataset-version": "v2.0",
                    }
                ),
                spec=NS(containers=[container]),
            ),
        ),
        status=NS(
            ready_replicas=ready,
            available_replicas=ready,
            updated_replicas=replicas,
            unavailable_replicas=max(replicas - ready, 0),
            conditions=[NS(type="Available", status="True" if ready else "False", reason="", message="")],
        ),
    )


def make_event(reason: str, obj: str, message: str, etype: str = "Warning", kind: str = "Pod", minutes_ago: int = 1):
    ts = _ts(minutes_ago)
    return NS(
        type=etype,
        reason=reason,
        message=message,
        count=3,
        involved_object=NS(kind=kind, name=obj),
        last_timestamp=ts,
        event_time=None,
        first_timestamp=ts,
        metadata=NS(creation_timestamp=ts),
    )


def make_service(name: str = "inference-api"):
    return NS(
        metadata=NS(name=name),
        spec=NS(
            type="ClusterIP",
            cluster_ip="10.96.0.10",
            selector={"app.kubernetes.io/name": name},
            ports=[NS(port=80, target_port="http", protocol="TCP")],
        ),
    )


def make_replicaset(revision: int, image: str, deployment: str = "inference-api"):
    template = NS(
        metadata=NS(labels={"app.kubernetes.io/name": deployment, "pod-template-hash": f"h{revision}"}),
        spec=NS(containers=[NS(image=image, name="inference-api")]),
    )
    return NS(
        metadata=NS(
            annotations={"deployment.kubernetes.io/revision": str(revision)},
            owner_references=[NS(kind="Deployment", name=deployment)],
        ),
        spec=NS(template=template),
    )


class ApiError(Exception):
    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"({status}) {reason}")
        self.status = status
        self.reason = reason


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, NS):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


class FakeCore:
    def __init__(
        self, pods=None, services=None, events=None, logs=None, configmaps=None, error: Exception | None = None
    ):
        self.pods = pods or []
        self.services = services or []
        self.events = events or []
        self.logs = logs or {}
        self.configmaps = configmaps if configmaps is not None else {"inference-config": {"MODEL_VERSION": "3"}}
        self.error = error

    def _check(self):
        if self.error:
            raise self.error

    def list_namespaced_pod(self, ns, label_selector=None, _request_timeout=None):
        self._check()
        pods = self.pods
        if label_selector:
            wanted = dict(kv.split("=", 1) for kv in label_selector.split(","))
            pods = [p for p in pods if all(p.metadata.labels.get(k) == v for k, v in wanted.items())]
        return NS(items=pods)

    def read_namespaced_pod(self, name, ns, _request_timeout=None):
        self._check()
        for p in self.pods:
            if p.metadata.name == name:
                return p
        raise ApiError(404, "pod not found")

    def read_namespaced_pod_log(self, name, ns, _request_timeout=None, **kwargs):
        self._check()
        return self.logs.get((name, bool(kwargs.get("previous"))), self.logs.get(name, ""))

    def list_namespaced_service(self, ns, _request_timeout=None):
        self._check()
        return NS(items=self.services)

    def list_namespaced_event(self, ns, _request_timeout=None):
        self._check()
        return NS(items=self.events)

    def read_namespaced_config_map(self, name, ns, _request_timeout=None):
        self._check()
        if name not in self.configmaps:
            raise ApiError(404, "configmap not found")
        return NS(data=self.configmaps[name])


class FakeApps:
    def __init__(self, deployments=None, replicasets=None, error: Exception | None = None):
        self.deployments = {d.metadata.name: d for d in (deployments or [])}
        self.replicasets = replicasets or []
        self.patches: list[tuple[str, str, dict]] = []
        self.error = error
        self.api_client = NS(sanitize_for_serialization=_to_dict)

    def _check(self):
        if self.error:
            raise self.error

    def list_namespaced_deployment(self, ns, _request_timeout=None):
        self._check()
        return NS(items=list(self.deployments.values()))

    def read_namespaced_deployment(self, name, ns, _request_timeout=None):
        self._check()
        if name not in self.deployments:
            raise ApiError(404, "deployment not found")
        return self.deployments[name]

    def patch_namespaced_deployment(self, name, ns, body, _request_timeout=None):
        self._check()
        self.patches.append(("deployment", name, body))
        return self.deployments.get(name)

    def patch_namespaced_deployment_scale(self, name, ns, body, _request_timeout=None):
        self._check()
        self.patches.append(("scale", name, body))
        return NS()

    def list_namespaced_replica_set(self, ns, label_selector=None, _request_timeout=None):
        self._check()
        return NS(items=self.replicasets)


def k8s_clients(core: FakeCore | None = None, apps: FakeApps | None = None):
    return NS(core=core or FakeCore(), apps=apps or FakeApps())


def healthy_cluster():
    pods = [make_pod("inference-api-abc-1"), make_pod("inference-api-abc-2")]
    return k8s_clients(
        FakeCore(pods=pods, services=[make_service()], events=[]),
        FakeApps(deployments=[make_deployment()]),
    )


# --------------------------------------------------------------------------- MLflow fake (light-weight)


class FakeMlflow:
    """Minimal MlflowClient stand-in for cross-system tests (the MLflow tool tests use a real store)."""

    def __init__(
        self,
        runs: dict[str, Any] | None = None,
        versions: dict[str, Any] | None = None,
        aliases: dict[str, str] | None = None,
        error: Exception | None = None,
    ):
        self.runs = runs or {}
        self.versions = versions or {}
        self.aliases = aliases or {}
        self.error = error

    def _check(self):
        if self.error:
            raise self.error

    def get_run(self, run_id):
        self._check()
        if run_id not in self.runs:
            raise RuntimeError(f"RESOURCE_DOES_NOT_EXIST: run {run_id}")
        return self.runs[run_id]

    def get_model_version(self, name, version):
        self._check()
        return self.versions[str(version)]

    def get_model_version_by_alias(self, name, alias):
        self._check()
        return self.versions[self.aliases[alias]]


def fake_run(
    run_id: str,
    f1: float = 0.66,
    md5: str = "md5v2",
    version: str = "v2.0",
    gate: str = "passed",
    name: str = "lr-run",
    days_ago: float = 2.0,
):
    start = int((datetime.now(UTC) - timedelta(days=days_ago)).timestamp() * 1000)
    return NS(
        info=NS(run_id=run_id, run_name=name, status="FINISHED", start_time=start, end_time=start + 1000),
        data=NS(
            metrics={"f1": f1, "recall": 0.8, "roc_auc": 0.8, "precision": 0.55, "accuracy": 0.75},
            params={"model_type": "logistic_regression"},
            tags={
                "dataset.md5": md5,
                "dataset.version": version,
                "dataset.path": "data/raw/credit_risk.csv",
                "git.commit": "c" * 40,
                "quality_gate.status": gate,
                "model_type": "logistic_regression",
            },
        ),
    )


def fake_version(version: str, run_id: str, days_ago: float = 2.0):
    created = int((datetime.now(UTC) - timedelta(days=days_ago)).timestamp() * 1000)
    return NS(version=version, run_id=run_id, creation_timestamp=created, status="READY", tags={})


# --------------------------------------------------------------------------- Airflow mock server


class AirflowMock:
    """In-memory Airflow 3 REST API (httpx.MockTransport)."""

    def __init__(
        self,
        runs: list[dict] | None = None,
        tasks: dict[str, list[dict]] | None = None,
        logs: dict[tuple[str, str], Any] | None = None,
        paused: bool = False,
        password: str = "pw",
        fail_with: int | None = None,
        expire_first_token: bool = False,
    ):
        self.runs = runs if runs is not None else []
        self.tasks = tasks or {}
        self.logs = logs or {}
        self.paused = paused
        self.password = password
        self.fail_with = fail_with
        self.expire_first_token = expire_first_token
        self.tokens_issued = 0
        self.requests: list[httpx.Request] = []
        self.triggered: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/auth/token":
            body = json.loads(request.content)
            if body.get("password") != self.password:
                return httpx.Response(401, json={"detail": "invalid credentials"})
            self.tokens_issued += 1
            return httpx.Response(201, json={"access_token": f"tok{self.tokens_issued}"})
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer tok"):
            return httpx.Response(401)
        if self.expire_first_token and auth == "Bearer tok1":
            return httpx.Response(401, json={"detail": "token expired"})
        if self.fail_with:
            return httpx.Response(self.fail_with, text="boom")
        parts = path.strip("/").split("/")  # api v2 dags {dag} dagRuns {run} taskInstances {task} logs {try}
        if path == "/api/v2/dags":
            return httpx.Response(
                200,
                json={
                    "dags": [
                        {
                            "dag_id": "ml_training_pipeline",
                            "is_paused": self.paused,
                            "tags": [{"name": "mlops"}],
                            "timetable_summary": None,
                            "has_import_errors": False,
                        }
                    ],
                    "total_entries": 1,
                },
            )
        if len(parts) == 4 and parts[3] == "ml_training_pipeline" and request.method == "GET":
            return httpx.Response(200, json={"dag_id": "ml_training_pipeline", "is_paused": self.paused})
        if len(parts) == 4 and request.method == "PATCH":
            self.paused = json.loads(request.content)["is_paused"]
            return httpx.Response(200, json={"dag_id": parts[3], "is_paused": self.paused})
        if len(parts) == 4:
            return httpx.Response(404, json={"detail": "DAG not found"})
        if len(parts) == 5 and parts[4] == "dagRuns" and request.method == "GET":
            return httpx.Response(200, json={"dag_runs": self.runs, "total_entries": len(self.runs)})
        if len(parts) == 5 and parts[4] == "dagRuns" and request.method == "POST":
            body = json.loads(request.content)
            self.triggered.append(body)
            run = {
                "dag_run_id": "manual__2026-11-15T10:00:00",
                "state": "queued",
                "conf": body.get("conf"),
                "run_type": "manual",
                "logical_date": None,
            }
            self.runs.insert(0, run)
            return httpx.Response(200, json=run)
        if len(parts) == 6:
            run = next((r for r in self.runs if r["dag_run_id"] == parts[5]), None)
            return httpx.Response(200, json=run) if run else httpx.Response(404, json={"detail": "run not found"})
        if len(parts) == 7 and parts[6] == "taskInstances":
            return httpx.Response(200, json={"task_instances": self.tasks.get(parts[5], [])})
        if len(parts) == 8:
            ti = next((t for t in self.tasks.get(parts[5], []) if t["task_id"] == parts[7]), None)
            return httpx.Response(200, json=ti) if ti else httpx.Response(404)
        if len(parts) == 10 and parts[8] == "logs":
            content = self.logs.get((parts[5], parts[7]), [])
            return httpx.Response(200, json={"content": content, "continuation_token": None})
        return httpx.Response(404, json={"detail": f"no route {path}"})


def pipeline_runs() -> tuple[list[dict], dict[str, list[dict]]]:
    ok_id, bad_id = "manual__2026-11-14T09:00:00", "manual__2026-11-15T09:00:00"
    names = ["validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate", "register_model"]
    runs = [
        {
            "dag_run_id": bad_id,
            "state": "failed",
            "run_type": "manual",
            "start_date": "2026-11-15T09:00:01Z",
            "end_date": "2026-11-15T09:03:00Z",
            "conf": {"model_type": "gradient_boosting"},
        },
        {
            "dag_run_id": ok_id,
            "state": "success",
            "run_type": "manual",
            "start_date": "2026-11-14T09:00:01Z",
            "end_date": "2026-11-14T09:04:00Z",
            "conf": {},
        },
    ]
    ok_tasks = [{"task_id": n, "state": "success", "try_number": 1, "max_tries": 2, "duration": 4.2} for n in names]
    ok_tasks[2]["try_number"] = 2
    bad_tasks = [
        {"task_id": n, "state": "success", "try_number": 1, "max_tries": 2, "duration": 3.0} for n in names[:4]
    ]
    bad_tasks += [
        {"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 0, "duration": 1.0},
        {"task_id": "register_model", "state": "upstream_failed", "try_number": 0, "max_tries": 2},
    ]
    return runs, {ok_id: ok_tasks, bad_id: bad_tasks}
