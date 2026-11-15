"""Apache Airflow 3 workflow tools (REST API v2 with JWT auth from ``POST /auth/token``)."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agent.safety import ServiceUnavailable, ToolInputError, check_conf, check_triggerable_dag
from agent.schemas import PermissionLevel
from agent.tools.base import ToolContext, ToolOutput, ToolRegistry

TERMINAL = {"success", "failed"}
_DAG_ID = r"^[A-Za-z0-9_.\-]{1,250}$"
_RUN_ID = r"^[A-Za-z0-9_.:+\-T ]{1,250}$"


class AirflowClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not password:
            raise ServiceUnavailable("Airflow credentials not configured (set AIRFLOW_USERNAME / AIRFLOW_PASSWORD).")
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._token: str | None = None
        self._lock = threading.Lock()

    def _authenticate(self) -> str:
        try:
            resp = self._http.post("/auth/token", json={"username": self._username, "password": self._password})
        except httpx.TransportError as exc:
            raise ServiceUnavailable(f"Airflow API unreachable at {self.base_url}: {type(exc).__name__}") from exc
        if resp.status_code in (401, 403):
            raise ServiceUnavailable("Airflow rejected the configured credentials.")
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise ServiceUnavailable("Airflow /auth/token returned no access_token.")
        return token

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in range(2):
            with self._lock:
                if self._token is None:
                    self._token = self._authenticate()
                token = self._token
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", **kwargs.pop("headers", {})}
            try:
                resp = self._http.request(method, path, headers=headers, **kwargs)
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"Airflow API timed out on {method} {path}") from exc
            except httpx.TransportError as exc:
                raise ServiceUnavailable(f"Airflow API unreachable at {self.base_url}: {type(exc).__name__}") from exc
            if resp.status_code == 401 and attempt == 0:  # expired JWT -> refresh once
                with self._lock:
                    self._token = None
                continue
            if resp.status_code == 404:
                raise ToolInputError(f"Airflow: not found ({path}).")
            if resp.status_code >= 500:
                raise ServiceUnavailable(f"Airflow API error {resp.status_code} on {path}")
            if resp.status_code >= 400:
                detail = resp.text[:400]
                raise ToolInputError(f"Airflow rejected {method} {path}: {resp.status_code} {detail}")
            return resp.json() if resp.content else {}
        raise ServiceUnavailable("Airflow authentication failed after token refresh.")

    # convenience wrappers -------------------------------------------------------------
    def dag_runs(self, dag_id: str, limit: int) -> list[dict[str, Any]]:
        try:
            body = self.request(
                "GET", f"/api/v2/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-run_after"}
            )
        except ToolInputError as exc:
            if "404" in str(exc) or "not found" in str(exc):
                raise
            body = self.request(
                "GET", f"/api/v2/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-start_date"}
            )
        return body.get("dag_runs", [])

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        body = self.request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        return body.get("task_instances", [])


def airflow_factory(ctx: ToolContext) -> AirflowClient:
    s = ctx.settings
    return AirflowClient(s.airflow_base_url, s.airflow_username, s.airflow_password, timeout=min(s.tool_timeout_s, 15))


def _client(ctx: ToolContext) -> AirflowClient:
    return ctx.client("airflow")


def _run_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": r.get("dag_run_id") or r.get("run_id"),
        "state": r.get("state"),
        "run_type": r.get("run_type"),
        "logical_date": r.get("logical_date"),
        "run_after": r.get("run_after"),
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
        "conf": r.get("conf") or {},
    }


def _ti_row(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": t.get("task_id"),
        "state": t.get("state"),
        "try_number": t.get("try_number"),
        "max_tries": t.get("max_tries"),
        "start_date": t.get("start_date"),
        "end_date": t.get("end_date"),
        "duration_s": round(t["duration"], 2) if isinstance(t.get("duration"), int | float) else None,
    }


def _resolve_run(client: AirflowClient, dag_id: str, run_id: str | None, want_state: str | None = None) -> dict:
    if run_id and run_id != "latest":
        return client.request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}")
    runs = client.dag_runs(dag_id, 25)
    if want_state:
        runs = [r for r in runs if r.get("state") == want_state]
    if not runs:
        raise ToolInputError(f"No {want_state + ' ' if want_state else ''}runs found for DAG '{dag_id}'.")
    return runs[0]


_ERR_PATTERNS = re.compile(
    r"(error|exception|traceback|failed|failure injection|quality gate|validation)", re.IGNORECASE
)


def _log_lines(body: Any) -> list[str]:
    content = body.get("content", body) if isinstance(body, dict) else body
    lines: list[str] = []
    if isinstance(content, str):
        lines = content.splitlines()
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                ev = item.get("event") or item.get("message") or ""
                lvl = item.get("level", "")
                ts = item.get("timestamp", "")
                extra = item.get("error_detail")
                lines.append(f"{ts} {lvl.upper()} {ev}".strip())
                if extra:
                    lines.append(str(extra)[:1500])
            elif isinstance(item, list | tuple) and len(item) == 2:  # legacy [(host, text)]
                lines.extend(str(item[1]).splitlines())
            else:
                lines.append(str(item))
    return [ln for ln in lines if ln.strip()]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Base):
    pass


class DagArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    limit: int = Field(10, ge=1, le=100)


class DagRunArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    run_id: str = Field("latest", pattern=_RUN_ID, description="DAG run id, or 'latest'.")


class FailedTasksArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    run_id: str | None = Field(None, pattern=_RUN_ID, description="Run to inspect; default = most recent FAILED run.")


class TaskLogArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    run_id: str = Field(..., pattern=_RUN_ID)
    task_id: str = Field(..., pattern=_DAG_ID)
    try_number: int | None = Field(None, ge=1, le=20, description="Default: the latest attempt.")
    tail: int = Field(60, ge=5, le=400)


class TriggerArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    conf: dict[str, Any] = Field(default_factory=dict, description="Run conf, e.g. {'reason': 'new data'}.")


class WaitArgs(_Base):
    dag_id: str = Field("ml_training_pipeline", pattern=_DAG_ID)
    run_id: str = Field(..., pattern=_RUN_ID)
    timeout_s: int = Field(300, ge=10, le=900)
    poll_s: int = Field(10, ge=2, le=60)


def _trigger_precheck(args: TriggerArgs, ctx: ToolContext) -> None:
    check_triggerable_dag(args.dag_id, ctx.policy)
    check_conf(args.conf, ctx.policy)


def register(reg: ToolRegistry) -> None:
    @reg.tool("airflow_list_dags", "List Airflow DAGs with paused state, tags and schedule.", NoArgs, system="airflow")
    def list_dags(args: NoArgs, ctx: ToolContext) -> ToolOutput:
        body = _client(ctx).request("GET", "/api/v2/dags", params={"limit": 100})
        dags = [
            {
                "dag_id": d.get("dag_id"),
                "is_paused": d.get("is_paused"),
                "tags": [t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])],
                "schedule": d.get("timetable_summary") or d.get("timetable_description"),
                "has_import_errors": d.get("has_import_errors"),
            }
            for d in body.get("dags", [])
        ]
        return ToolOutput(
            f"{len(dags)} DAG(s): " + ", ".join(f"{d['dag_id']}{' (paused)' if d['is_paused'] else ''}" for d in dags),
            {"dags": dags},
        )

    @reg.tool(
        "airflow_list_dag_runs",
        "List recent runs of a DAG with their state and timing (status of recent training workflows).",
        DagArgs,
        system="airflow",
    )
    def list_runs(args: DagArgs, ctx: ToolContext) -> ToolOutput:
        runs = [_run_row(r) for r in _client(ctx).dag_runs(args.dag_id, args.limit)]
        counts: dict[str, int] = {}
        for r in runs:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        latest = runs[0] if runs else None
        summary = f"{args.dag_id}: {len(runs)} recent run(s) {counts}"
        if latest:
            summary += f"; latest {latest['run_id']} is {latest['state']} (started {latest['start_date']})"
        return ToolOutput(summary + ".", {"dag_id": args.dag_id, "runs": runs, "state_counts": counts})

    @reg.tool(
        "airflow_get_dag_run",
        "Get one DAG run's state with every task instance's state, attempt number and duration.",
        DagRunArgs,
        system="airflow",
    )
    def get_run(args: DagRunArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        run = _run_row(_resolve_run(client, args.dag_id, args.run_id))
        tis = [_ti_row(t) for t in client.task_instances(args.dag_id, run["run_id"])]
        tasks = ", ".join(f"{t['task_id']}={t['state']}" for t in tis)
        return ToolOutput(f"Run {run['run_id']} of {args.dag_id} is {run['state']}: {tasks}.", {**run, "tasks": tis})

    @reg.tool(
        "airflow_get_failed_tasks",
        "Identify failed / upstream_failed tasks in a DAG run (default: the most recent failed run).",
        FailedTasksArgs,
        system="airflow",
    )
    def failed_tasks(args: FailedTasksArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        run = _run_row(_resolve_run(client, args.dag_id, args.run_id, want_state=None if args.run_id else "failed"))
        tis = [_ti_row(t) for t in client.task_instances(args.dag_id, run["run_id"])]
        failed = [t for t in tis if t["state"] == "failed"]
        blocked = [t for t in tis if t["state"] in ("upstream_failed", "skipped")]
        summary = (
            f"Run {run['run_id']} ({run['state']}): failed tasks {[t['task_id'] for t in failed] or 'none'}; "
            f"blocked downstream {[t['task_id'] for t in blocked] or 'none'}."
        )
        return ToolOutput(summary, {"run": run, "failed_tasks": failed, "blocked_tasks": blocked})

    @reg.tool(
        "airflow_get_task_logs",
        "Retrieve a task attempt's log tail plus extracted error lines (failure diagnostics).",
        TaskLogArgs,
        system="airflow",
    )
    def task_logs(args: TaskLogArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        try_number = args.try_number
        if try_number is None:
            ti = client.request("GET", f"/api/v2/dags/{args.dag_id}/dagRuns/{args.run_id}/taskInstances/{args.task_id}")
            try_number = max(int(ti.get("try_number") or 1), 1)
        body = client.request(
            "GET",
            f"/api/v2/dags/{args.dag_id}/dagRuns/{args.run_id}/taskInstances/{args.task_id}/logs/{try_number}",
            params={"full_content": "true"},
        )
        lines = _log_lines(body)
        errors = [ln[:400] for ln in lines if _ERR_PATTERNS.search(ln) and "INFO" not in ln[:40]][-15:]
        root = next(
            (
                ln
                for ln in reversed(errors)
                if any(k in ln for k in ("AirflowFailException", "Error:", "FAILED", "Exception:"))
            ),
            errors[-1] if errors else None,
        )
        return ToolOutput(
            f"{args.task_id} try {try_number}: {len(lines)} log lines; root-cause line: {root or 'no error lines found'}",
            {
                "task_id": args.task_id,
                "try_number": try_number,
                "error_lines": errors,
                "root_cause_line": root,
                "tail": [ln[:400] for ln in lines[-args.tail :]],
            },
        )

    @reg.tool(
        "airflow_trigger_dag",
        "Trigger an APPROVED DAG (only the training pipeline) with an optional conf. Requires human approval.",
        TriggerArgs,
        system="airflow",
        permission=PermissionLevel.OPERATE,
        precheck=_trigger_precheck,
        describe_action=lambda a: f"Trigger Airflow DAG '{a.dag_id}' with conf {a.conf or {}}",
    )
    def trigger(args: TriggerArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        dag = client.request("GET", f"/api/v2/dags/{args.dag_id}")
        unpaused = False
        if dag.get("is_paused"):
            client.request("PATCH", f"/api/v2/dags/{args.dag_id}", json={"is_paused": False})
            unpaused = True
        run = client.request(
            "POST",
            f"/api/v2/dags/{args.dag_id}/dagRuns",
            json={"logical_date": None, "conf": args.conf, "note": "triggered by MLOps agent after human approval"},
        )
        row = _run_row(run)
        return ToolOutput(
            f"Triggered {args.dag_id}: run {row['run_id']} is {row['state']}"
            + (" (DAG was paused and has been unpaused)" if unpaused else "")
            + ". Use airflow_wait_for_dag_run to follow it.",
            {**row, "unpaused": unpaused},
        )

    @reg.tool(
        "airflow_wait_for_dag_run",
        "Poll a DAG run until it reaches success/failed (or timeout) and report the final task states.",
        WaitArgs,
        system="airflow",
        timeout_s=920,
    )
    def wait(args: WaitArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        deadline = time.monotonic() + args.timeout_s
        run: dict[str, Any] = {}
        while True:
            run = _run_row(client.request("GET", f"/api/v2/dags/{args.dag_id}/dagRuns/{args.run_id}"))
            if run["state"] in TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(args.poll_s)
        tis = [_ti_row(t) for t in client.task_instances(args.dag_id, args.run_id)]
        finished = run["state"] in TERMINAL
        retried = [t["task_id"] for t in tis if (t.get("try_number") or 1) > 1]
        summary = (
            f"Run {args.run_id} {'finished as ' + run['state'].upper() if finished else 'still ' + str(run['state']) + ' after timeout'}: "
            + ", ".join(f"{t['task_id']}={t['state']}" for t in tis)
        )
        if retried:
            summary += f"; tasks that needed retries: {retried}"
        return ToolOutput(summary + ".", {**run, "finished": finished, "tasks": tis, "retried_tasks": retried})
