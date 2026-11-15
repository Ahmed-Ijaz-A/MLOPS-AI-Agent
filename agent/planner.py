"""Deterministic offline planner + evidence-only answer composer.

Used when no LLM credentials are configured (``LLM_PROVIDER=offline`` or no API key) and in CI. It
maps intents to tool plans, executes them through the same executor (same safety + tracing), and
composes the answer ONLY from tool summaries - so it is grounded by construction.

Steps are callables receiving previous results, so later steps can use earlier evidence (e.g. the best
run's id feeds the DVC provenance lookup).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agent.schemas import ToolResult

Step = tuple[str, dict[str, Any]] | Callable[[list[ToolResult]], tuple[str, dict[str, Any]] | None]

_RUN_ID = re.compile(r"\b([0-9a-f]{32})\b")
_VERSION = re.compile(r"\b(?:data-)?(v\d+\.\d+)\b", re.I)
_NAMESPACE = re.compile(r"(?:namespace|ns|-n)\s+([a-z0-9][-a-z0-9]*)", re.I)
_REPLICAS = re.compile(r"\bto\s+(\d+)\s*(?:replicas?|pods?)?", re.I)
_LAST_N = re.compile(r"\blast\s+(\d+)", re.I)
_METRICS = ("f1", "roc_auc", "recall", "precision", "accuracy")


def _metric(text: str) -> str:
    t = text.lower().replace("-", "_").replace(" ", "_")
    if "auc" in t:
        return "roc_auc"
    for m in _METRICS:
        if m in t:
            return m
    return "f1"


def _ns(text: str) -> str:
    m = _NAMESPACE.search(text)
    return m.group(1) if m else "mlops-agent"


def _find(results: list[ToolResult], tool: str) -> ToolResult | None:
    for r in reversed(results):
        if r.tool == tool and r.ok:
            return r
    return None


def _best_run_id(results: list[ToolResult]) -> str | None:
    best = _find(results, "mlflow_find_best_run")
    if best:
        return best.data["best"]["run_id"]
    runs = _find(results, "mlflow_list_runs")
    if runs and runs.data.get("runs"):
        return runs.data["runs"][0]["run_id"]
    return None


def _first_failed_task_logs(results: list[ToolResult]):
    r = _find(results, "airflow_get_failed_tasks")
    if r and r.data["failed_tasks"]:
        return (
            "airflow_get_task_logs",
            {"run_id": r.data["run"]["run_id"], "task_id": r.data["failed_tasks"][0]["task_id"]},
        )
    return None


def _unhealthy_pod_logs(results: list[ToolResult]):
    r = _find(results, "k8s_find_unhealthy_pods")
    if r and r.data["unhealthy"]:
        pod = r.data["unhealthy"][0]
        return ("k8s_get_pod_logs", {"pod": pod["name"], "previous": pod["restarts"] > 0, "tail": 80})
    return None


def plan(message: str) -> tuple[str, list[Step]]:
    t = message.lower()
    ns = _ns(message)
    m_run = _RUN_ID.search(t)
    run_id = m_run.group(1) if m_run else None
    versions = [v.lower() for v in _VERSION.findall(message)]

    def has(*words: str) -> bool:
        return any(w in t for w in words)

    # --- mutations (always end in a pending confirmation) ---
    if has("restart"):
        return "k8s_restart", [("k8s_restart_deployment", {"namespace": ns, "deployment": "inference-api"})]
    if has("scale"):
        m = _REPLICAS.search(t)
        replicas = int(m.group(1)) if m else 2
        return "k8s_scale", [
            ("k8s_scale_deployment", {"namespace": ns, "deployment": "inference-api", "replicas": replicas})
        ]
    if has("rollback", "roll back"):
        return "k8s_rollback", [("k8s_rollback_deployment", {"namespace": ns, "deployment": "inference-api"})]
    if has("trigger", "kick off", "start the training", "run the training", "launch"):
        conf: dict[str, Any] = {"reason": "requested via agent"}
        for inj in ("corrupt_data", "low_quality", "flaky_training"):
            if inj in t or inj.replace("_", " ") in t:
                conf["inject_failure"] = inj
        return "airflow_trigger", [("airflow_trigger_dag", {"dag_id": "ml_training_pipeline", "conf": conf})]

    # --- the mandatory multi-tool scenario ---
    if has("retrain") or (has("best") and has("deploy") and has("dataset")):
        return "integration", [
            ("mlflow_find_best_run", {"metric": _metric(t)}),
            lambda rs: ("dvc_resolve_run_dataset", {"run_id": _best_run_id(rs)}) if _best_run_id(rs) else None,
            ("mlflow_get_model_registry", {}),
            ("k8s_get_deployed_version", {"namespace": ns}),
            ("k8s_diagnose_deployment", {"namespace": ns}),
            ("airflow_list_dag_runs", {"limit": 5}),
            ("dvc_get_dataset_version", {}),
            ("assess_retraining_need", {}),
        ]

    # --- DVC ---
    if has("dataset", "data version", "dvc", "data changed", "reproduce"):
        if run_id:
            return "dvc_run_provenance", [("dvc_resolve_run_dataset", {"run_id": run_id})]
        if has("experiment", "run", "model") and has("used", "trained"):
            return "dvc_run_provenance", [
                ("mlflow_find_best_run", {}) if has("best") else ("mlflow_list_runs", {"max_results": 5}),
                lambda rs: ("dvc_resolve_run_dataset", {"run_id": _best_run_id(rs)}) if _best_run_id(rs) else None,
            ]
        if has("reproduce", "restore", "previous version", "checkout"):
            return "dvc_reproduce", [("dvc_reproduce_dataset", {"version": versions[0] if versions else "v1.0"})]
        if has("compare", "diff", "drift", "between") or len(versions) >= 2:
            a = versions[0] if versions else "v1.0"
            b = versions[1] if len(versions) > 1 else "HEAD"
            return "dvc_compare", [("dvc_compare_versions", {"rev_a": a, "rev_b": b})]
        if has("changed", "modified", "status", "stale"):
            return "dvc_status", [("dvc_check_data_status", {}), ("dvc_get_dataset_version", {})]
        if has("list", "which datasets", "tracked"):
            return "dvc_list", [("dvc_list_tracked_datasets", {})]
        return "dvc_version", [("dvc_get_dataset_version", {}), ("dvc_list_dataset_versions", {"limit": 10})]

    # --- Airflow ---
    if has("pipeline", "dag", "airflow", "workflow", "task"):
        if has("fail", "error", "why", "broke", "diagnos"):
            return "airflow_failure", [("airflow_get_failed_tasks", {}), _first_failed_task_logs]
        if has("list", "available", "which dags"):
            return "airflow_list", [("airflow_list_dags", {})]
        return "airflow_status", [
            ("airflow_list_dag_runs", {"limit": 5}),
            ("airflow_get_dag_run", {"run_id": "latest"}),
        ]

    # --- Kubernetes ---
    if has("unavailable", "down", "not working", "diagnos", "why is", "health", "crash", "broken"):
        return "k8s_diagnose", [
            ("k8s_diagnose_deployment", {"namespace": ns}),
            ("k8s_get_events", {"namespace": ns, "warnings_only": True}),
        ]
    if has("log"):
        return "k8s_logs", [("k8s_find_unhealthy_pods", {"namespace": ns}), _unhealthy_pod_logs]
    if has("service", "endpoint"):
        return "k8s_services", [("k8s_list_services", {"namespace": ns})]
    if has("unhealthy", "restart count", "restarts"):
        return "k8s_unhealthy", [("k8s_find_unhealthy_pods", {"namespace": ns})]
    if has("version") and has("deploy", "running", "serving", "kubernetes", "k8s"):
        return "k8s_version", [("k8s_get_deployed_version", {"namespace": ns})]
    if has("pod", "deployment", "kubernetes", "k8s", "cluster", "deployed", "running"):
        return "k8s_workloads", [
            ("k8s_list_workloads", {"namespace": ns}),
            ("k8s_get_deployed_version", {"namespace": ns}),
        ]
    if has("event"):
        return "k8s_events", [("k8s_get_events", {"namespace": ns})]

    # --- MLflow ---
    if has("compare") and has("experiment"):
        return "mlflow_compare_experiments", [("mlflow_list_experiments", {})]
    if has("compare", "candidate", "recommend"):
        m = _LAST_N.search(t)
        return "mlflow_compare", [
            ("mlflow_compare_runs", {"last_n": int(m.group(1)) if m else 5, "metric": _metric(t)})
        ]
    if has("registry", "registered", "champion", "model version"):
        return "mlflow_registry", [("mlflow_get_model_registry", {})]
    if has("best", "top model", "highest"):
        return "mlflow_best", [("mlflow_find_best_run", {"metric": _metric(t)})]
    if run_id:
        return "mlflow_run", [("mlflow_get_run", {"run_id": run_id})]
    if has("experiment"):
        return "mlflow_experiments", [("mlflow_list_experiments", {})]
    if has("run", "metric", "model", "training"):
        return "mlflow_runs", [("mlflow_list_runs", {"max_results": 10})]

    return "overview", [
        ("dvc_get_dataset_version", {}),
        ("mlflow_list_runs", {"max_results": 5}),
        ("airflow_list_dag_runs", {"limit": 5}),
        ("k8s_list_workloads", {"namespace": ns}),
    ]


def compose(intent: str, results: list[ToolResult]) -> str:
    facts, gaps, actions = [], [], []
    for r in results:
        if r.status == "ok":
            facts.append(f"- {r.summary} [{r.evidence_id}]")
        elif r.status == "requires_confirmation" and r.confirmation:
            actions.append(
                f"- **Approval required:** {r.confirmation.description} - confirmation id "
                f"`{r.confirmation.confirmation_id}` (nothing has been executed yet) [{r.evidence_id}]"
            )
        elif r.status == "denied":
            gaps.append(f"- Request refused by policy: {r.error} [{r.evidence_id}]")
        else:
            gaps.append(
                f"- `{r.tool}` returned **{r.status}**: {r.error} - no facts from this source are reported [{r.evidence_id}]"
            )
        if not r.ok:
            continue
        d = r.data or {}
        if r.tool == "assess_retraining_need":
            actions += [f"- {a} [{r.evidence_id}]" for a in d.get("recommended_actions", [])]
            gaps += [f"- {u} [{r.evidence_id}]" for u in d.get("uncertainties", [])]
        elif r.tool in ("mlflow_find_best_run", "mlflow_compare_runs"):
            caveats = d.get("caveats", [])
            gaps += [f"- {c} [{r.evidence_id}]" for c in caveats[:-1]]
            if caveats:
                actions.append(f"- {caveats[-1]} [{r.evidence_id}]")
        elif r.tool == "k8s_diagnose_deployment":
            actions += [f"- {f['issue']}: {f['suggested_fix']} [{r.evidence_id}]" for f in d.get("findings", [])]
        elif r.tool == "k8s_get_deployed_version" and d.get("consistency_issues"):
            gaps += [f"- Version inconsistency: {i} [{r.evidence_id}]" for i in d["consistency_issues"]]
        elif r.tool == "dvc_reproduce_dataset":
            actions.append("- Commands:\n  ```\n  " + "\n  ".join(d.get("commands", [])) + "\n  ```")

    ok = [r for r in results if r.status == "ok"]
    if not results:
        summary = "No tools were run for this request."
    elif not ok and gaps:
        summary = "I could not verify anything for this request because the required systems are unavailable or the request was refused; no operational results are reported."
    elif any(r.status == "requires_confirmation" for r in results):
        summary = "The requested action needs your explicit approval before anything changes."
    else:
        summary = (
            ok[-1].summary
            if len(ok) == 1
            else f"Collected evidence from {len({r.system for r in ok})} system(s); details below."
        )

    out = [f"**Summary** - {summary}", "", "**Verified facts**", *(facts or ["- none"])]
    if gaps:
        out += ["", "**Uncertainties**", *dict.fromkeys(gaps)]
    out += [
        "",
        "**Recommended actions**",
        *(dict.fromkeys(actions) or ["- No action required based on the collected evidence."]),
    ]
    out += ["", f"_Offline planner mode (intent: `{intent}`) - answer composed strictly from tool output._"]
    return "\n".join(out)
