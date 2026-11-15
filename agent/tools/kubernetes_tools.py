"""Kubernetes operations tools (official ``kubernetes`` Python client).

* Read AND write access is restricted to the namespaces in ``kubernetes.allowed_namespaces``.
* restart / scale / rollback are MUTATE tools: they are policy-checked first (namespace, mutable
  deployment list, replica bounds) and then require an explicit human confirmation.
* ``k8s_diagnose_deployment`` turns raw cluster state into evidence-backed findings with rule-based
  root-cause classification (ImagePullBackOff, CrashLoopBackOff, OOMKilled, failing probes, ...).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.safety import (
    PolicyError,
    ServiceUnavailable,
    ToolInputError,
    check_mutable_deployment,
    check_namespace,
    check_replicas,
)
from agent.schemas import PermissionLevel
from agent.tools.base import ToolContext, ToolOutput, ToolRegistry

_NAME = r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$"
DEFAULT_DEPLOYMENT = "inference-api"
APP_SELECTOR = "app.kubernetes.io/part-of=mlops-agent"


class K8sClients:
    def __init__(self, core: Any, apps: Any) -> None:
        self.core = core
        self.apps = apps


def k8s_factory(ctx: ToolContext) -> K8sClients:
    try:
        from kubernetes.config.config_exception import ConfigException

        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover
        raise ServiceUnavailable("kubernetes client library not installed") from exc
    try:
        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config(context=ctx.settings.kube_context or None)
    except Exception as exc:
        raise ServiceUnavailable(f"No usable kubeconfig / in-cluster config: {exc}") from exc
    cfg = client.Configuration.get_default_copy()
    cfg.retries = 1
    api = client.ApiClient(cfg)
    return K8sClients(client.CoreV1Api(api), client.AppsV1Api(api))


def _k(ctx: ToolContext) -> K8sClients:
    return ctx.client("kubernetes")


def _call(fn, *args, **kwargs):
    """Translate ApiException into the agent's error taxonomy."""
    try:
        return fn(*args, _request_timeout=10, **kwargs)
    except Exception as exc:
        status = getattr(exc, "status", None)
        if status == 404:
            raise ToolInputError(f"Kubernetes object not found: {getattr(exc, 'reason', exc)}") from exc
        if status == 403:
            raise PolicyError(f"Kubernetes RBAC denied the request: {getattr(exc, 'reason', exc)}") from exc
        if status == 401:
            raise ServiceUnavailable("Kubernetes credentials rejected (401).") from exc
        raise


def _age(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    secs = int((datetime.now(UTC) - ts).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit}"
    return f"{secs}s"


def _pod_row(pod: Any) -> dict[str, Any]:
    statuses = pod.status.container_statuses or []
    ready = sum(1 for c in statuses if c.ready)
    containers = []
    for c in statuses:
        waiting = c.state.waiting if c.state else None
        terminated = c.state.terminated if c.state else None
        last = c.last_state.terminated if c.last_state else None
        containers.append(
            {
                "name": c.name,
                "image": c.image,
                "ready": c.ready,
                "restart_count": c.restart_count,
                "state": "waiting" if waiting else ("terminated" if terminated else "running"),
                "waiting_reason": waiting.reason if waiting else None,
                "waiting_message": (waiting.message or "")[:300] if waiting else None,
                "last_termination_reason": last.reason if last else None,
                "last_exit_code": last.exit_code if last else None,
            }
        )
    conditions = {c.type: c.status for c in (pod.status.conditions or [])}
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": f"{ready}/{len(statuses) or len(pod.spec.containers)}",
        "is_ready": conditions.get("Ready") == "True",
        "restarts": sum(c.restart_count for c in statuses),
        "node": pod.spec.node_name,
        "age": _age(pod.metadata.creation_timestamp),
        "labels": pod.metadata.labels or {},
        "conditions": conditions,
        "containers": containers,
    }


def _deployment_row(d: Any) -> dict[str, Any]:
    c0 = d.spec.template.spec.containers[0]
    st = d.status
    return {
        "name": d.metadata.name,
        "desired": d.spec.replicas,
        "ready": st.ready_replicas or 0,
        "available": st.available_replicas or 0,
        "updated": st.updated_replicas or 0,
        "unavailable": st.unavailable_replicas or 0,
        "image": c0.image,
        "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        "revision": (d.metadata.annotations or {}).get("deployment.kubernetes.io/revision"),
        "selector": d.spec.selector.match_labels or {},
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": (c.message or "")[:200]}
            for c in (st.conditions or [])
        ],
    }


def _events(ctx: ToolContext, ns: str, name_prefix: str | None = None, warnings_only: bool = False, limit: int = 30):
    evs = _call(_k(ctx).core.list_namespaced_event, ns).items
    rows = []
    for e in evs:
        obj = e.involved_object
        if name_prefix and not (obj.name or "").startswith(name_prefix):
            continue
        if warnings_only and e.type != "Warning":
            continue
        ts = e.last_timestamp or e.event_time or e.first_timestamp or e.metadata.creation_timestamp
        rows.append(
            {
                "type": e.type,
                "reason": e.reason,
                "object": f"{obj.kind}/{obj.name}",
                "message": (e.message or "")[:300],
                "count": e.count,
                "last_seen": ts.isoformat() if ts else None,
                "_sort": ts.timestamp() if ts else 0,
            }
        )
    rows.sort(key=lambda r: r["_sort"], reverse=True)
    for r in rows:
        r.pop("_sort")
    return rows[:limit]


_LOG_ERR = re.compile(r"(error|exception|traceback|fatal|failed|refused|killed)", re.I)


def _pod_logs(ctx: ToolContext, ns: str, pod: str, container: str | None, tail: int, previous: bool) -> list[str]:
    kwargs: dict[str, Any] = {"tail_lines": tail, "previous": previous}
    if container:
        kwargs["container"] = container
    text = _call(_k(ctx).core.read_namespaced_pod_log, pod, ns, **kwargs)
    return (text or "").splitlines()


# ------------------------------------------------------------------------- diagnosis rules


def diagnose(
    dep: dict[str, Any] | None,
    pods: list[dict[str, Any]],
    events: list[dict[str, Any]],
    ready_endpoints: int | None,
    logs: dict[str, list[str]],
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def add(severity: str, issue: str, evidence: str, fix: str) -> None:
        findings.append({"severity": severity, "issue": issue, "evidence": evidence, "suggested_fix": fix})

    if dep is None:
        add(
            "critical",
            "Deployment not found",
            "no Deployment object",
            "Apply kubernetes/ manifests (kubectl apply -k kubernetes/).",
        )
    elif dep["ready"] < (dep["desired"] or 0):
        add(
            "high",
            "Deployment below desired replicas",
            f"{dep['ready']}/{dep['desired']} ready, {dep['unavailable']} unavailable",
            "See pod-level findings below.",
        )

    for p in pods:
        for c in p["containers"]:
            wr, lr = c["waiting_reason"], c["last_termination_reason"]
            ev = f"pod {p['name']} container {c['name']}"
            if wr in ("ImagePullBackOff", "ErrImagePull", "InvalidImageName"):
                add(
                    "critical",
                    "Image cannot be pulled",
                    f"{ev}: {wr} for image {c['image']}. {c['waiting_message'] or ''}",
                    "Verify the image tag exists locally / in the registry (docker images), fix the tag in the "
                    "Deployment, or roll back to the previous revision.",
                )
            elif wr == "CreateContainerConfigError":
                add(
                    "critical",
                    "Container config error (missing ConfigMap/Secret key)",
                    f"{ev}: {c['waiting_message']}",
                    "Create the referenced ConfigMap/Secret or fix the key name.",
                )
            elif wr == "CrashLoopBackOff" or (lr and c["restart_count"] > 0):
                tail = [ln for ln in logs.get(p["name"], []) if _LOG_ERR.search(ln)][-3:]
                if lr == "OOMKilled":
                    add(
                        "critical",
                        "Container OOMKilled",
                        f"{ev}: last termination OOMKilled, {c['restart_count']} restarts",
                        "Raise the memory limit or reduce model/batch memory usage.",
                    )
                else:
                    add(
                        "critical" if wr == "CrashLoopBackOff" else "medium",
                        "Container crashing / restarting",
                        f"{ev}: {wr or 'restarted'}, last exit code {c['last_exit_code']} ({lr}), {c['restart_count']} restarts. "
                        f"Log evidence: {tail or 'none captured'}",
                        "Inspect the previous container logs; fix the configuration or code error shown.",
                    )
        if p["phase"] == "Pending":
            sched = [e for e in events if e["reason"] == "FailedScheduling" and p["name"] in e["object"]]
            add(
                "high",
                "Pod cannot be scheduled",
                f"pod {p['name']} Pending. {sched[0]['message'] if sched else ''}",
                "Lower resource requests or add cluster capacity.",
            )
        elif p["phase"] == "Running" and not p["is_ready"] and all(c["state"] == "running" for c in p["containers"]):
            probe = [e for e in events if "probe failed" in e["message"].lower() and p["name"] in e["object"]]
            add(
                "high",
                "Readiness probe failing",
                f"pod {p['name']} running but not Ready. "
                f"{probe[0]['message'] if probe else 'No probe event captured.'}",
                "Check the /ready endpoint and model loading; the pod receives no traffic until Ready.",
            )

    if ready_endpoints == 0:
        add(
            "critical",
            "Service has no ready endpoints",
            "0 pods behind the Service are Ready - requests to the prediction API will fail.",
            "Resolve the pod-level findings; the Service recovers automatically once a pod is Ready.",
        )

    warning_reasons = sorted({e["reason"] for e in events if e["type"] == "Warning"})
    if dep and not findings:
        status = "healthy"
    elif any(f["severity"] == "critical" for f in findings) and (ready_endpoints == 0 or (dep and dep["ready"] == 0)):
        status = "down"
    else:
        status = "degraded" if findings else "healthy"
    return {"status": status, "findings": findings, "recent_warning_reasons": warning_reasons}


# ------------------------------------------------------------------------- schemas


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NsArgs(_Base):
    namespace: str = Field("mlops-agent", pattern=_NAME)


class PodArgs(NsArgs):
    pod: str = Field(..., pattern=_NAME)


class LogArgs(PodArgs):
    container: str | None = Field(None, pattern=_NAME)
    tail: int = Field(100, ge=1, le=500)
    previous: bool = Field(False, description="Logs of the previous (crashed) container instance.")


class EventArgs(NsArgs):
    warnings_only: bool = False
    object_prefix: str | None = Field(
        None, pattern=_NAME, description="Only events for objects whose name starts with this."
    )
    limit: int = Field(30, ge=1, le=200)


class DeploymentArgs(NsArgs):
    deployment: str = Field(DEFAULT_DEPLOYMENT, pattern=_NAME)


class ScaleArgs(DeploymentArgs):
    replicas: int = Field(..., ge=0, le=50)


class RollbackArgs(DeploymentArgs):
    to_revision: int | None = Field(None, ge=1, description="Default: the previous revision.")


def _ns_precheck(args: NsArgs, ctx: ToolContext) -> None:
    check_namespace(args.namespace, ctx.policy)


def _mutation_precheck(args: DeploymentArgs, ctx: ToolContext) -> None:
    check_namespace(args.namespace, ctx.policy)
    check_mutable_deployment(args.deployment, ctx.policy)
    if isinstance(args, ScaleArgs):
        check_replicas(args.replicas, ctx.policy)


def _ready_endpoints(ctx: ToolContext, ns: str, selector: dict[str, str]) -> int:
    if not selector:
        return 0
    sel = ",".join(f"{k}={v}" for k, v in selector.items())
    pods = _call(_k(ctx).core.list_namespaced_pod, ns, label_selector=sel).items
    return sum(1 for p in pods if _pod_row(p)["is_ready"] and not p.metadata.deletion_timestamp)


def register(reg: ToolRegistry) -> None:
    @reg.tool(
        "k8s_list_workloads",
        "List Deployments and Pods (status, readiness, restarts, image) in an approved namespace.",
        NsArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def list_workloads(args: NsArgs, ctx: ToolContext) -> ToolOutput:
        k = _k(ctx)
        deps = [_deployment_row(d) for d in _call(k.apps.list_namespaced_deployment, args.namespace).items]
        pods = [_pod_row(p) for p in _call(k.core.list_namespaced_pod, args.namespace).items]
        for p in pods:
            p.pop("labels", None)
        dep_txt = ", ".join(f"{d['name']} {d['ready']}/{d['desired']} ready ({d['image']})" for d in deps) or "none"
        pod_txt = (
            ", ".join(f"{p['name']} {p['phase']} ready={p['ready']} restarts={p['restarts']}" for p in pods) or "none"
        )
        return ToolOutput(
            f"[{args.namespace}] Deployments: {dep_txt}. Pods: {pod_txt}.",
            {"namespace": args.namespace, "deployments": deps, "pods": pods},
        )

    @reg.tool(
        "k8s_get_pod_status",
        "Detailed status of one Pod: phase, conditions, container states, restart reasons, exit codes.",
        PodArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def pod_status(args: PodArgs, ctx: ToolContext) -> ToolOutput:
        pod = _call(_k(ctx).core.read_namespaced_pod, args.pod, args.namespace)
        row = _pod_row(pod)
        c = row["containers"][0] if row["containers"] else {}
        return ToolOutput(
            f"Pod {args.pod}: {row['phase']}, ready {row['ready']}, restarts {row['restarts']}"
            + (f", waiting: {c.get('waiting_reason')}" if c.get("waiting_reason") else "")
            + (
                f", last termination: {c.get('last_termination_reason')} (exit {c.get('last_exit_code')})"
                if c.get("last_termination_reason")
                else ""
            )
            + ".",
            row,
        )

    @reg.tool(
        "k8s_get_pod_logs",
        "Tail a Pod's logs (optionally the previous crashed instance) with extracted error lines.",
        LogArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def pod_logs(args: LogArgs, ctx: ToolContext) -> ToolOutput:
        cap = int(ctx.policy.get("kubernetes", {}).get("log_tail_lines_max", 500))
        lines = _pod_logs(ctx, args.namespace, args.pod, args.container, min(args.tail, cap), args.previous)
        errors = [ln[:400] for ln in lines if _LOG_ERR.search(ln)][-10:]
        return ToolOutput(
            f"{args.pod}{' (previous)' if args.previous else ''}: {len(lines)} lines, {len(errors)} error-like lines"
            + (f"; last: {errors[-1]}" if errors else "")
            + ".",
            {
                "pod": args.pod,
                "previous": args.previous,
                "error_lines": errors,
                "tail": [ln[:400] for ln in lines[-60:]],
            },
        )

    @reg.tool(
        "k8s_list_services",
        "List Services with type, ClusterIP, ports, selector and number of READY backing pods.",
        NsArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def services(args: NsArgs, ctx: ToolContext) -> ToolOutput:
        rows = []
        for s in _call(_k(ctx).core.list_namespaced_service, args.namespace).items:
            selector = s.spec.selector or {}
            rows.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [
                        {"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                        for p in s.spec.ports or []
                    ],
                    "selector": selector,
                    "ready_endpoints": _ready_endpoints(ctx, args.namespace, selector),
                }
            )
        return ToolOutput(
            f"[{args.namespace}] "
            + "; ".join(
                f"{r['name']} {r['type']} {r['cluster_ip']}:{','.join(str(p['port']) for p in r['ports'])} "
                f"ready endpoints={r['ready_endpoints']}"
                for r in rows
            )
            if rows
            else f"No services in {args.namespace}.",
            {"namespace": args.namespace, "services": rows},
        )

    @reg.tool(
        "k8s_get_events",
        "Recent Kubernetes events (scheduling, image pulls, probe failures, back-offs), newest first.",
        EventArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def events(args: EventArgs, ctx: ToolContext) -> ToolOutput:
        rows = _events(ctx, args.namespace, args.object_prefix, args.warnings_only, args.limit)
        warn = [r for r in rows if r["type"] == "Warning"]
        return ToolOutput(
            f"{len(rows)} event(s), {len(warn)} warning(s)"
            + (": " + "; ".join(f"{r['reason']} {r['object']} x{r['count']}" for r in warn[:5]) if warn else "")
            + ".",
            {"events": rows},
        )

    @reg.tool(
        "k8s_find_unhealthy_pods",
        "Find Pods that are not Ready, waiting (back-off) or have restarted, with reasons.",
        NsArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def unhealthy(args: NsArgs, ctx: ToolContext) -> ToolOutput:
        pods = [_pod_row(p) for p in _call(_k(ctx).core.list_namespaced_pod, args.namespace).items]
        bad = []
        for p in pods:
            reasons = [c["waiting_reason"] for c in p["containers"] if c["waiting_reason"]]
            reasons += [
                f"last terminated: {c['last_termination_reason']}"
                for c in p["containers"]
                if c["last_termination_reason"]
            ]
            if not p["is_ready"] or p["restarts"] > 0 or p["phase"] not in ("Running", "Succeeded"):
                bad.append(
                    {
                        "name": p["name"],
                        "phase": p["phase"],
                        "ready": p["ready"],
                        "restarts": p["restarts"],
                        "reasons": reasons or (["not ready"] if not p["is_ready"] else ["restarted"]),
                    }
                )
        return ToolOutput(
            f"{len(bad)} of {len(pods)} pod(s) unhealthy or restarted"
            + (
                ": " + "; ".join(f"{b['name']} ({', '.join(b['reasons'])}, restarts={b['restarts']})" for b in bad)
                if bad
                else "."
            ),
            {"total_pods": len(pods), "unhealthy": bad},
        )

    @reg.tool(
        "k8s_get_deployed_version",
        "Report the currently deployed application/model version (image tag, version label, model annotations, "
        "ConfigMap) and check they agree with each other and with the MLflow 'champion' alias.",
        DeploymentArgs,
        system="kubernetes",
        precheck=_ns_precheck,
    )
    def deployed_version(args: DeploymentArgs, ctx: ToolContext) -> ToolOutput:
        k = _k(ctx)
        d = _call(k.apps.read_namespaced_deployment, args.deployment, args.namespace)
        ann = d.spec.template.metadata.annotations or {}
        image = d.spec.template.spec.containers[0].image
        image_tag = image.rsplit(":", 1)[-1] if ":" in image else "latest"
        cm_version = None
        for src in d.spec.template.spec.containers[0].env_from or []:
            if src.config_map_ref:
                try:
                    cm = _call(k.core.read_namespaced_config_map, src.config_map_ref.name, args.namespace)
                    cm_version = (cm.data or {}).get("MODEL_VERSION")
                except ToolInputError:
                    cm_version = "<configmap missing>"
        data = {
            "deployment": args.deployment,
            "image": image,
            "image_tag": image_tag,
            "app_version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
            "model_name": ann.get("mlops.agent/model-name"),
            "model_version": ann.get("mlops.agent/model-version"),
            "model_run_id": ann.get("mlops.agent/model-run-id"),
            "dataset_version": ann.get("mlops.agent/dataset-version"),
            "configmap_model_version": cm_version,
            "rollout_revision": (d.metadata.annotations or {}).get("deployment.kubernetes.io/revision"),
            "ready": f"{d.status.ready_replicas or 0}/{d.spec.replicas}",
        }
        mismatches = []
        if data["model_version"] and cm_version and str(cm_version) != str(data["model_version"]):
            mismatches.append(
                f"annotation model-version={data['model_version']} but ConfigMap MODEL_VERSION={cm_version}"
            )
        if data["model_version"] and f"model-v{data['model_version']}" != image_tag:
            mismatches.append(f"image tag '{image_tag}' does not encode model version {data['model_version']}")
        try:  # optional cross-check with the registry
            client = ctx.client("mlflow")
            name = data["model_name"] or ctx.policy.get("mlflow", {}).get("registered_model_name")
            champion = client.get_model_version_by_alias(name, "champion")
            data["registry_champion_version"] = champion.version
            if data["model_version"] and str(champion.version) != str(data["model_version"]):
                mismatches.append(f"MLflow champion is v{champion.version} but v{data['model_version']} is deployed")
        except Exception as exc:
            data["registry_champion_version"] = f"<unavailable: {type(exc).__name__}>"
        data["consistency_issues"] = mismatches
        return ToolOutput(
            f"{args.deployment} runs {image} = model {data['model_name']} v{data['model_version']} "
            f"(run {str(data['model_run_id'])[:8]}, dataset {data['dataset_version']}), revision {data['rollout_revision']}, "
            f"ready {data['ready']}"
            + (f". Consistency issues: {mismatches}" if mismatches else ". Version metadata consistent."),
            data,
        )

    @reg.tool(
        "k8s_diagnose_deployment",
        "One-shot evidence-based diagnosis of a deployment ('why is my API unavailable?'): rollout state, pod "
        "states, probe failures, warning events, crash logs and Service endpoints, classified into findings "
        "with suggested fixes.",
        DeploymentArgs,
        system="kubernetes",
        precheck=_ns_precheck,
        timeout_s=45,
    )
    def diagnose_deployment(args: DeploymentArgs, ctx: ToolContext) -> ToolOutput:
        k = _k(ctx)
        try:
            dep = _deployment_row(_call(k.apps.read_namespaced_deployment, args.deployment, args.namespace))
        except ToolInputError:
            dep = None
        selector = dep["selector"] if dep else {"app.kubernetes.io/name": args.deployment}
        sel = ",".join(f"{a}={b}" for a, b in selector.items())
        pods = [_pod_row(p) for p in _call(k.core.list_namespaced_pod, args.namespace, label_selector=sel).items]
        evs = _events(ctx, args.namespace, None, False, 60)
        relevant = [e for e in evs if args.deployment in e["object"]]
        logs: dict[str, list[str]] = {}
        for p in pods:
            if p["restarts"] > 0 or any(c["waiting_reason"] == "CrashLoopBackOff" for c in p["containers"]):
                try:
                    logs[p["name"]] = _pod_logs(ctx, args.namespace, p["name"], None, 40, previous=True)
                except Exception:
                    logs[p["name"]] = []
        endpoints = _ready_endpoints(ctx, args.namespace, selector)
        result = diagnose(dep, pods, relevant, endpoints, logs)
        for p in pods:
            p.pop("labels", None)
        data = {
            "deployment": dep,
            "pods": pods,
            "ready_endpoints": endpoints,
            "events": relevant[:15],
            "crash_logs": {k2: v[-15:] for k2, v in logs.items()},
            **result,
        }
        top = "; ".join(f"[{f['severity']}] {f['issue']}" for f in result["findings"][:4]) or "no problems found"
        return ToolOutput(
            f"{args.deployment} is {result['status'].upper()}: "
            f"{dep['ready'] if dep else 0}/{dep['desired'] if dep else '?'} replicas ready, {endpoints} ready endpoint(s); {top}.",
            data,
        )

    # ----------------------------------------------------------------- mutations

    @reg.tool(
        "k8s_restart_deployment",
        "Rolling-restart a deployment (requires human approval; approved namespace/deployments only).",
        DeploymentArgs,
        system="kubernetes",
        permission=PermissionLevel.MUTATE,
        precheck=_mutation_precheck,
        describe_action=lambda a: f"Rolling restart of deployment {a.namespace}/{a.deployment}",
    )
    def restart(args: DeploymentArgs, ctx: ToolContext) -> ToolOutput:
        now = datetime.now(UTC).isoformat()
        body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}}
        _call(_k(ctx).apps.patch_namespaced_deployment, args.deployment, args.namespace, body)
        return ToolOutput(
            f"Rolling restart of {args.namespace}/{args.deployment} started at {now}.",
            {"deployment": args.deployment, "restarted_at": now},
        )

    @reg.tool(
        "k8s_scale_deployment",
        "Scale a deployment within the approved replica bounds (requires human approval).",
        ScaleArgs,
        system="kubernetes",
        permission=PermissionLevel.MUTATE,
        precheck=_mutation_precheck,
        describe_action=lambda a: f"Scale deployment {a.namespace}/{a.deployment} to {a.replicas} replicas",
    )
    def scale(args: ScaleArgs, ctx: ToolContext) -> ToolOutput:
        k = _k(ctx)
        before = _call(k.apps.read_namespaced_deployment, args.deployment, args.namespace).spec.replicas
        _call(
            k.apps.patch_namespaced_deployment_scale,
            args.deployment,
            args.namespace,
            {"spec": {"replicas": args.replicas}},
        )
        return ToolOutput(
            f"Scaled {args.namespace}/{args.deployment} from {before} to {args.replicas} replicas.",
            {"deployment": args.deployment, "from": before, "to": args.replicas},
        )

    @reg.tool(
        "k8s_rollback_deployment",
        "Roll a deployment back to its previous (or a given) revision (requires human approval).",
        RollbackArgs,
        system="kubernetes",
        permission=PermissionLevel.MUTATE,
        precheck=_mutation_precheck,
        describe_action=lambda a: (
            f"Roll back {a.namespace}/{a.deployment} to "
            f"{'revision ' + str(a.to_revision) if a.to_revision else 'the previous revision'}"
        ),
    )
    def rollback(args: RollbackArgs, ctx: ToolContext) -> ToolOutput:
        k = _k(ctx)
        dep = _call(k.apps.read_namespaced_deployment, args.deployment, args.namespace)
        current = int((dep.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "0"))
        sel = ",".join(f"{a}={b}" for a, b in (dep.spec.selector.match_labels or {}).items())
        rsets = []
        for rs in _call(k.apps.list_namespaced_replica_set, args.namespace, label_selector=sel).items:
            owners = rs.metadata.owner_references or []
            if any(o.kind == "Deployment" and o.name == args.deployment for o in owners):
                rev = int((rs.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "0"))
                rsets.append((rev, rs))
        rsets.sort(key=lambda x: x[0], reverse=True)
        if args.to_revision:
            target = next((rs for rev, rs in rsets if rev == args.to_revision), None)
        else:
            target = next((rs for rev, rs in rsets if rev < current), None)
        if target is None:
            raise ToolInputError(f"No earlier revision available for {args.deployment} (current revision {current}).")
        template = k.apps.api_client.sanitize_for_serialization(target.spec.template)
        template.get("metadata", {}).get("labels", {}).pop("pod-template-hash", None)
        _call(k.apps.patch_namespaced_deployment, args.deployment, args.namespace, {"spec": {"template": template}})
        image = target.spec.template.spec.containers[0].image
        target_rev = (target.metadata.annotations or {}).get("deployment.kubernetes.io/revision")
        return ToolOutput(
            f"Rolled back {args.namespace}/{args.deployment} from revision {current} to the template of "
            f"revision {target_rev} (image {image}).",
            {"from_revision": current, "to_revision": target_rev, "image": image},
        )
