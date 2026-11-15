"""Kubernetes tools: reads, diagnosis rules, and guarded mutations (fake API objects)."""

from __future__ import annotations

import pytest

from agent.tools.base import RequestTrace
from tests.fakes import (
    ApiError,
    FakeApps,
    FakeCore,
    FakeMlflow,
    fake_version,
    healthy_cluster,
    k8s_clients,
    make_deployment,
    make_event,
    make_pod,
    make_replicaset,
    make_service,
)

NS_ARGS = {"namespace": "mlops-agent"}


def run(make_executor, k8s, tool, args=None, **clients):
    return make_executor({"kubernetes": k8s, **clients}).execute(tool, {**NS_ARGS, **(args or {})})


def test_list_workloads_reports_deployments_and_pods(make_executor):
    r = run(make_executor, healthy_cluster(), "k8s_list_workloads")
    assert r.ok and r.data["deployments"][0]["desired"] == 2
    assert len(r.data["pods"]) == 2 and "2/2 ready" in r.summary


def test_healthy_deployment_diagnosed_healthy(make_executor):
    r = run(make_executor, healthy_cluster(), "k8s_diagnose_deployment")
    assert r.ok and r.data["status"] == "healthy" and r.data["findings"] == []


def test_image_pull_backoff_diagnosed(make_executor):
    pods = [
        make_pod(
            f"inference-api-x-{i}",
            ready=False,
            waiting="ImagePullBackOff",
            waiting_msg='Back-off pulling image "mlops-inference:model-v99"',
            image="mlops-inference:model-v99",
        )
        for i in range(2)
    ]
    k8s = k8s_clients(
        FakeCore(
            pods=pods,
            services=[make_service()],
            events=[make_event("Failed", "inference-api-x-0", "Failed to pull image")],
        ),
        FakeApps(deployments=[make_deployment(ready=0)]),
    )
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    assert r.data["status"] == "down"
    issues = [f["issue"] for f in r.data["findings"]]
    assert "Image cannot be pulled" in issues and "Service has no ready endpoints" in issues
    assert "model-v99" in r.data["findings"][1]["evidence"] or "model-v99" in str(r.data["findings"])


def test_crashloop_diagnosed_with_log_evidence(make_executor):
    pods = [
        make_pod(
            "inference-api-c-1", ready=False, restarts=5, waiting="CrashLoopBackOff", last_term="Error", exit_code=3
        )
    ]
    logs = {("inference-api-c-1", True): "INFO starting\nERROR FATAL: model file not found at MODEL_PATH=/wrong\n"}
    k8s = k8s_clients(
        FakeCore(pods=pods, services=[make_service()], logs=logs),
        FakeApps(deployments=[make_deployment(replicas=1, ready=0)]),
    )
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    crash = next(f for f in r.data["findings"] if f["issue"] == "Container crashing / restarting")
    assert crash["severity"] == "critical" and "model file not found" in crash["evidence"]
    assert r.data["crash_logs"]["inference-api-c-1"]


def test_oomkilled_diagnosed(make_executor):
    pods = [
        make_pod(
            "inference-api-o-1",
            ready=False,
            restarts=2,
            waiting="CrashLoopBackOff",
            last_term="OOMKilled",
            exit_code=137,
        )
    ]
    k8s = k8s_clients(FakeCore(pods=pods, services=[make_service()]), FakeApps(deployments=[make_deployment(ready=0)]))
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    assert any(f["issue"] == "Container OOMKilled" for f in r.data["findings"])


def test_failing_readiness_probe_diagnosed(make_executor):
    pods = [make_pod("inference-api-r-1", ready=False), make_pod("inference-api-r-2")]
    events = [
        make_event("Unhealthy", "inference-api-r-1", "Readiness probe failed: HTTP probe failed with statuscode: 503")
    ]
    k8s = k8s_clients(
        FakeCore(pods=pods, services=[make_service()], events=events), FakeApps(deployments=[make_deployment(ready=1)])
    )
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    probe = next(f for f in r.data["findings"] if f["issue"] == "Readiness probe failing")
    assert "503" in probe["evidence"] and r.data["status"] == "degraded"


def test_pending_pod_scheduling_diagnosed(make_executor):
    pods = [make_pod("inference-api-p-1", phase="Pending", ready=False)]
    events = [make_event("FailedScheduling", "inference-api-p-1", "0/1 nodes are available: Insufficient memory.")]
    k8s = k8s_clients(
        FakeCore(pods=pods, services=[make_service()], events=events),
        FakeApps(deployments=[make_deployment(replicas=1, ready=0)]),
    )
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    assert any("Insufficient memory" in f["evidence"] for f in r.data["findings"])


def test_missing_deployment_is_critical(make_executor):
    k8s = k8s_clients(FakeCore(), FakeApps())
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    assert r.data["findings"][0]["issue"] == "Deployment not found"


def test_unhealthy_pods_found_with_reasons(make_executor):
    pods = [make_pod("ok-1"), make_pod("bad-1", ready=False, restarts=4, waiting="CrashLoopBackOff", last_term="Error")]
    r = run(make_executor, k8s_clients(FakeCore(pods=pods)), "k8s_find_unhealthy_pods")
    assert len(r.data["unhealthy"]) == 1 and "CrashLoopBackOff" in r.summary


def test_pod_status_and_logs(make_executor):
    pods = [make_pod("p-1", ready=False, restarts=1, last_term="Error", exit_code=1)]
    k8s = k8s_clients(FakeCore(pods=pods, logs={("p-1", True): "Traceback\nValueError: bad config"}))
    st = run(make_executor, k8s, "k8s_get_pod_status", {"pod": "p-1"})
    assert st.ok and "exit 1" in st.summary
    logs = run(make_executor, k8s, "k8s_get_pod_logs", {"pod": "p-1", "previous": True})
    assert logs.data["error_lines"] == ["Traceback", "ValueError: bad config"]


def test_services_report_ready_endpoints(make_executor):
    pods = [make_pod("inference-api-1"), make_pod("inference-api-2", ready=False)]
    r = run(make_executor, k8s_clients(FakeCore(pods=pods, services=[make_service()])), "k8s_list_services")
    svc = r.data["services"][0]
    assert svc["type"] == "ClusterIP" and svc["ready_endpoints"] == 1


def test_events_newest_first_and_warning_filter(make_executor):
    events = [
        make_event("Pulled", "p", "ok", etype="Normal", minutes_ago=1),
        make_event("BackOff", "p", "Back-off restarting", minutes_ago=3),
        make_event("Failed", "p", "older", minutes_ago=9),
    ]
    r = run(make_executor, k8s_clients(FakeCore(events=events)), "k8s_get_events", {"warnings_only": True})
    assert [e["reason"] for e in r.data["events"]] == ["BackOff", "Failed"]


def test_deployed_version_consistent(make_executor):
    mlflow = FakeMlflow(versions={"3": fake_version("3", "a" * 32)}, aliases={"champion": "3"})
    r = run(make_executor, healthy_cluster(), "k8s_get_deployed_version", mlflow=mlflow)
    assert r.data["model_version"] == "3" and r.data["consistency_issues"] == []
    assert "consistent" in r.summary


def test_deployed_version_detects_drift_from_registry_and_configmap(make_executor):
    mlflow = FakeMlflow(versions={"4": fake_version("4", "b" * 32)}, aliases={"champion": "4"})
    core = FakeCore(pods=[], configmaps={"inference-config": {"MODEL_VERSION": "2"}})
    k8s = k8s_clients(core, FakeApps(deployments=[make_deployment()]))
    r = run(make_executor, k8s, "k8s_get_deployed_version", mlflow=mlflow)
    issues = " ".join(r.data["consistency_issues"])
    assert "MODEL_VERSION=2" in issues and "champion is v4" in issues


def test_deployed_version_works_when_mlflow_down(make_executor):
    r = run(make_executor, healthy_cluster(), "k8s_get_deployed_version")
    assert r.ok and r.data["registry_champion_version"].startswith("<unavailable")


# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("tool", ["k8s_list_workloads", "k8s_get_events", "k8s_diagnose_deployment"])
def test_reads_outside_approved_namespace_denied(make_executor, tool):
    r = make_executor({"kubernetes": healthy_cluster()}).execute(tool, {"namespace": "kube-system"})
    assert r.status == "denied"


def test_restart_requires_confirmation_and_does_not_patch(make_executor):
    k8s = healthy_cluster()
    r = run(make_executor, k8s, "k8s_restart_deployment", {"deployment": "inference-api"})
    assert r.status == "requires_confirmation" and k8s.apps.patches == []


@pytest.mark.parametrize(
    "tool,args",
    [
        ("k8s_restart_deployment", {"namespace": "kube-system", "deployment": "inference-api"}),
        ("k8s_restart_deployment", {"namespace": "default", "deployment": "inference-api"}),
        ("k8s_scale_deployment", {"namespace": "mlops-agent", "deployment": "inference-api", "replicas": 40}),
        ("k8s_scale_deployment", {"namespace": "mlops-agent", "deployment": "coredns", "replicas": 2}),
        ("k8s_rollback_deployment", {"namespace": "production", "deployment": "inference-api"}),
    ],
)
def test_unauthorized_kubernetes_modifications_rejected(make_executor, tool, args):
    k8s = healthy_cluster()
    ex = make_executor({"kubernetes": k8s})
    r = ex.execute(tool, args)
    assert r.status == "denied"
    assert r.confirmation is None and not ex.confirmations.list()  # never even offered for approval
    assert k8s.apps.patches == []


def test_approved_scale_patches_cluster(make_executor):
    k8s = healthy_cluster()
    ex = make_executor({"kubernetes": k8s})
    pending = ex.execute("k8s_scale_deployment", {**NS_ARGS, "deployment": "inference-api", "replicas": 3}).confirmation
    _, result = ex.execute_confirmed(pending.confirmation_id, RequestTrace("req"))
    assert result.ok and k8s.apps.patches == [("scale", "inference-api", {"spec": {"replicas": 3}})]


def test_approved_restart_sets_restarted_at_annotation(make_executor):
    k8s = healthy_cluster()
    ex = make_executor({"kubernetes": k8s})
    pending = ex.execute("k8s_restart_deployment", {**NS_ARGS, "deployment": "inference-api"}).confirmation
    ex.execute_confirmed(pending.confirmation_id, RequestTrace("req"))
    body = k8s.apps.patches[0][2]
    assert "kubectl.kubernetes.io/restartedAt" in body["spec"]["template"]["metadata"]["annotations"]


def test_rollback_targets_previous_revision(make_executor):
    rsets = [
        make_replicaset(4, "mlops-inference:model-v99"),
        make_replicaset(3, "mlops-inference:model-v3"),
        make_replicaset(2, "mlops-inference:model-v2"),
    ]
    k8s = k8s_clients(FakeCore(), FakeApps(deployments=[make_deployment(revision="4")], replicasets=rsets))
    ex = make_executor({"kubernetes": k8s})
    pending = ex.execute("k8s_rollback_deployment", {**NS_ARGS, "deployment": "inference-api"}).confirmation
    _, result = ex.execute_confirmed(pending.confirmation_id, RequestTrace("req"))
    assert result.ok and result.data["image"] == "mlops-inference:model-v3"
    template = k8s.apps.patches[0][2]["spec"]["template"]
    assert "pod-template-hash" not in template["metadata"]["labels"]


def test_rbac_forbidden_maps_to_denied(make_executor):
    k8s = k8s_clients(FakeCore(error=ApiError(403, "Forbidden")), FakeApps(error=ApiError(403, "Forbidden")))
    assert run(make_executor, k8s, "k8s_list_workloads").status == "denied"


def test_cluster_unreachable_maps_to_unavailable(make_executor):
    err = RuntimeError("HTTPSConnectionPool(host='127.0.0.1', port=6443): Max retries exceeded")
    k8s = k8s_clients(FakeCore(error=err), FakeApps(error=err))
    r = run(make_executor, k8s, "k8s_list_workloads")
    assert r.status == "unavailable" and r.data is None


def test_no_kubeconfig_is_unavailable(make_executor):
    r = make_executor().execute("k8s_list_workloads", NS_ARGS)
    assert r.status == "unavailable"


def test_unknown_pod_is_invalid_request(make_executor):
    assert run(make_executor, healthy_cluster(), "k8s_get_pod_status", {"pod": "ghost"}).status == "invalid_request"


def test_crash_correlated_with_misconfigured_configmap(make_executor):
    pods = [
        make_pod(
            "inference-api-m-1", ready=False, restarts=3, waiting="CrashLoopBackOff", last_term="Error", exit_code=3
        )
    ]
    logs = {
        ("inference-api-m-1", True): "ERROR FATAL: model file not found at MODEL_PATH=/app/model/model-MISSING.joblib"
    }
    core = FakeCore(
        pods=pods,
        services=[make_service()],
        logs=logs,
        configmaps={"inference-config": {"MODEL_PATH": "/app/model/model-MISSING.joblib", "LOG_LEVEL": "INFO"}},
    )
    k8s = k8s_clients(core, FakeApps(deployments=[make_deployment(replicas=1, ready=0)]))
    r = run(make_executor, k8s, "k8s_diagnose_deployment")
    cm = next(f for f in r.data["findings"] if f["issue"] == "Misconfigured ConfigMap value")
    assert "MODEL_PATH" in cm["evidence"] and "inference-config" in cm["suggested_fix"]
    assert not any("LOG_LEVEL" in f["evidence"] for f in r.data["findings"])  # 'INFO' isn't in an error line
    assert r.data["configmaps"]["inference-config"]["MODEL_PATH"].endswith("MISSING.joblib")


def test_scheduled_pod_waiting_on_image_is_not_a_scheduling_problem(make_executor):
    good = [make_pod("inference-api-ok-1"), make_pod("inference-api-ok-2")]
    pulling = make_pod(
        "inference-api-new-1",
        phase="Pending",
        ready=False,
        waiting="ImagePullBackOff",
        image="mlops-inference:model-v404",
    )
    pulling.status.container_statuses = [
        make_pod(
            "x", ready=False, waiting="ImagePullBackOff", image="mlops-inference:model-v404"
        ).status.container_statuses[0]
    ]
    k8s = k8s_clients(
        FakeCore(pods=[*good, pulling], services=[make_service()]), FakeApps(deployments=[make_deployment(ready=2)])
    )
    issues = [f["issue"] for f in run(make_executor, k8s, "k8s_diagnose_deployment").data["findings"]]
    assert "Image cannot be pulled" in issues and "Pod cannot be scheduled" not in issues
