"""Policy allowlists and the human-confirmation store."""

from __future__ import annotations

import time

import pytest

from agent.safety import (
    AuditLog,
    ConfirmationStore,
    PolicyError,
    check_command,
    check_conf,
    check_mutable_deployment,
    check_namespace,
    check_replicas,
    check_triggerable_dag,
    fingerprint,
)
from agent.schemas import PermissionLevel


@pytest.mark.parametrize("ns", ["mlops-agent"])
def test_approved_namespace_allowed(ns, policy):
    assert check_namespace(ns, policy) == ns


@pytest.mark.parametrize("ns", ["default", "kube-system", "production", "mlops-agent-2", ""])
def test_unapproved_namespaces_rejected(ns, policy):
    with pytest.raises(PolicyError, match="not approved"):
        check_namespace(ns, policy)


@pytest.mark.parametrize("replicas", [1, 2, 5])
def test_replicas_within_bounds(replicas, policy):
    assert check_replicas(replicas, policy) == replicas


@pytest.mark.parametrize("replicas", [0, 6, 50])
def test_replicas_out_of_bounds_rejected(replicas, policy):
    with pytest.raises(PolicyError, match="outside the approved range"):
        check_replicas(replicas, policy)


def test_only_mutable_deployments_can_change(policy):
    assert check_mutable_deployment("inference-api", policy) == "inference-api"
    with pytest.raises(PolicyError):
        check_mutable_deployment("coredns", policy)


def test_only_approved_dags_can_be_triggered(policy):
    assert check_triggerable_dag("ml_training_pipeline", policy)
    with pytest.raises(PolicyError):
        check_triggerable_dag("delete_everything", policy)


def test_dag_conf_keys_are_allowlisted(policy):
    assert check_conf({"reason": "new data", "inject_failure": "none"}, policy)
    with pytest.raises(PolicyError, match="not allowed"):
        check_conf({"bash_command": "rm -rf /"}, policy)
    with pytest.raises(PolicyError, match="short scalar"):
        check_conf({"reason": "x" * 500}, policy)


@pytest.mark.parametrize(
    "tool,args",
    [("dvc", ["status", "--json"]), ("dvc", ["diff", "HEAD~1"]), ("git", ["log", "-n5"]), ("git", ["show", "HEAD:x"])],
)
def test_allowlisted_commands_pass(tool, args, policy):
    assert check_command(tool, args, policy) == args


@pytest.mark.parametrize(
    "tool,args",
    [
        ("dvc", ["push"]),
        ("dvc", ["gc", "-w"]),
        ("dvc", ["remove", "data.dvc"]),
        ("dvc", ["checkout"]),
        ("git", ["checkout", "main"]),
        ("git", ["reset", "--hard"]),
        ("git", ["push", "--force"]),
        ("rm", ["-rf", "/"]),
        ("dvc", []),
    ],
)
def test_dangerous_commands_rejected(tool, args, policy):
    with pytest.raises(PolicyError):
        check_command(tool, args, policy)


@pytest.mark.parametrize("bad", ["HEAD; rm -rf /", "x | cat", "$(whoami)", "a && b", "`id`", "a > b"])
def test_shell_metacharacters_rejected(bad, policy):
    with pytest.raises(PolicyError, match="disallowed characters"):
        check_command("git", ["log", bad], policy)


def test_fingerprint_is_stable_and_argument_sensitive():
    a = fingerprint("k8s_scale_deployment", {"replicas": 3, "namespace": "mlops-agent"})
    b = fingerprint("k8s_scale_deployment", {"namespace": "mlops-agent", "replicas": 3})
    c = fingerprint("k8s_scale_deployment", {"namespace": "mlops-agent", "replicas": 4})
    assert a == b != c


def test_confirmation_consumed_exactly_once():
    store = ConfirmationStore()
    action = store.create("k8s_restart_deployment", {"deployment": "inference-api"}, "restart", PermissionLevel.MUTATE)
    assert store.consume(action.confirmation_id).tool == "k8s_restart_deployment"
    with pytest.raises(PolicyError, match="unknown, already used, or expired"):
        store.consume(action.confirmation_id)


def test_confirmation_expires():
    store = ConfirmationStore(ttl_seconds=0)
    action = store.create("k8s_restart_deployment", {}, "restart", PermissionLevel.MUTATE)
    time.sleep(0.01)
    with pytest.raises(PolicyError):
        store.consume(action.confirmation_id)
    assert any(e["event"] == "action_expired" for e in store.audit.events)


def test_confirmation_bound_to_session():
    store = ConfirmationStore()
    action = store.create("k8s_restart_deployment", {}, "restart", PermissionLevel.MUTATE, session_id="alice")
    with pytest.raises(PolicyError, match="different session"):
        store.consume(action.confirmation_id, session_id="mallory")


def test_tampered_pending_action_rejected():
    store = ConfirmationStore()
    action = store.create("k8s_scale_deployment", {"replicas": 2}, "scale", PermissionLevel.MUTATE)
    action.arguments["replicas"] = 50  # simulate tampering after the human saw it
    with pytest.raises(PolicyError, match="modified"):
        store.consume(action.confirmation_id)


def test_audit_log_writes_jsonl_and_redacts(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record("action_requested", tool="x", arguments={"password": "hunter2", "replicas": 2})
    text = (tmp_path / "audit.jsonl").read_text()
    assert "hunter2" not in text and "REDACTED" in text and '"replicas": 2' in text


def test_list_filters_by_session():
    store = ConfirmationStore()
    store.create("t", {}, "a", PermissionLevel.MUTATE, session_id="s1")
    store.create("t", {"x": 1}, "b", PermissionLevel.MUTATE, session_id="s2")
    assert len(store.list("s1")) == 1 and len(store.list()) == 2
