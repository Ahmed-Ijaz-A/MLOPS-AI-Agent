"""Retraining policy (deterministic, evidence-based) and the grounding verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.grounding import verify
from agent.schemas import ToolResult
from agent.tools.airflow_tools import AirflowClient
from tests.fakes import AirflowMock, FakeMlflow, fake_run, fake_version, healthy_cluster, pipeline_runs

RUN = "a" * 32


def _airflow(gate_failed: bool):
    runs, tasks = pipeline_runs()
    if not gate_failed:
        runs = runs[1:]
    mock = AirflowMock(runs=runs, tasks=tasks)
    return mock, AirflowClient("http://airflow.test", "airflow", "pw", transport=mock.transport())


@pytest.fixture
def repo_with_md5(tmp_path, settings):
    def _make(md5: str):
        (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "raw" / "credit_risk.csv.dvc").write_text(
            f"outs:\n- md5: {md5}\n  size: 10\n  path: credit_risk.csv\n", encoding="utf-8"
        )
        return replace(settings, repo_root=tmp_path)

    return _make


def _assess(
    make_executor,
    s,
    *,
    md5_trained="md5v2",
    f1=0.66,
    days=2.0,
    gate_failed=False,
    observed=None,
    k8s=True,
    mlflow=True,
    airflow=True,
):
    clients = {}
    mock = None
    if mlflow:
        clients["mlflow"] = FakeMlflow(
            runs={RUN: fake_run(RUN, f1=f1, md5=md5_trained)},
            versions={"3": fake_version("3", RUN, days_ago=days)},
            aliases={"champion": "3"},
        )
    if k8s:
        clients["kubernetes"] = healthy_cluster()
    if airflow:
        mock, clients["airflow"] = _airflow(gate_failed)
    args = {"observed_metrics": observed} if observed else {}
    return make_executor(clients, settings_override=s).execute("assess_retraining_need", args), mock


def test_no_retraining_when_policy_satisfied(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v2"))
    assert r.data["decision"] == "no_retraining_needed"
    assert all(rule["fired"] is False for rule in r.data["rules"])


def test_retrain_when_data_changed_since_training(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v3"), md5_trained="md5v2")
    assert r.data["decision"] == "retrain_recommended"
    assert any(x["rule"] == "data_changed" and x["fired"] for x in r.data["rules"])


def test_retrain_when_model_too_old(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v2"), days=45)
    assert any(x["rule"] == "freshness" and x["fired"] for x in r.data["rules"])


def test_retrain_when_quality_gate_failed(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v2"), gate_failed=True)
    assert r.data["decision"] == "retrain_recommended"
    assert any("Investigate the failed quality gate" in a for a in r.data["recommended_actions"])


def test_retrain_on_supplied_degradation(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v2"), observed={"f1": 0.41})
    assert any(x["rule"] == "observed_degradation" and x["fired"] for x in r.data["rules"])


def test_insufficient_evidence_when_services_down(make_executor, repo_with_md5):
    r, _ = _assess(make_executor, repo_with_md5("md5v2"), k8s=False, mlflow=False, airflow=False)
    assert r.data["decision"] == "insufficient_evidence"
    assert len(r.data["uncertainties"]) >= 3


def test_assessment_never_triggers_training(make_executor, repo_with_md5):
    r, mock = _assess(make_executor, repo_with_md5("md5v3"), md5_trained="md5v2", gate_failed=True)
    assert r.data["decision"] == "retrain_recommended"
    assert mock.triggered == [] and all(req.method == "GET" or req.url.path == "/auth/token" for req in mock.requests)


# ------------------------------------------------------------------ grounding


def _ok(summary, data=None):
    return ToolResult(evidence_id="E1", tool="t", status="ok", summary=summary, data=data or {})


def test_grounded_values_pass():
    results = [_ok("best run abc12345 f1=0.6575", {"commit": "c0ffee1234567", "recall": 0.8})]
    report = verify("F1 is 0.6575 [E1], recall 80% from commit c0ffee1 (run abc12345).", results)
    assert report.grounded and report.checked_values >= 3


def test_rounded_values_accepted():
    assert verify("F1 = 0.66", [_ok("", {"f1": 0.6575})]).grounded


def test_invented_metric_is_flagged():
    report = verify("The best model reached F1 = 0.9431.", [_ok("", {"f1": 0.6575})])
    assert not report.grounded and "0.9431" in report.unsupported_values


def test_invented_commit_is_flagged():
    report = verify("Trained at commit deadbeef1.", [_ok("commit c0ffee1234")])
    assert "deadbeef1" in report.unsupported_values


def test_values_from_failed_tools_do_not_count():
    failed = ToolResult(evidence_id="E1", tool="t", status="unavailable", error="down 0.91")
    assert not verify("accuracy 0.91", [failed]).grounded


def test_citations_are_not_treated_as_values():
    assert verify("See [E12] and [E3].", []).grounded


def test_failure_injection_drills_are_not_evidence(make_executor, repo_with_md5):
    runs, tasks = pipeline_runs()
    genuine_ok = runs[1:]  # latest genuine run succeeded
    drill = {"dag_run_id": "manual__drill", "state": "failed", "conf": {"inject_failure": "low_quality"}}
    mock = AirflowMock(runs=[drill, *genuine_ok], tasks={**tasks, "manual__drill": tasks[runs[0]["dag_run_id"]]})
    clients = {
        "mlflow": FakeMlflow(
            runs={RUN: fake_run(RUN)}, versions={"3": fake_version("3", RUN)}, aliases={"champion": "3"}
        ),
        "kubernetes": healthy_cluster(),
        "airflow": AirflowClient("http://airflow.test", "airflow", "pw", transport=mock.transport()),
    }
    r = make_executor(clients, settings_override=repo_with_md5("md5v2")).execute("assess_retraining_need", {})
    gate = next(x for x in r.data["rules"] if x["rule"] == "failed_quality_gate")
    assert gate["fired"] is False and "drill" in gate["evidence"]
    assert r.data["facts"]["ignored_failure_injection_runs"][0]["inject_failure"] == "low_quality"
    assert r.data["decision"] == "no_retraining_needed"
