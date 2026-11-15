"""Static checks of the DAG, Kubernetes manifests and CI configuration (no cluster / Airflow needed)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DAG = ROOT / "dags" / "training_pipeline.py"


def _manifests():
    docs = []
    for f in sorted((ROOT / "kubernetes").glob("*.yaml")):
        if f.name != "kustomization.yaml":
            docs += [d for d in yaml.safe_load_all(f.read_text(encoding="utf-8")) if d]
    return docs


def _kind(kind, name=None):
    return next(d for d in _manifests() if d["kind"] == kind and (name is None or d["metadata"]["name"] == name))


# ------------------------------------------------------------------ DAG


@pytest.fixture(scope="module")
def dag_source():
    return DAG.read_text(encoding="utf-8")


def test_dag_parses_and_declares_required_stages(dag_source):
    tree = ast.parse(dag_source)
    tasks = [
        f.name
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and any(getattr(d, "id", getattr(getattr(d, "func", None), "id", None)) == "task" for d in f.decorator_list)
    ]
    assert tasks == [
        "validate_dataset",
        "preprocess_data",
        "train_model",
        "evaluate_and_log",
        "quality_gate",
        "register_model",
    ]


def test_dag_chain_order(dag_source):
    assert (
        "register_model(quality_gate(evaluate_and_log(train_model(preprocess_data(validate_dataset())))))" in dag_source
    )


def test_dag_has_retries_and_fail_fast_for_deterministic_failures(dag_source):
    assert '"retries": 2' in dag_source and "retry_delay" in dag_source
    assert dag_source.count("raise AirflowFailException") >= 3  # validation (x2) and the quality gate
    assert "@task(retries=0)\n    def quality_gate" in dag_source


def test_dag_uses_airflow3_task_sdk(dag_source):
    assert "from airflow.sdk import" in dag_source and 'dag_id="ml_training_pipeline"' in dag_source


def test_dag_failure_injection_params(dag_source):
    for mode in ("corrupt_data", "low_quality", "flaky_training"):
        assert mode in dag_source


def test_dag_imports_in_real_airflow():
    airflow = pytest.importorskip("airflow", reason="Airflow not installed locally (validated in CI dag-integrity job)")
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAG), include_examples=False)
    assert not bag.import_errors and "ml_training_pipeline" in bag.dags
    assert airflow


# ------------------------------------------------------------------ Kubernetes


def test_deployment_requirements():
    dep = _kind("Deployment", "inference-api")
    assert dep["spec"]["replicas"] >= 2
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(c["resources"][kind])
    assert c["envFrom"][0]["configMapRef"]["name"] == _kind("ConfigMap")["metadata"]["name"]
    assert c["securityContext"]["allowPrivilegeEscalation"] is False


def test_service_is_clusterip_and_selects_deployment():
    svc, dep = _kind("Service", "inference-api"), _kind("Deployment", "inference-api")
    assert svc["spec"]["type"] == "ClusterIP"
    assert svc["spec"]["selector"].items() <= dep["spec"]["template"]["metadata"]["labels"].items()


def test_configmap_has_no_secrets():
    data = _kind("ConfigMap")["data"]
    assert not any(k for k in data if any(s in k.lower() for s in ("password", "secret", "token", "key")))


def test_all_objects_in_approved_namespace():
    for d in _manifests():
        if d["kind"] != "Namespace":
            assert d["metadata"]["namespace"] == "mlops-agent", d["metadata"]["name"]


def test_agent_rbac_is_least_privilege():
    role = _kind("Role")
    verbs = {v for r in role["rules"] for v in r["verbs"]}
    resources = {res for r in role["rules"] for res in r["resources"]}
    assert not verbs & {"delete", "create", "deletecollection", "*", "escalate", "bind"}
    assert "secrets" not in resources and "pods/exec" not in resources


def test_kustomization_lists_every_manifest():
    k = yaml.safe_load((ROOT / "kubernetes" / "kustomization.yaml").read_text())
    files = {f.name for f in (ROOT / "kubernetes").glob("*.yaml")} - {"kustomization.yaml"}
    assert set(k["resources"]) == files


# ------------------------------------------------------------------ repo hygiene


def test_ci_runs_lint_and_tests_on_pull_requests():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    triggers = ci.get("on") or ci.get(True)
    assert "pull_request" in triggers
    steps = " ".join(str(s.get("run", "")) for job in ci["jobs"].values() for s in job["steps"])
    assert "ruff check" in steps and "pytest" in steps


def test_no_secrets_committed():
    ignored = (ROOT / ".gitignore").read_text()
    assert ".env" in ignored
    for f in list(ROOT.glob("*.py")) + list((ROOT / "agent").rglob("*.py")) + list((ROOT / "kubernetes").glob("*")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        assert "sk-ant-api" not in text, f


def test_cd_pipeline_structure():
    cd = yaml.safe_load((ROOT / ".github" / "workflows" / "cd.yml").read_text(encoding="utf-8"))
    triggers = cd.get("on") or cd.get(True)
    assert triggers["workflow_run"]["workflows"] == ["CI"]  # main is delivered only after CI
    assert "v*.*.*" in triggers["push"]["tags"]
    jobs = cd["jobs"]
    assert set(jobs) == {"build-and-push", "deploy-staging", "release"}
    assert "conclusion == 'success'" in jobs["build-and-push"]["if"]
    assert jobs["build-and-push"]["permissions"]["packages"] == "write"
    steps = " ".join(str(s.get("run", "")) + str(s.get("with", "")) for s in jobs["build-and-push"]["steps"])
    assert "package_model.py" in steps and "Dockerfile.agent" in steps and "ghcr.io" in str(cd["env"]["REGISTRY"])
    assert jobs["deploy-staging"]["needs"] == "build-and-push"
    staging = " ".join(str(s.get("run", "")) for s in jobs["deploy-staging"]["steps"])
    assert "rollout status" in staging and "/predict" in staging and "agent.main" in staging
    assert jobs["release"]["if"].startswith("startsWith(github.ref, 'refs/tags/v')")
    assert jobs["release"]["permissions"]["contents"] == "write"


def test_agent_image_runs_as_non_root():
    text = (ROOT / "Dockerfile.agent").read_text(encoding="utf-8")
    assert "USER 10002" in text and "HEALTHCHECK" in text and "api.main:app" in text


def test_release_manifest_rendering():
    import shutil
    import subprocess
    import sys

    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not installed")
    meta = ROOT / "tests" / "_meta.json"
    meta.write_text('{"model_version": "9.9.9", "dataset_version": "v2.0", "run_id": null}', encoding="utf-8")
    try:
        out = subprocess.run(
            [
                sys.executable,
                "scripts/render_manifests.py",
                "--image",
                "ghcr.io/x/mlops-inference:model-v9.9.9",
                "--metadata",
                str(meta),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        meta.unlink()
    docs = {d["kind"]: d for d in yaml.safe_load_all(out) if d and d["kind"] in ("Deployment", "ConfigMap")}
    dep = docs["Deployment"]
    assert dep["spec"]["template"]["spec"]["containers"][0]["image"] == "ghcr.io/x/mlops-inference:model-v9.9.9"
    assert dep["spec"]["template"]["metadata"]["annotations"]["mlops.agent/model-version"] == "9.9.9"
    assert docs["ConfigMap"]["data"]["MODEL_VERSION"] == "9.9.9"
