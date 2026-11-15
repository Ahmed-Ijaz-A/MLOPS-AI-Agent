"""MLflow logging helpers. Every run records full provenance so the agent can answer
"which dataset version / commit trained this model?" from recorded facts, not guesses."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from training.data import CATEGORICAL, NUMERIC

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit-risk-classification")
MODEL_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL", "credit-risk-classifier")
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def _setup(tracking_uri: str | None, experiment: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def provenance_tags(prov: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "dataset.path": prov.get("path"),
        "dataset.version": prov.get("declared_version"),
        "dataset.md5": prov.get("tracked_md5"),
        "dataset.workspace_md5": prov.get("workspace_md5"),
        "dataset.in_sync_with_dvc": prov.get("in_sync_with_dvc"),
        "git.commit": prov.get("git_commit"),
        "git.branch": prov.get("git_branch"),
        "git.describe": prov.get("git_describe"),
    }
    return {k: str(v) for k, v in mapping.items() if v is not None}


def log_evaluation(
    *,
    model: Any,
    params: dict[str, Any],
    evaluation: dict[str, Any],
    provenance: dict[str, Any],
    validation: dict[str, Any] | None = None,
    preprocess_info: dict[str, Any] | None = None,
    extra_tags: dict[str, str] | None = None,
    sample: pd.DataFrame | None = None,
    tracking_uri: str | None = None,
    experiment: str = EXPERIMENT,
    run_name: str | None = None,
) -> str:
    _setup(tracking_uri, experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in evaluation["metrics"].items()})
        tags = {
            **provenance_tags(provenance),
            "model_type": str(params.get("model_type")),
            "quality_gate.status": "pending",
            **(extra_tags or {}),
        }
        mlflow.set_tags(tags)
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "confusion_matrix": evaluation.get("confusion_matrix"),
                "validation": validation,
                "preprocess": preprocess_info,
                "provenance": provenance,
            }
            p = Path(tmp) / "evaluation_report.json"
            p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(p))
        input_example = sample[[*NUMERIC, *CATEGORICAL]].head(3) if sample is not None else None
        try:
            # MLflow 3 serialises sklearn with skops (no arbitrary pickle code execution). Only sklearn's own
            # tree structure - used by random forest / gradient boosting - is added to the trusted list.
            mlflow.sklearn.log_model(
                model, name="model", input_example=input_example, skops_trusted_types=SKOPS_TRUSTED_TYPES
            )
        except TypeError:  # MLflow 2.x signature
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)
        return run.info.run_id


def record_quality_gate(run_id: str, passed: bool, reasons: list[str], tracking_uri: str | None = None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_tag(run_id, "quality_gate.status", "passed" if passed else "failed")
    client.set_tag(run_id, "quality_gate.reasons", " | ".join(reasons)[:4900])
    client.log_metric(run_id, "quality_gate_passed", 1.0 if passed else 0.0)


def register_candidate(
    run_id: str,
    model_name: str = MODEL_NAME,
    alias: str = "candidate",
    tracking_uri: str | None = None,
) -> str:
    """Register a gate-approved model and point the ``candidate`` alias at it.

    Promotion to ``champion`` (what Kubernetes serves) is a separate, human-approved step.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    if run.data.tags.get("quality_gate.status") != "passed":
        raise PermissionError(f"run {run_id} has not passed the quality gate; refusing to register")
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    for key in ("dataset.version", "dataset.md5", "git.commit", "model_type"):
        if key in run.data.tags:
            client.set_model_version_tag(model_name, mv.version, key, run.data.tags[key])
    client.set_model_version_tag(model_name, mv.version, "quality_gate", "passed")
    client.set_model_version_tag(model_name, mv.version, "f1", str(run.data.metrics.get("f1")))
    client.set_registered_model_alias(model_name, alias, mv.version)
    return str(mv.version)
