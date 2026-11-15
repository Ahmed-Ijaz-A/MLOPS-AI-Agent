"""Promote a registered model and deploy it to Kubernetes (a deliberate, human-run release step).

    python scripts/deploy_model.py --version 3          # promote v3 -> champion, build, deploy
    python scripts/deploy_model.py --alias candidate    # deploy whatever 'candidate' points at

Steps
 1. resolve the model version in the MLflow registry and set alias ``champion``
 2. export the sklearn pipeline + provenance metadata into serving/model/
 3. docker build -t mlops-inference:model-v<N>
 4. kubectl apply -k kubernetes/  (namespace, ConfigMap, Deployment, Service, PDB, RBAC)
 5. patch image, version label, provenance annotations and ConfigMap MODEL_VERSION; wait for rollout
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NAMESPACE = "mlops-agent"
DEPLOYMENT = "inference-api"


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import joblib
    import mlflow
    from mlflow.tracking import MlflowClient

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--version")
    g.add_argument("--alias")
    ap.add_argument("--model-name", default="credit-risk-classifier")
    ap.add_argument("--no-promote", action="store_true", help="deploy without moving the champion alias")
    ap.add_argument("--context", default=os.environ.get("KUBE_CONTEXT", ""))
    a = ap.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    mv = (
        client.get_model_version(a.model_name, a.version)
        if a.version
        else client.get_model_version_by_alias(a.model_name, a.alias)
    )
    run = client.get_run(mv.run_id)
    if run.data.tags.get("quality_gate.status") != "passed":
        sys.exit(f"refusing to deploy v{mv.version}: quality gate status is {run.data.tags.get('quality_gate.status')}")
    if not a.no_promote:
        client.set_registered_model_alias(a.model_name, "champion", mv.version)
        print(f"alias champion -> v{mv.version}")

    out = ROOT / "serving" / "model"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    model = mlflow.sklearn.load_model(f"models:/{a.model_name}/{mv.version}")
    joblib.dump(model, out / "model.joblib")
    meta = {
        "model_name": a.model_name,
        "model_version": str(mv.version),
        "run_id": mv.run_id,
        "dataset_version": run.data.tags.get("dataset.version"),
        "dataset_md5": run.data.tags.get("dataset.md5"),
        "git_commit": run.data.tags.get("git.commit"),
        "metrics": {k: run.data.metrics[k] for k in ("f1", "recall", "roc_auc", "accuracy") if k in run.data.metrics},
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    image = f"mlops-inference:model-v{mv.version}"
    sh("docker", "build", "-t", image, "--build-arg", f"MODEL_VERSION={mv.version}", ".")
    ctx = ["--context", a.context] if a.context else []
    sh("kubectl", *ctx, "apply", "-k", "kubernetes/")

    patch = {
        "metadata": {"labels": {"app.kubernetes.io/version": f"model-v{mv.version}"}},
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "mlops.agent/model-name": a.model_name,
                        "mlops.agent/model-version": str(mv.version),
                        "mlops.agent/model-run-id": mv.run_id,
                        "mlops.agent/dataset-version": meta["dataset_version"] or "",
                    }
                },
                "spec": {"containers": [{"name": DEPLOYMENT, "image": image}]},
            }
        },
    }
    sh(
        "kubectl",
        *ctx,
        "-n",
        NAMESPACE,
        "patch",
        "configmap",
        "inference-config",
        "--type",
        "merge",
        "-p",
        json.dumps({"data": {"MODEL_VERSION": str(mv.version)}}),
    )
    sh(
        "kubectl",
        *ctx,
        "-n",
        NAMESPACE,
        "patch",
        "deployment",
        DEPLOYMENT,
        "--type",
        "strategic",
        "-p",
        json.dumps(patch),
    )
    sh("kubectl", *ctx, "-n", NAMESPACE, "rollout", "status", f"deployment/{DEPLOYMENT}", "--timeout=180s")
    print(
        f"\nDeployed {a.model_name} v{mv.version} ({image}); dataset {meta['dataset_version']}; metrics {meta['metrics']}"
    )


if __name__ == "__main__":
    main()
