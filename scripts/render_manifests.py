"""Render kubernetes/ (kustomize) for a specific release: image, version label, provenance annotations and
ConfigMap MODEL_VERSION. Prints YAML to stdout, ready for `kubectl apply -f -`.

    python scripts/render_manifests.py --image ghcr.io/me/mlops-inference:model-v1.2.0 \
        --metadata serving/model/metadata.json | kubectl apply -f -
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT, CONFIGMAP = "inference-api", "inference-config"


def render(image: str, meta: dict) -> str:
    rendered = subprocess.run(
        ["kubectl", "kustomize", "kubernetes/"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    docs = [d for d in yaml.safe_load_all(rendered) if d]
    version = str(meta.get("model_version", "unknown"))
    for d in docs:
        if d["kind"] == "ConfigMap" and d["metadata"]["name"] == CONFIGMAP:
            d["data"]["MODEL_VERSION"] = version
        if d["kind"] == "Deployment" and d["metadata"]["name"] == DEPLOYMENT:
            d["metadata"]["labels"]["app.kubernetes.io/version"] = f"model-v{version}"
            d["spec"]["template"]["metadata"]["annotations"].update(
                {
                    "mlops.agent/model-name": meta.get("model_name", "credit-risk-classifier"),
                    "mlops.agent/model-version": version,
                    "mlops.agent/model-run-id": meta.get("run_id") or "",
                    "mlops.agent/dataset-version": meta.get("dataset_version") or "",
                }
            )
            d["spec"]["template"]["spec"]["containers"][0]["image"] = image
    return yaml.safe_dump_all(docs, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--metadata", default=str(ROOT / "serving" / "model" / "metadata.json"))
    a = ap.parse_args()
    meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    sys.stdout.write(render(a.image, meta))


if __name__ == "__main__":
    main()
