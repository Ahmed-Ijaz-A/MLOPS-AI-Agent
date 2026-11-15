"""Deliberately break (and restore) the deployment so the agent's diagnosis can be demonstrated.

    python scripts/inject_failure.py bad-image     # ImagePullBackOff: deploy a tag that does not exist
    python scripts/inject_failure.py bad-config    # CrashLoopBackOff: ConfigMap points MODEL_PATH at a missing file
    python scripts/inject_failure.py restore       # roll back to the last healthy configuration

Pipeline failures are injected through the DAG instead, e.g. ask the agent:
    "Trigger the training pipeline with corrupt_data injection"   (validation fails)
    "Trigger the training pipeline with low_quality injection"    (quality gate fails)
"""

from __future__ import annotations

import json
import subprocess
import sys

NS, DEP = "mlops-agent", "inference-api"


def kubectl(*args: str) -> str:
    print("$ kubectl", " ".join(args))
    return subprocess.run(["kubectl", "-n", NS, *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "bad-image":
        kubectl("set", "image", f"deployment/{DEP}", f"{DEP}=mlops-inference:model-v404-does-not-exist")
    elif mode == "bad-config":
        kubectl(
            "patch",
            "configmap",
            "inference-config",
            "--type",
            "merge",
            "-p",
            json.dumps({"data": {"MODEL_PATH": "/app/model/model-MISSING.joblib"}}),
        )
        kubectl("rollout", "restart", f"deployment/{DEP}")
    elif mode == "restore":
        kubectl(
            "patch",
            "configmap",
            "inference-config",
            "--type",
            "merge",
            "-p",
            json.dumps({"data": {"MODEL_PATH": "/app/model/model.joblib"}}),
        )
        image = kubectl("get", f"deployment/{DEP}", "-o", "jsonpath={.spec.template.spec.containers[0].image}")
        if "does-not-exist" in image:
            kubectl("rollout", "undo", f"deployment/{DEP}")
        else:
            kubectl("rollout", "restart", f"deployment/{DEP}")
        print(kubectl("rollout", "status", f"deployment/{DEP}", "--timeout=120s"))
    else:
        sys.exit(__doc__)
    print(f'done: {mode}. Now ask the agent: "Why is my prediction API unavailable?"')


if __name__ == "__main__":
    main()
