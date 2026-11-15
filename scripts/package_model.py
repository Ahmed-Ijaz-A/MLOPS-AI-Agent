"""Rebuild the release model from the DVC-tracked dataset and package it for the inference image (used by CD).

    python scripts/package_model.py --version 1.2.0

1. materialise dataset v<params.dataset.version> from its source (OpenML credit-g)
2. VERIFY its md5 equals the md5 recorded in data/raw/credit_risk.csv.dvc  -> abort on mismatch
   (the release is provably trained on the exact bytes DVC tracks at this commit)
3. run the pipeline (validate -> preprocess -> train -> evaluate -> quality gate); abort if the gate fails
4. write serving/model/model.joblib + metadata.json (version, dataset md5, git commit, metrics)

Locally, releases normally come from the MLflow registry (scripts/deploy_model.py). CI runners cannot reach the
local MLflow server, so CD rebuilds the identical model deterministically and proves it via the DVC checksum.
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


def main() -> int:
    from training.data import make_dataset
    from training.provenance import file_md5, read_dvc_file
    from training.train import load_params, run_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("RELEASE_VERSION", "dev"), help="release / model version label")
    ap.add_argument("--skip-materialise", action="store_true", help="use the dataset already in the workspace")
    a = ap.parse_args()

    params = load_params()
    ds = params["dataset"]
    data_path = ROOT / ds["path"]
    expected = read_dvc_file(ROOT / f"{ds['path']}.dvc")["md5"]
    if not a.skip_materialise:
        make_dataset(ds["version"], data_path)
    actual = file_md5(data_path)
    print(f"dataset {ds['version']}: expected md5 {expected} (DVC), got {actual}")
    if actual != expected:
        print("ERROR: rebuilt dataset does not match the DVC-tracked version - refusing to release", file=sys.stderr)
        return 3

    result = run_all(params, log_to_mlflow=False, register=False)
    if not result["quality_gate"]:
        print(f"ERROR: quality gate failed: {result['reasons']} - refusing to release", file=sys.stderr)
        return 2

    out = ROOT / "serving" / "model"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(ROOT / "models" / "model.joblib", out / "model.joblib")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = os.environ.get("GITHUB_SHA", "")
    meta = {
        "model_name": "credit-risk-classifier",
        "model_version": a.version,
        "run_id": None,
        "source": "cd-rebuild (deterministic retrain from DVC-verified data)",
        "model_type": params["train"]["model_type"],
        "dataset_version": ds["version"],
        "dataset_md5": actual,
        "git_commit": commit or os.environ.get("GITHUB_SHA"),
        "metrics": {k: result["metrics"][k] for k in ("f1", "recall", "roc_auc", "accuracy")},
        "quality_gate": "passed",
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:  # shows up on the workflow run page
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Release model {a.version}\n\n| field | value |\n|---|---|\n")
            for k in ("dataset_version", "dataset_md5", "git_commit", "model_type"):
                fh.write(f"| {k} | `{meta[k]}` |\n")
            for k, v in meta["metrics"].items():
                fh.write(f"| {k} | {v} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
