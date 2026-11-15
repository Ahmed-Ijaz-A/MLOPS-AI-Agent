"""Create the DVC-tracked dataset history (run once on a fresh clone of the code).

    python scripts/bootstrap_data_versions.py

1. ``dvc init`` + a local DVC remote (``.dvc-storage/``, git-ignored; use S3/GCS in production)
2. v1.0 = first 700 applications  -> dvc add -> git commit -> tag data-v1.0 -> dvc push
3. v2.0 = all 1000 applications   -> dvc add -> git commit -> tag data-v2.0 -> dvc push

Only DVC metadata (.dvc pointer, .gitignore, params.yaml, .dvc/config) is committed - never the CSV.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = "data/raw/credit_risk.csv"


def run(*cmd: str, check: bool = True) -> str:
    print("$", " ".join(cmd))
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and out.returncode != 0:
        sys.exit(f"failed: {out.stderr or out.stdout}")
    return out.stdout.strip()


def dvc(*args: str, check: bool = True) -> str:
    return run(sys.executable, "-m", "dvc", *args, check=check)


def set_params_version(version: str) -> None:
    p = ROOT / "params.yaml"
    p.write_text(
        re.sub(r"(?m)^(  version: ).*$", rf"\g<1>{version}", p.read_text(encoding="utf-8"), count=1), encoding="utf-8"
    )


def make_version(version: str, message: str) -> None:
    from training.data import make_dataset

    make_dataset(version, ROOT / DATA)
    set_params_version(version)
    dvc("add", DATA)
    run("git", "add", f"{DATA}.dvc", "data/raw/.gitignore", "params.yaml")
    run("git", "commit", "-m", message)
    run("git", "tag", "-a", f"data-{version}", "-m", f"Dataset {version}: {message}")
    dvc("push")


def main() -> None:
    if run("git", "status", "--porcelain", "--untracked-files=no"):
        sys.exit("Commit or stash your code changes first - this script makes data-only commits.")
    if run("git", "tag", "--list", "data-v*"):
        sys.exit("Dataset tags already exist; nothing to do.")
    if not (ROOT / ".dvc").exists():
        dvc("init")
        dvc("remote", "add", "-d", "localstore", "../.dvc-storage")
        dvc("config", "core.autostage", "false")
        run("git", "add", ".dvc/config", ".dvc/.gitignore", ".dvcignore")
        run("git", "commit", "-m", "chore(dvc): initialise DVC with local remote")
    make_version("v1.0", "data: credit-risk v1.0 - initial extract (700 applications)")
    make_version("v2.0", "data: credit-risk v2.0 - append new batch (1000 applications)")
    print("\nDone. Dataset history:")
    print(run("git", "log", "--oneline", "--decorate", "-n", "4"))


if __name__ == "__main__":
    main()
