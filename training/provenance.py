"""Data/code provenance helpers that work with or without the git binary.

Inside the Airflow container the repository is bind-mounted but ``git`` may not be installed, so
HEAD is resolved by reading ``.git`` directly when the CLI is unavailable.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - md5 matches DVC's content addressing, not used for security
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_dvc_file(dvc_file: str | Path) -> dict[str, Any]:
    """Return the first ``outs`` entry of a ``.dvc`` file (md5, size, path)."""
    with open(dvc_file, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    outs = meta.get("outs") or [{}]
    return outs[0]


def _git(repo: Path, *args: str) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _read_head_without_git(repo: Path) -> tuple[str | None, str | None]:
    git_dir = repo / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return None, None
    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head, None
    ref = head.split(" ", 1)[1].strip()
    branch = ref.rsplit("/", 1)[-1]
    ref_file = git_dir / ref
    if ref_file.exists():
        return ref_file.read_text(encoding="utf-8").strip(), branch
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0], branch
    return None, branch


def git_info(repo: str | Path) -> dict[str, Any]:
    repo = Path(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    tag = _git(repo, "describe", "--tags", "--always")
    dirty = _git(repo, "status", "--porcelain")
    if commit is None:
        commit, branch = _read_head_without_git(repo)
    return {
        "commit": commit,
        "short_commit": commit[:7] if commit else None,
        "branch": branch,
        "describe": tag,
        "dirty": bool(dirty) if dirty is not None else None,
    }


def dataset_provenance(repo: str | Path, data_path: str, declared_version: str | None = None) -> dict[str, Any]:
    """Compare the workspace file against the md5 recorded in its .dvc pointer."""
    repo = Path(repo)
    file_path = repo / data_path
    dvc_file = repo / f"{data_path}.dvc"
    tracked = read_dvc_file(dvc_file) if dvc_file.exists() else {}
    actual = file_md5(file_path) if file_path.exists() else None
    return {
        "path": data_path,
        "declared_version": declared_version,
        "tracked_md5": tracked.get("md5"),
        "workspace_md5": actual,
        "size_bytes": tracked.get("size"),
        "in_sync_with_dvc": bool(actual and tracked.get("md5") == actual),
        **{f"git_{k}": v for k, v in git_info(repo).items()},
    }
