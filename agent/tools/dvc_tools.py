"""DVC data-management tools.

All CLI access goes through ``DvcRunner`` which enforces the sub-command allowlist from
``config/agent_policy.yaml`` (no push/gc/remove/checkout of the working tree) and never uses a shell.
Dataset versions are named by git tags ``data-<version>`` on the commits that changed the ``.dvc`` file.
"""

from __future__ import annotations

import configparser
import hashlib
import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent.safety import PolicyError, ServiceUnavailable, ToolInputError, check_command
from agent.tools.base import ToolContext, ToolOutput, ToolRegistry

DEFAULT_DATASET = "data/raw/credit_risk.csv"
_REV_PATTERN = r"^[A-Za-z0-9._/\-~^]{1,100}$"


class DvcRunner:
    def __init__(self, repo: Path, policy: dict[str, Any], timeout: float = 60.0) -> None:
        self.repo = Path(repo)
        self.policy = policy
        self.timeout = timeout
        if not (self.repo / ".dvc").is_dir():
            raise ServiceUnavailable(f"{self.repo} is not a DVC repository (no .dvc directory)")

    def _run(self, exe: list[str], args: list[str]) -> str:
        try:
            proc = subprocess.run(
                [*exe, *args],
                cwd=self.repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailable(f"executable not found: {exe[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"{' '.join(exe[-1:] + args[:2])} timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(msg[-3:])[:600]}")
        return proc.stdout

    def dvc(self, *args: str) -> str:
        check_command("dvc", list(args), self.policy)
        return self._run([sys.executable, "-m", "dvc"], list(args))

    def git(self, *args: str) -> str:
        check_command("git", list(args), self.policy)
        if shutil.which("git") is None:
            raise ServiceUnavailable("git executable not found")
        return self._run(["git"], list(args))


def _runner(ctx: ToolContext) -> DvcRunner:
    return ctx.client("dvc")


def dvc_factory(ctx: ToolContext) -> DvcRunner:
    return DvcRunner(ctx.settings.repo_root, ctx.policy, timeout=max(ctx.settings.tool_timeout_s - 2, 5))


# ----------------------------------------------------------------------------- helpers


def _md5_from_dvc_text(text: str) -> dict[str, Any]:
    meta = yaml.safe_load(text) or {}
    out = (meta.get("outs") or [{}])[0]
    return {"md5": out.get("md5"), "size": out.get("size"), "path": out.get("path")}


def _tag_map(r: DvcRunner) -> dict[str, list[str]]:
    """commit sha -> [tags] (handles annotated + lightweight tags)."""
    raw = r.git("tag", "--list", "--format=%(refname:short) %(objectname) %(*objectname)")
    mapping: dict[str, list[str]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        tag, obj = parts[0], parts[-1] if len(parts) == 3 else parts[1]
        mapping.setdefault(obj, []).append(tag)
    return mapping


def _version_from_tags(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("data-"):
            return t[len("data-") :]
    return None


def _history(r: DvcRunner, path: str, limit: int = 20) -> list[dict[str, Any]]:
    dvc_file = f"{path}.dvc"
    log = r.git("log", f"-n{limit}", "--format=%H %h %cI %s", "--", dvc_file)
    tags = _tag_map(r)
    entries = []
    for line in log.splitlines():
        sha, short, date, *msg = line.split(" ")
        try:
            meta = _md5_from_dvc_text(r.git("show", f"{sha}:{dvc_file}"))
        except RuntimeError:  # file deleted in that commit
            meta = {"md5": None, "size": None}
        entries.append(
            {
                "commit": sha,
                "short_commit": short,
                "date": date,
                "message": " ".join(msg),
                "md5": meta["md5"],
                "size_bytes": meta["size"],
                "tags": tags.get(sha, []),
                "version": _version_from_tags(tags.get(sha, [])),
            }
        )
    return entries


def _resolve_rev(r: DvcRunner, rev: str) -> str:
    """Accept a commit, a tag, or a bare version name like 'v1.0' (-> tag data-v1.0)."""
    candidates = [rev] if rev.startswith("data-") else [f"data-{rev}", rev]
    for cand in candidates:
        try:
            return r.git("rev-parse", "--verify", f"{cand}^{{commit}}").strip()
        except RuntimeError:
            continue
    raise ToolInputError(f"Unknown revision or dataset version '{rev}'.")


def _remote_info(repo: Path) -> dict[str, Any]:
    cfg_path = repo / ".dvc" / "config"
    if not cfg_path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(cfg_path, encoding="utf-8")
    default = parser.get("core", "remote", fallback=None)
    remotes = {}
    for section in parser.sections():
        if section.startswith("'remote"):
            name = section.split('"')[1]
            remotes[name] = parser.get(section, "url", fallback=None)
    return {"default_remote": default, "remotes": remotes}


def _load_at_rev(repo: Path, path: str, rev: str):
    import dvc.api
    import pandas as pd

    with dvc.api.open(path, repo=str(repo), rev=rev, mode="r", encoding="utf-8") as fh:
        return pd.read_csv(io.StringIO(fh.read()))


def _psi(expected, actual, bins: int = 10) -> float:
    import numpy as np

    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(exp, edges)[0] / len(exp)
    a = np.histogram(np.clip(act, edges[0], edges[-1]), edges)[0] / len(act)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ----------------------------------------------------------------------------- schemas


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Base):
    pass


class DatasetArg(_Base):
    path: str = Field(DEFAULT_DATASET, description="Repo-relative path of the DVC-tracked dataset.")


class HistoryArgs(DatasetArg):
    limit: int = Field(20, ge=1, le=100)


class CompareArgs(_Base):
    rev_a: str = Field(
        ..., pattern=_REV_PATTERN, description="Older version: tag (data-v1.0), version (v1.0) or commit."
    )
    rev_b: str = Field("HEAD", pattern=_REV_PATTERN, description="Newer version (default HEAD).")
    path: str = DEFAULT_DATASET


class RunArg(_Base):
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$", description="MLflow run id (32 hex chars).")


class ReproduceArgs(_Base):
    version: str = Field(..., pattern=_REV_PATTERN, description="Dataset version (v1.0), tag or commit to reproduce.")
    path: str = DEFAULT_DATASET
    verify: bool = Field(True, description="Materialise into a scratch dir and verify the md5 checksum.")


def _check_path(ctx: ToolContext, path: str) -> str:
    if ".." in Path(path).parts or Path(path).is_absolute():
        raise PolicyError("Dataset path must be repo-relative without '..'.")
    if not (ctx.settings.repo_root / f"{path}.dvc").exists() and not (ctx.settings.repo_root / "dvc.lock").exists():
        raise ToolInputError(f"'{path}' is not tracked by DVC (no {path}.dvc).")
    return path


# ----------------------------------------------------------------------------- tools


def register(reg: ToolRegistry) -> None:
    @reg.tool(
        "dvc_list_tracked_datasets",
        "List every dataset / artefact tracked by DVC (.dvc pointer files and dvc.lock pipeline outputs) "
        "with md5, size and the remote storage configuration.",
        NoArgs,
        system="dvc",
    )
    def list_tracked(args: NoArgs, ctx: ToolContext) -> ToolOutput:
        repo = ctx.settings.repo_root
        if not (repo / ".dvc").is_dir():
            raise ServiceUnavailable(f"{repo} is not a DVC repository")
        items = []
        for f in sorted(repo.rglob("*.dvc")):
            if any(part in {".dvc", ".venv", "node_modules", ".git"} for part in f.relative_to(repo).parts):
                continue
            meta = _md5_from_dvc_text(f.read_text(encoding="utf-8"))
            rel = f.relative_to(repo).as_posix()[: -len(".dvc")]
            items.append({"path": rel, "kind": "dvc-file", "md5": meta["md5"], "size_bytes": meta["size"]})
        lock = repo / "dvc.lock"
        if lock.exists():
            data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
            for stage, body in (data.get("stages") or {}).items():
                for out in body.get("outs", []) or []:
                    items.append(
                        {
                            "path": out.get("path"),
                            "kind": f"pipeline-output ({stage})",
                            "md5": out.get("md5"),
                            "size_bytes": out.get("size"),
                        }
                    )
        remote = _remote_info(repo)
        summary = f"{len(items)} DVC-tracked item(s): " + ", ".join(i["path"] for i in items[:8])
        if remote.get("default_remote"):
            summary += f"; default remote '{remote['default_remote']}'"
        return ToolOutput(summary, {"datasets": items, **remote})

    @reg.tool(
        "dvc_get_dataset_version",
        "Get the CURRENT version of a dataset: tracked md5, git commit, the dataset version tag it maps "
        "to, and whether the workspace copy matches what DVC tracks.",
        DatasetArg,
        system="dvc",
    )
    def get_version(args: DatasetArg, ctx: ToolContext) -> ToolOutput:
        path = _check_path(ctx, args.path)
        r = _runner(ctx)
        repo = ctx.settings.repo_root
        tracked = _md5_from_dvc_text((repo / f"{path}.dvc").read_text(encoding="utf-8"))
        head = r.git("rev-parse", "HEAD").strip()
        hist = _history(r, path, limit=50)
        version, version_commit = None, None
        for h in hist:  # newest first: the first commit with the same md5 that carries a version tag
            if h["md5"] == tracked["md5"] and h["version"]:
                version, version_commit = h["version"], h["commit"]
                break
        workspace_md5 = None
        file_path = repo / path
        if file_path.exists():
            md5 = hashlib.md5()  # noqa: S324 - DVC content hash
            with file_path.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    md5.update(block)
            workspace_md5 = md5.hexdigest()
        params = (
            yaml.safe_load((repo / "params.yaml").read_text(encoding="utf-8"))
            if (repo / "params.yaml").exists()
            else {}
        )
        data = {
            "path": path,
            "version": version,
            "version_commit": version_commit,
            "tracked_md5": tracked["md5"],
            "size_bytes": tracked["size"],
            "head_commit": head,
            "workspace_md5": workspace_md5,
            "workspace_matches_tracked": workspace_md5 == tracked["md5"] if workspace_md5 else None,
            "declared_version_params_yaml": (params.get("dataset") or {}).get("version"),
            **_remote_info(repo),
        }
        state = (
            "workspace in sync"
            if data["workspace_matches_tracked"]
            else ("workspace file MISSING" if workspace_md5 is None else "workspace MODIFIED vs tracked")
        )
        summary = (
            f"{path} is at dataset version {version or 'UNTAGGED'} (md5 {tracked['md5']}), "
            f"git HEAD {head[:7]}; {state}."
        )
        return ToolOutput(summary, data)

    @reg.tool(
        "dvc_list_dataset_versions",
        "List the version history of a DVC-tracked dataset: every git commit that changed it, its md5, "
        "date and version tag.",
        HistoryArgs,
        system="dvc",
    )
    def list_versions(args: HistoryArgs, ctx: ToolContext) -> ToolOutput:
        path = _check_path(ctx, args.path)
        hist = _history(_runner(ctx), path, args.limit)
        tagged = [f"{h['version']}@{h['short_commit']}" for h in hist if h["version"]]
        return ToolOutput(
            f"{len(hist)} change(s) to {path}; tagged versions: {', '.join(tagged) or 'none'}.",
            {"path": path, "history": hist},
        )

    @reg.tool(
        "dvc_compare_versions",
        "Compare two versions of a dataset (tags like data-v1.0, version names like v1.0, or commits): "
        "md5 change, row counts, class balance, schema changes and per-feature drift (PSI).",
        CompareArgs,
        system="dvc",
        timeout_s=60,
    )
    def compare(args: CompareArgs, ctx: ToolContext) -> ToolOutput:
        path = _check_path(ctx, args.path)
        r = _runner(ctx)
        repo = ctx.settings.repo_root
        sha_a, sha_b = _resolve_rev(r, args.rev_a), _resolve_rev(r, args.rev_b)
        meta_a = _md5_from_dvc_text(r.git("show", f"{sha_a}:{path}.dvc"))
        meta_b = _md5_from_dvc_text(r.git("show", f"{sha_b}:{path}.dvc"))
        data: dict[str, Any] = {
            "path": path,
            "a": {"rev": args.rev_a, "commit": sha_a, **meta_a},
            "b": {"rev": args.rev_b, "commit": sha_b, **meta_b},
            "content_changed": meta_a["md5"] != meta_b["md5"],
        }
        if not data["content_changed"]:
            return ToolOutput(f"{path}: {args.rev_a} and {args.rev_b} are byte-identical (md5 {meta_a['md5']}).", data)
        try:
            df_a, df_b = _load_at_rev(repo, path, sha_a), _load_at_rev(repo, path, sha_b)
        except Exception as exc:
            data["content_comparison_error"] = f"could not load both versions from DVC cache/remote: {exc}"
            return ToolOutput(f"{path} changed between {args.rev_a} and {args.rev_b} (md5 differs).", data)
        from pandas.api.types import is_numeric_dtype

        cols_a, cols_b = set(df_a.columns), set(df_b.columns)
        drift = {}
        for col in sorted(cols_a & cols_b):
            if is_numeric_dtype(df_a[col]) and is_numeric_dtype(df_b[col]) and col != "default":
                drift[col] = round(_psi(df_a[col].dropna(), df_b[col].dropna()), 4)
        target = "default" if "default" in cols_a & cols_b else None
        data.update(
            {
                "rows": {"a": len(df_a), "b": len(df_b), "delta": len(df_b) - len(df_a)},
                "columns_added": sorted(cols_b - cols_a),
                "columns_removed": sorted(cols_a - cols_b),
                "target_rate": (
                    {"a": round(float(df_a[target].mean()), 4), "b": round(float(df_b[target].mean()), 4)}
                    if target
                    else None
                ),
                "psi_by_feature": drift,
                "drifted_features": [c for c, v in drift.items() if v >= 0.1],
            }
        )
        summary = (
            f"{path}: {args.rev_a} ({len(df_a)} rows) -> {args.rev_b} ({len(df_b)} rows), "
            f"{len(df_b) - len(df_a):+d} rows"
        )
        if target:
            summary += f"; default rate {data['target_rate']['a']:.1%} -> {data['target_rate']['b']:.1%}"
        summary += f"; features with PSI>=0.1: {data['drifted_features'] or 'none'}."
        return ToolOutput(summary, data)

    @reg.tool(
        "dvc_check_data_status",
        "Detect whether DVC-tracked data or pipeline stages have changed (`dvc status`): modified "
        "workspace data, stale pipeline outputs, missing cache.",
        NoArgs,
        system="dvc",
        timeout_s=60,
    )
    def status(args: NoArgs, ctx: ToolContext) -> ToolOutput:
        import json

        raw = _runner(ctx).dvc("status", "--json")
        parsed = json.loads(raw or "{}")
        changes = []
        for target, entries in parsed.items():
            for entry in entries if isinstance(entries, list) else [entries]:
                if isinstance(entry, dict):
                    for kind, val in entry.items():
                        changes.append({"target": target, "kind": kind, "detail": val})
                else:
                    changes.append({"target": target, "kind": "status", "detail": entry})
        if not changes:
            return ToolOutput("DVC status clean: tracked data and pipeline outputs are up to date.", {"changes": []})
        summary = f"{len(parsed)} DVC target(s) changed: " + ", ".join(sorted(parsed)[:6])
        return ToolOutput(summary, {"changes": changes, "raw": parsed})

    @reg.tool(
        "dvc_resolve_run_dataset",
        "Provenance: given an MLflow run id, find the exact dataset version (DVC md5 -> git commit -> "
        "version tag) that the run was trained on, from tags recorded at training time.",
        RunArg,
        system="dvc",
    )
    def resolve_run(args: RunArg, ctx: ToolContext) -> ToolOutput:
        client = ctx.client("mlflow")
        run = client.get_run(args.run_id)
        tags = run.data.tags
        md5 = tags.get("dataset.md5")
        path = tags.get("dataset.path", DEFAULT_DATASET)
        if not md5:
            raise ToolInputError(f"Run {args.run_id} has no recorded dataset.md5 tag; provenance unknown.")
        r = _runner(ctx)
        matches = [h for h in _history(r, path, limit=100) if h["md5"] == md5]
        version = next((h["version"] for h in matches if h["version"]), None)
        data = {
            "run_id": args.run_id,
            "run_name": run.info.run_name,
            "dataset_path": path,
            "dataset_md5": md5,
            "recorded_version_tag": tags.get("dataset.version"),
            "recorded_git_commit": tags.get("git.commit"),
            "workspace_was_in_sync": tags.get("dataset.in_sync_with_dvc"),
            "resolved_version": version,
            "commits_with_this_data": [{"commit": h["commit"], "date": h["date"], "tags": h["tags"]} for h in matches],
        }
        if not matches:
            summary = (
                f"Run {args.run_id[:8]} used {path} md5 {md5}, which does NOT match any committed DVC version "
                f"(data was not committed when the run happened). Recorded commit: {tags.get('git.commit', '?')[:7]}."
            )
        else:
            summary = (
                f"Run {args.run_id[:8]} ({run.info.run_name}) trained on {path} version "
                f"{version or 'untagged'} (md5 {md5}); recorded git commit {tags.get('git.commit', '?')[:7]}."
            )
            rec = tags.get("dataset.version")
            if rec and version and rec != version:
                summary += f" WARNING: params.yaml declared '{rec}' but the bytes match '{version}'."
        return ToolOutput(summary, data)

    @reg.tool(
        "dvc_reproduce_dataset",
        "Explain AND verify how to reproduce a previous dataset version: materialises that version from "
        "the DVC cache/remote into a scratch directory (never touches the working tree), verifies its md5, "
        "and returns the exact commands an engineer should run.",
        ReproduceArgs,
        system="dvc",
        timeout_s=90,
    )
    def reproduce(args: ReproduceArgs, ctx: ToolContext) -> ToolOutput:
        path = _check_path(ctx, args.path)
        r = _runner(ctx)
        sha = _resolve_rev(r, args.version)
        meta = _md5_from_dvc_text(r.git("show", f"{sha}:{path}.dvc"))
        commands = [
            f"git checkout {sha[:7]} -- {path}.dvc",
            f"dvc checkout {path}",
            "# or, without touching the workspace:",
            f"dvc get . {path} --rev {sha[:7]} -o <output.csv>",
            "# back to the latest version:",
            f"git checkout HEAD -- {path}.dvc && dvc checkout {path}",
        ]
        data: dict[str, Any] = {
            "version": args.version,
            "commit": sha,
            "expected_md5": meta["md5"],
            "commands": commands,
        }
        if args.verify:
            out_dir = ctx.settings.repo_root / "artifacts" / "repro" / sha[:7]
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / Path(path).name
            if target.exists():
                target.unlink()
            r.dvc("get", ".", path, "--rev", sha, "-o", str(target))
            md5 = hashlib.md5(target.read_bytes()).hexdigest()  # noqa: S324
            data.update(
                {
                    "materialised_to": str(target.relative_to(ctx.settings.repo_root)),
                    "actual_md5": md5,
                    "verified": md5 == meta["md5"],
                }
            )
        verdict = ""
        if args.verify:
            verdict = " Verified: checksum matches." if data.get("verified") else " CHECKSUM MISMATCH!"
        return ToolOutput(
            f"Dataset {args.version} = commit {sha[:7]}, md5 {meta['md5']}.{verdict} "
            f"Reproduce with: git checkout {sha[:7]} -- {path}.dvc && dvc checkout {path}",
            data,
        )
