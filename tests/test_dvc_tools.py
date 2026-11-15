"""DVC tools against a REAL temporary git + DVC repository with two tagged dataset versions."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent.tools.dvc_tools import DvcRunner
from tests.fakes import FakeMlflow, fake_run


@pytest.fixture
def ex(make_executor, settings, dvc_repo, policy):
    s = replace(settings, repo_root=dvc_repo, tool_timeout_s=60)
    return make_executor({"dvc": DvcRunner(dvc_repo, policy, timeout=50)}, settings_override=s)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_list_tracked_datasets(ex):
    r = ex.execute("dvc_list_tracked_datasets", {})
    assert r.ok and r.data["datasets"][0]["path"] == "data/raw/credit_risk.csv"
    assert r.data["default_remote"] == "local"


def test_current_version_and_commit(ex, dvc_repo):
    r = ex.execute("dvc_get_dataset_version", {})
    assert r.ok and r.data["version"] == "v2.0"
    assert r.data["workspace_matches_tracked"] is True
    assert len(r.data["head_commit"]) == 40 and r.data["head_commit"][:7] in r.summary


def test_version_history_lists_both_tags(ex):
    r = ex.execute("dvc_list_dataset_versions", {})
    assert [h["version"] for h in r.data["history"]] == ["v2.0", "v1.0"]
    assert r.data["history"][0]["md5"] != r.data["history"][1]["md5"]


def test_compare_versions_reports_rows_and_drift(ex):
    r = ex.execute("dvc_compare_versions", {"rev_a": "v1.0", "rev_b": "v2.0"})
    assert r.ok and r.data["content_changed"] is True
    assert r.data["rows"] == {"a": 200, "b": 300, "delta": 100}
    assert "age" in r.data["psi_by_feature"] and r.data["columns_added"] == []


def test_compare_identical_versions(ex):
    r = ex.execute("dvc_compare_versions", {"rev_a": "data-v2.0", "rev_b": "HEAD"})
    assert r.data["content_changed"] is False and "byte-identical" in r.summary


def test_detects_modified_tracked_data(ex, dvc_repo):
    target = dvc_repo / "data" / "raw" / "credit_risk.csv"
    original = target.read_bytes()
    clean = ex.execute("dvc_check_data_status", {})
    assert clean.ok and clean.data["changes"] == []
    try:
        target.write_bytes(
            original
            + b"999,1,1,1,30,1,1,<0,all paid,business,<100,<1,male single,none,car,none,own,skilled,none,yes,1\n"
        )
        dirty = ex.execute("dvc_check_data_status", {})
        assert dirty.ok and dirty.data["changes"], dirty.summary
        version = ex.execute("dvc_get_dataset_version", {})
        assert version.data["workspace_matches_tracked"] is False and "MODIFIED" in version.summary
    finally:
        target.write_bytes(original)


def test_reproduce_previous_version_verifies_checksum(ex, dvc_repo):
    r = ex.execute("dvc_reproduce_dataset", {"version": "v1.0"})
    assert r.ok and r.data["verified"] is True
    materialised = dvc_repo / r.data["materialised_to"]
    assert _md5(materialised) == r.data["expected_md5"]
    assert any("dvc checkout" in c for c in r.data["commands"])
    # the working tree is untouched: still v2.0
    assert ex.execute("dvc_get_dataset_version", {}).data["version"] == "v2.0"


def test_resolve_run_dataset_from_mlflow_tags(make_executor, settings, dvc_repo, policy):
    v1 = make_executor({"dvc": DvcRunner(dvc_repo, policy)}, settings_override=replace(settings, repo_root=dvc_repo))
    md5_v1 = v1.execute("dvc_list_dataset_versions", {}).data["history"][1]["md5"]
    run_id = "f" * 32
    mlflow = FakeMlflow(runs={run_id: fake_run(run_id, md5=md5_v1, version="v1.0")})
    ex = make_executor(
        {"dvc": DvcRunner(dvc_repo, policy), "mlflow": mlflow},
        settings_override=replace(settings, repo_root=dvc_repo, tool_timeout_s=30),
    )
    r = ex.execute("dvc_resolve_run_dataset", {"run_id": run_id})
    assert r.ok and r.data["resolved_version"] == "v1.0" and r.data["commits_with_this_data"]


def test_run_with_uncommitted_data_is_flagged(make_executor, settings, dvc_repo, policy):
    run_id = "e" * 32
    mlflow = FakeMlflow(runs={run_id: fake_run(run_id, md5="0" * 32)})
    ex = make_executor(
        {"dvc": DvcRunner(dvc_repo, policy), "mlflow": mlflow},
        settings_override=replace(settings, repo_root=dvc_repo, tool_timeout_s=30),
    )
    r = ex.execute("dvc_resolve_run_dataset", {"run_id": run_id})
    assert r.ok and r.data["resolved_version"] is None and "does NOT match" in r.summary


def test_unknown_version_is_invalid_request(ex):
    assert ex.execute("dvc_reproduce_dataset", {"version": "v9.9"}).status == "invalid_request"


@pytest.mark.parametrize("path", ["../../etc/passwd", "C:/Windows/system.ini"])
def test_path_traversal_denied(ex, path):
    assert ex.execute("dvc_get_dataset_version", {"path": path}).status == "denied"


def test_runner_refuses_dangerous_subcommands(dvc_repo, policy):
    from agent.safety import PolicyError

    runner = DvcRunner(dvc_repo, policy)
    for args in (["push"], ["gc", "-w", "-f"], ["remove", "data/raw/credit_risk.csv.dvc"]):
        with pytest.raises(PolicyError):
            runner.dvc(*args)


def test_non_dvc_directory_unavailable(make_executor, settings, tmp_path):
    ex = make_executor(settings_override=replace(settings, repo_root=tmp_path))
    assert ex.execute("dvc_list_tracked_datasets", {}).status == "unavailable"
