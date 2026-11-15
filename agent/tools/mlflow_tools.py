"""MLflow experiment-intelligence tools (MlflowClient only - no UI scraping)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.safety import ToolInputError
from agent.tools.base import ToolContext, ToolOutput, ToolRegistry

KEY_METRICS = ["f1", "roc_auc", "recall", "precision", "accuracy"]


def mlflow_factory(ctx: ToolContext):
    # keep client-side retries short so an unavailable server is reported quickly
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")
    mlflow.set_tracking_uri(ctx.settings.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=ctx.settings.mlflow_tracking_uri)


def _client(ctx: ToolContext):
    return ctx.client("mlflow")


def _ts(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds") if ms else None


def _age_days(ms: int | None) -> float | None:
    if not ms:
        return None
    return round((datetime.now(UTC).timestamp() * 1000 - ms) / 86_400_000, 2)


def _run_row(run: Any, metrics: list[str] | None = None) -> dict[str, Any]:
    tags = run.data.tags
    m = run.data.metrics
    keep = metrics or KEY_METRICS
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": _ts(run.info.start_time),
        "age_days": _age_days(run.info.start_time),
        "metrics": {k: m[k] for k in keep if k in m},
        "model_type": tags.get("model_type") or run.data.params.get("model_type"),
        "dataset_version": tags.get("dataset.version"),
        "dataset_md5": tags.get("dataset.md5"),
        "git_commit": tags.get("git.commit"),
        "quality_gate": tags.get("quality_gate.status", "unknown"),
        "orchestrator": tags.get("pipeline.orchestrator"),
        "failure_injection": tags.get("failure_injection"),
    }


def _experiment(ctx: ToolContext, name: str | None):
    client = _client(ctx)
    name = name or ctx.policy.get("mlflow", {}).get("experiment_name")
    exp = client.get_experiment_by_name(name)
    if exp is None:
        names = [e.name for e in client.search_experiments()]
        raise ToolInputError(f"Experiment '{name}' not found. Existing experiments: {names}")
    return exp


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Base):
    pass


class ListRunsArgs(_Base):
    experiment: str | None = Field(None, description="Experiment name (default: the project experiment).")
    max_results: int = Field(10, ge=1, le=100)
    status: Literal["FINISHED", "FAILED", "RUNNING", "ANY"] = "FINISHED"


class RunArgs(_Base):
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")


class BestRunArgs(_Base):
    metric: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]{1,60}$", description="Metric to rank by.")
    mode: Literal["max", "min"] = "max"
    experiment: str | None = None
    last_n: int = Field(50, ge=1, le=500, description="Only consider the most recent N finished runs.")
    require_quality_gate: bool = Field(False, description="Only consider runs whose quality gate passed.")


class CompareRunsArgs(_Base):
    run_ids: list[str] | None = Field(None, description="Explicit run ids. If omitted, the last_n most recent runs.")
    last_n: int = Field(5, ge=2, le=25)
    metric: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]{1,60}$", description="Metric used to recommend a candidate.")
    experiment: str | None = None


class CompareExperimentsArgs(_Base):
    experiments: list[str] = Field(..., min_length=2, max_length=6)
    metric: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]{1,60}$")


class RegistryArgs(_Base):
    model_name: str | None = Field(None, description="Registered model name (default from policy).")


def _caveats(best: dict[str, Any], runner_up: dict[str, Any] | None, metric: str, all_rows: list[dict]) -> list[str]:
    out = []
    if best["quality_gate"] != "passed":
        out.append(f"Quality gate status is '{best['quality_gate']}' - not promotable until it passes.")
    if runner_up:
        margin = best["metrics"].get(metric, 0) - runner_up["metrics"].get(metric, 0)
        if abs(margin) < 0.01:
            out.append(
                f"Margin over runner-up {runner_up['run_id'][:8]} is only {margin:+.4f} {metric} - within noise "
                "for a 200-row test set; treat as a tie and consider recall / ROC AUC."
            )
    datasets = {r["dataset_md5"] for r in all_rows if r["dataset_md5"]}
    if len(datasets) > 1:
        out.append("Compared runs were trained on DIFFERENT dataset versions; metrics are not directly comparable.")
    if best.get("failure_injection") not in (None, "none"):
        out.append(f"Run was produced with failure injection '{best['failure_injection']}'.")
    if best.get("age_days") is not None and best["age_days"] > 30:
        out.append(f"Run is {best['age_days']:.0f} days old.")
    out.append(
        "Offline test metrics alone do not make a model production-ready: check it is registered, passed the "
        "gate, matches the current data version, and is healthy when deployed."
    )
    return out


def register(reg: ToolRegistry) -> None:
    @reg.tool(
        "mlflow_list_experiments",
        "List MLflow experiments with their ids, artifact location and number of runs.",
        NoArgs,
        system="mlflow",
    )
    def list_experiments(args: NoArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        rows = []
        for exp in client.search_experiments():
            runs = client.search_runs([exp.experiment_id], max_results=200)
            rows.append(
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "runs": len(runs),
                    "created": _ts(getattr(exp, "creation_time", None)),
                }
            )
        return ToolOutput(
            f"{len(rows)} experiment(s): " + ", ".join(f"{r['name']} ({r['runs']} runs)" for r in rows),
            {"experiments": rows},
        )

    @reg.tool(
        "mlflow_list_runs",
        "List the most recent training runs of an experiment with key metrics, dataset version and quality-gate status.",
        ListRunsArgs,
        system="mlflow",
    )
    def list_runs(args: ListRunsArgs, ctx: ToolContext) -> ToolOutput:
        exp = _experiment(ctx, args.experiment)
        filt = "" if args.status == "ANY" else f"attributes.status = '{args.status}'"
        runs = _client(ctx).search_runs(
            [exp.experiment_id],
            filter_string=filt,
            order_by=["attributes.start_time DESC"],
            max_results=args.max_results,
        )
        rows = [_run_row(r) for r in runs]
        lines = [
            f"{r['run_id'][:8]} {r['model_type']} f1={r['metrics'].get('f1', 'n/a')} gate={r['quality_gate']}"
            for r in rows[:5]
        ]
        return ToolOutput(
            f"{len(rows)} run(s) in '{exp.name}': " + "; ".join(lines), {"experiment": exp.name, "runs": rows}
        )

    @reg.tool(
        "mlflow_get_run",
        "Get full details of one MLflow run: all metrics, parameters, tags (incl. dataset provenance) and artifact list.",
        RunArgs,
        system="mlflow",
    )
    def get_run(args: RunArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        run = client.get_run(args.run_id)
        try:
            artifacts = [a.path for a in client.list_artifacts(args.run_id)]
        except Exception as exc:  # artifact store may be separately unavailable
            artifacts = [f"<unavailable: {type(exc).__name__}>"]
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.") or k == "mlflow.runName"}
        data = {
            **_run_row(run, metrics=list(run.data.metrics)),
            "params": run.data.params,
            "tags": tags,
            "artifacts": artifacts,
            "end_time": _ts(run.info.end_time),
        }
        m = run.data.metrics
        return ToolOutput(
            f"Run {args.run_id[:8]} ({run.info.run_name}, {run.info.status}): "
            + ", ".join(f"{k}={m[k]:.4f}" for k in KEY_METRICS if k in m)
            + f"; dataset {tags.get('dataset.version', '?')} md5 {tags.get('dataset.md5', '?')}; gate {tags.get('quality_gate.status', '?')}.",
            data,
        )

    @reg.tool(
        "mlflow_find_best_run",
        "Identify the best recent run by a metric, with the runner-up, the margin and explicit caveats about "
        "why the top score alone does not make it production-ready.",
        BestRunArgs,
        system="mlflow",
    )
    def best_run(args: BestRunArgs, ctx: ToolContext) -> ToolOutput:
        exp = _experiment(ctx, args.experiment)
        runs = _client(ctx).search_runs(
            [exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=args.last_n,
        )
        rows = [_run_row(r, metrics=sorted({args.metric, *KEY_METRICS})) for r in runs]
        rows = [r for r in rows if args.metric in r["metrics"]]
        if args.require_quality_gate:
            rows = [r for r in rows if r["quality_gate"] == "passed"]
        if not rows:
            raise ToolInputError(f"No finished runs with metric '{args.metric}' in '{exp.name}'.")
        rows.sort(key=lambda r: r["metrics"][args.metric], reverse=args.mode == "max")
        best, runner_up = rows[0], (rows[1] if len(rows) > 1 else None)
        data = {
            "metric": args.metric,
            "considered_runs": len(rows),
            "best": best,
            "runner_up": runner_up,
            "caveats": _caveats(best, runner_up, args.metric, rows),
        }
        summary = (
            f"Best of {len(rows)} run(s) by {args.metric}: {best['run_id'][:8]} ({best['model_type']}) "
            f"{args.metric}={best['metrics'][args.metric]:.4f}, gate={best['quality_gate']}, dataset {best['dataset_version']}"
        )
        if runner_up:
            summary += f"; runner-up {runner_up['run_id'][:8]} {args.metric}={runner_up['metrics'][args.metric]:.4f}"
        return ToolOutput(summary + ".", data)

    @reg.tool(
        "mlflow_compare_runs",
        "Compare several runs side by side (explicit ids or the last N) across all key metrics and "
        "recommend a candidate with a justification.",
        CompareRunsArgs,
        system="mlflow",
    )
    def compare_runs(args: CompareRunsArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        if args.run_ids:
            runs = [client.get_run(rid) for rid in args.run_ids]
        else:
            exp = _experiment(ctx, args.experiment)
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=args.last_n,
            )
        rows = [_run_row(r) for r in runs]
        if len(rows) < 2:
            raise ToolInputError("Need at least two runs to compare.")
        leaders = {}
        for m in KEY_METRICS:
            scored = [r for r in rows if m in r["metrics"]]
            if scored:
                top = max(scored, key=lambda r: r["metrics"][m])
                leaders[m] = {"run_id": top["run_id"], "value": top["metrics"][m]}
        eligible = [r for r in rows if r["quality_gate"] == "passed" and args.metric in r["metrics"]]
        pool = eligible or [r for r in rows if args.metric in r["metrics"]]
        pool.sort(key=lambda r: (r["metrics"].get(args.metric, 0), r["metrics"].get("recall", 0)), reverse=True)
        rec = pool[0] if pool else None
        justification = []
        if rec:
            justification.append(
                f"highest {args.metric} ({rec['metrics'][args.metric]:.4f}) among "
                f"{'gate-passing' if eligible else 'all (none passed the gate!)'} runs"
            )
            if len(pool) > 1:
                gap = rec["metrics"][args.metric] - pool[1]["metrics"][args.metric]
                justification.append(f"margin {gap:+.4f} over {pool[1]['run_id'][:8]}")
            for m in ("recall", "roc_auc"):
                if m in leaders and leaders[m]["run_id"] != rec["run_id"]:
                    justification.append(f"note: {leaders[m]['run_id'][:8]} leads on {m} ({leaders[m]['value']:.4f})")
        data = {
            "runs": rows,
            "leaders_by_metric": leaders,
            "recommended": rec["run_id"] if rec else None,
            "justification": justification,
            "caveats": _caveats(rec, pool[1] if len(pool) > 1 else None, args.metric, rows) if rec else [],
        }
        table = "; ".join(
            f"{r['run_id'][:8]} {r['model_type']} f1={r['metrics'].get('f1', float('nan')):.4f} "
            f"recall={r['metrics'].get('recall', float('nan')):.4f} gate={r['quality_gate']}"
            for r in rows
        )
        return ToolOutput(
            f"Compared {len(rows)} runs: {table}. Recommended: {rec['run_id'][:8] if rec else 'none'} "
            f"({'; '.join(justification)}).",
            data,
        )

    @reg.tool(
        "mlflow_compare_experiments",
        "Compare two or more experiments: run counts, best / mean of a metric and quality-gate pass rate.",
        CompareExperimentsArgs,
        system="mlflow",
    )
    def compare_experiments(args: CompareExperimentsArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        out = []
        for name in args.experiments:
            exp = _experiment(ctx, name)
            runs = client.search_runs(
                [exp.experiment_id], filter_string="attributes.status = 'FINISHED'", max_results=500
            )
            vals = [r.data.metrics[args.metric] for r in runs if args.metric in r.data.metrics]
            passed = sum(1 for r in runs if r.data.tags.get("quality_gate.status") == "passed")
            out.append(
                {
                    "experiment": name,
                    "runs": len(runs),
                    f"best_{args.metric}": max(vals) if vals else None,
                    f"mean_{args.metric}": round(sum(vals) / len(vals), 4) if vals else None,
                    "gate_pass_rate": round(passed / len(runs), 3) if runs else None,
                }
            )
        return ToolOutput(
            "; ".join(
                f"{o['experiment']}: {o['runs']} runs, best {args.metric}={o[f'best_{args.metric}']}, "
                f"mean={o[f'mean_{args.metric}']}, gate pass rate={o['gate_pass_rate']}"
                for o in out
            ),
            {"metric": args.metric, "experiments": out},
        )

    @reg.tool(
        "mlflow_get_model_registry",
        "Get registered model versions, aliases (champion / candidate), their source runs, metrics and "
        "dataset versions from the MLflow Model Registry.",
        RegistryArgs,
        system="mlflow",
    )
    def registry(args: RegistryArgs, ctx: ToolContext) -> ToolOutput:
        client = _client(ctx)
        name = args.model_name or ctx.policy.get("mlflow", {}).get("registered_model_name")
        try:
            rm = client.get_registered_model(name)
        except Exception as exc:
            if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
                raise ToolInputError(f"Registered model '{name}' does not exist yet.") from exc
            raise
        aliases = {a: str(v) for a, v in (getattr(rm, "aliases", {}) or {}).items()}
        versions = client.search_model_versions(f"name='{name}'")
        rows = []
        for v in sorted(versions, key=lambda v: int(v.version), reverse=True)[:20]:
            rows.append(
                {
                    "version": str(v.version),
                    "aliases": [a for a, ver in aliases.items() if str(ver) == str(v.version)],
                    "run_id": v.run_id,
                    "created": _ts(v.creation_timestamp),
                    "age_days": _age_days(v.creation_timestamp),
                    "status": v.status,
                    "tags": dict(v.tags or {}),
                }
            )
        alias_txt = ", ".join(f"{a}=v{ver}" for a, ver in aliases.items()) or "no aliases"
        return ToolOutput(
            f"Model '{name}': {len(versions)} version(s); {alias_txt}.",
            {"model_name": name, "aliases": aliases, "versions": rows},
        )
