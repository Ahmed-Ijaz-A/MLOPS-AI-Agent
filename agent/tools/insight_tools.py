"""Cross-system reasoning tools.

``assess_retraining_need`` applies the explicit retraining policy in ``config/agent_policy.yaml`` to
live evidence from MLflow, DVC, Airflow and Kubernetes. It is deterministic and auditable: each rule
reports fired / not fired / unknown with the evidence it used. It NEVER triggers training - it only
recommends, and says "insufficient evidence" when data sources are unavailable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.base import ToolContext, ToolOutput, ToolRegistry
from training.provenance import read_dvc_file


class RetrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str | None = Field(None, description="Registered model (default from policy).")
    observed_metrics: dict[str, float] | None = Field(
        None, description="Optional production / monitoring metrics supplied by the user, e.g. {'f1': 0.52}."
    )
    dataset_path: str = "data/raw/credit_risk.csv"


def _rule(name: str, fired: bool | None, evidence: str) -> dict[str, Any]:
    return {"rule": name, "fired": fired, "evidence": evidence}


def register(reg: ToolRegistry) -> None:
    @reg.tool(
        "assess_retraining_need",
        "Apply the explicit retraining policy (freshness, minimum F1, data changed since training, failed "
        "quality gate, user-supplied degradation metrics) to live evidence and return a recommendation with "
        "per-rule evidence. Read-only: never starts training.",
        RetrainArgs,
        system="insight",
        timeout_s=60,
    )
    def assess(args: RetrainArgs, ctx: ToolContext) -> ToolOutput:
        policy = ctx.policy.get("retraining", {})
        name = args.model_name or ctx.policy.get("mlflow", {}).get("registered_model_name")
        rules: list[dict[str, Any]] = []
        uncertainties: list[str] = []
        facts: dict[str, Any] = {"model_name": name}

        # 1. which model is serving? (Kubernetes annotation, else MLflow champion alias)
        serving_version = None
        try:
            k = ctx.client("kubernetes")
            ns = ctx.policy.get("kubernetes", {}).get("default_namespace", "mlops-agent")
            dep = k.apps.read_namespaced_deployment("inference-api", ns, _request_timeout=10)
            serving_version = (dep.spec.template.metadata.annotations or {}).get("mlops.agent/model-version")
            facts["deployment_ready"] = f"{dep.status.ready_replicas or 0}/{dep.spec.replicas}"
            facts["serving_version_source"] = "kubernetes"
        except Exception as exc:
            uncertainties.append(
                f"Kubernetes unavailable ({type(exc).__name__}); serving version taken from MLflow alias."
            )

        mv = run = None
        try:
            client = ctx.client("mlflow")
            if serving_version:
                mv = client.get_model_version(name, str(serving_version))
            else:
                mv = client.get_model_version_by_alias(name, "champion")
                facts["serving_version_source"] = "mlflow champion alias"
            run = client.get_run(mv.run_id)
            facts.update({"model_version": mv.version, "run_id": mv.run_id})
        except Exception as exc:
            uncertainties.append(
                f"Could not resolve the serving model in MLflow ({type(exc).__name__}: {str(exc)[:120]})."
            )

        # Rule: freshness
        if mv is not None:
            age = (datetime.now(UTC).timestamp() * 1000 - mv.creation_timestamp) / 86_400_000
            facts["model_age_days"] = round(age, 2)
            limit = policy.get("max_model_age_days", 30)
            rules.append(_rule("freshness", age > limit, f"model v{mv.version} is {age:.1f} days old (limit {limit})"))
        else:
            rules.append(_rule("freshness", None, "serving model unknown"))

        # Rule: minimum offline F1
        if run is not None and "f1" in run.data.metrics:
            f1 = run.data.metrics["f1"]
            facts["model_f1"] = f1
            rules.append(
                _rule(
                    "min_f1",
                    f1 < policy.get("min_f1", 0.0),
                    f"recorded test F1 {f1:.4f} vs minimum {policy.get('min_f1')}",
                )
            )
        else:
            rules.append(_rule("min_f1", None, "no recorded F1 for the serving model"))

        # Rule: data changed since training
        if policy.get("retrain_on_data_change", True):
            try:
                current = read_dvc_file(ctx.settings.repo_root / f"{args.dataset_path}.dvc").get("md5")
                trained = run.data.tags.get("dataset.md5") if run is not None else None
                facts.update(
                    {
                        "current_dataset_md5": current,
                        "training_dataset_md5": trained,
                        "training_dataset_version": run.data.tags.get("dataset.version") if run else None,
                    }
                )
                if current and trained:
                    rules.append(
                        _rule(
                            "data_changed",
                            current != trained,
                            f"model trained on md5 {trained}; DVC now tracks md5 {current}",
                        )
                    )
                else:
                    rules.append(_rule("data_changed", None, "dataset hash missing on one side"))
            except Exception as exc:
                rules.append(_rule("data_changed", None, f"DVC metadata unavailable ({type(exc).__name__})"))

        # Rule: latest training pipeline failed its quality gate
        if policy.get("retrain_on_failed_quality_gate", True):
            try:
                af = ctx.client("airflow")
                runs = af.dag_runs("ml_training_pipeline", 5)
                # Deliberate failure-injection drills are not evidence about the model: judge the most recent
                # *genuine* run, and report the drills that were skipped.
                drills = [r for r in runs if (r.get("conf") or {}).get("inject_failure", "none") not in ("none", "")]
                genuine = [r for r in runs if r not in drills]
                if drills:
                    facts["ignored_failure_injection_runs"] = [
                        {
                            "run_id": r.get("dag_run_id"),
                            "inject_failure": r["conf"]["inject_failure"],
                            "state": r.get("state"),
                        }
                        for r in drills
                    ]
                if genuine:
                    latest = genuine[0]
                    rid = latest.get("dag_run_id") or latest.get("run_id")
                    tis = {t["task_id"]: t.get("state") for t in af.task_instances("ml_training_pipeline", rid)}
                    facts["latest_pipeline_run"] = {"run_id": rid, "state": latest.get("state"), "tasks": tis}
                    gate_failed = tis.get("quality_gate") == "failed"
                    note = f" ({len(drills)} failure-injection drill run(s) ignored)" if drills else ""
                    rules.append(
                        _rule(
                            "failed_quality_gate",
                            gate_failed,
                            f"latest genuine run {rid} is {latest.get('state')}; quality_gate={tis.get('quality_gate')}{note}",
                        )
                    )
                elif runs:
                    rules.append(_rule("failed_quality_gate", False, "only failure-injection drill runs recorded"))
                else:
                    rules.append(_rule("failed_quality_gate", False, "no pipeline runs recorded"))
            except Exception as exc:
                rules.append(_rule("failed_quality_gate", None, f"Airflow unavailable ({type(exc).__name__})"))

        # Rule: user-supplied degradation results
        if args.observed_metrics:
            obs_f1 = args.observed_metrics.get("f1")
            if obs_f1 is not None:
                rules.append(
                    _rule(
                        "observed_degradation",
                        obs_f1 < policy.get("min_f1", 0.0),
                        f"user-supplied production F1 {obs_f1} vs minimum {policy.get('min_f1')}",
                    )
                )
        else:
            uncertainties.append(
                "No production / monitoring metrics were supplied, so live model degradation is unknown."
            )

        fired = [r for r in rules if r["fired"] is True]
        unknown = [r for r in rules if r["fired"] is None]
        if fired:
            decision = "retrain_recommended"
        elif len(unknown) >= max(2, len(rules) // 2 + 1):
            decision = "insufficient_evidence"
        else:
            decision = "no_retraining_needed"
        for r in unknown:
            uncertainties.append(f"Rule '{r['rule']}' could not be evaluated: {r['evidence']}.")
        actions = []
        if decision == "retrain_recommended":
            actions.append(
                "Ask a human to approve triggering 'ml_training_pipeline' (agent will not trigger it automatically)."
            )
            if any(r["rule"] == "failed_quality_gate" for r in fired):
                actions.append("Investigate the failed quality gate first (airflow_get_failed_tasks / task logs).")
        elif decision == "insufficient_evidence":
            actions.append("Restore the unavailable services and re-run the assessment before deciding.")
        else:
            actions.append("No retraining needed under the current policy; keep monitoring.")

        summary = (
            f"Retraining decision: {decision.upper()}. "
            + (
                "Fired rules: " + "; ".join(f"{r['rule']} ({r['evidence']})" for r in fired)
                if fired
                else "No policy rule fired" + (f" ({len(unknown)} rule(s) could not be evaluated)" if unknown else "")
            )
            + "."
        )
        return ToolOutput(
            summary,
            {
                "decision": decision,
                "rules": rules,
                "facts": facts,
                "uncertainties": uncertainties,
                "recommended_actions": actions,
                "policy": policy,
            },
        )
