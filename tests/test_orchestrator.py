"""Agent loop: tool selection, multi-tool reasoning, failures, honesty, confirmations."""

from __future__ import annotations

import pytest

from agent import planner
from agent.orchestrator import Agent, required_systems
from tests.fakes import (
    AirflowMock,
    FakeMlflow,
    ScriptedLLM,
    fake_run,
    fake_version,
    healthy_cluster,
    pipeline_runs,
    text_block,
    tool_block,
)

RUN = "a" * 32


def _agent(make_executor, settings, llm=None, clients=None):
    return Agent(settings, executor=make_executor(clients or {}), llm=llm)


def _all_clients():
    from agent.tools.airflow_tools import AirflowClient

    runs, tasks = pipeline_runs()
    mock = AirflowMock(runs=runs, tasks=tasks)
    return {
        "kubernetes": healthy_cluster(),
        "mlflow": FakeMlflow(
            runs={RUN: fake_run(RUN)}, versions={"3": fake_version("3", RUN)}, aliases={"champion": "3"}
        ),
        "airflow": AirflowClient("http://airflow.test", "airflow", "pw", transport=mock.transport()),
    }, mock


# ------------------------------------------------------------------ LLM mode (scripted)


def test_llm_selects_and_executes_tools(make_executor, settings):
    llm = ScriptedLLM(
        [
            [text_block("Checking the cluster."), tool_block("k8s_diagnose_deployment", {"namespace": "mlops-agent"})],
            [text_block("**Summary** - inference-api is healthy [E1].")],
        ]
    )
    resp = _agent(make_executor, settings, llm, {"kubernetes": healthy_cluster()}).chat(
        "Why is my prediction API unavailable?"
    )
    assert [t.tool for t in resp.tool_calls] == ["k8s_diagnose_deployment"]
    assert resp.tool_calls[0].status == "ok" and "healthy" in resp.answer and resp.mode == "llm"


def test_parallel_tool_calls_return_in_one_message(make_executor, settings):
    clients, _ = _all_clients()
    llm = ScriptedLLM(
        [
            [
                tool_block("k8s_list_workloads", {}, "t1"),
                tool_block("airflow_list_dag_runs", {}, "t2"),
                tool_block("mlflow_get_model_registry", {}, "t3"),
            ],
            [text_block("done")],
        ]
    )
    _agent(make_executor, settings, llm, clients).chat("status of pods and pipeline runs")
    second_call = llm.calls[1]
    results = second_call[-1]["content"]
    assert second_call[-1]["role"] == "user" and len(results) == 3
    assert {r["tool_use_id"] for r in results} == {"t1", "t2", "t3"}
    assert all(r["content"].startswith("[E") for r in results)


def test_tool_errors_are_returned_to_model_as_errors(make_executor, settings):
    llm = ScriptedLLM([[tool_block("mlflow_list_runs", {})], [text_block("MLflow is unavailable; no metrics known.")]])
    _agent(make_executor, settings, llm).chat("list runs")
    result = llm.calls[1][-1]["content"][0]
    assert result["is_error"] is True and "unavailable" in result["content"]


def test_agent_does_not_invent_results_when_tool_unavailable(make_executor, settings):
    """A model that hallucinates a metric after the tool failed gets caught by the grounding check."""
    llm = ScriptedLLM(
        [
            [tool_block("mlflow_find_best_run", {"metric": "f1"})],
            [text_block("**Summary** - The best model has F1 = 0.9412 at commit 7f3a9c2.")],
        ]
    )
    resp = _agent(make_executor, settings, llm).chat("What is the best model?")
    assert resp.tool_calls[0].status == "unavailable"
    assert resp.grounding.grounded is False
    assert {"0.9412", "7f3a9c2"} <= set(resp.grounding.unsupported_values)
    assert "Grounding check" in resp.answer


def test_offline_agent_reports_unavailability_instead_of_values(make_executor, settings):
    resp = _agent(make_executor, settings).chat("What is the best model by F1?")
    assert resp.tool_calls[0].status == "unavailable"
    assert "could not verify anything" in resp.answer and "no facts from this source" in resp.answer
    assert resp.grounding.grounded and resp.grounding.checked_values == 0


def test_coverage_enforcer_nudges_for_missing_systems(make_executor, settings):
    clients, _ = _all_clients()
    llm = ScriptedLLM(
        [
            [tool_block("mlflow_get_model_registry", {})],
            [text_block("Looks fine.")],  # skipped DVC, Airflow and Kubernetes
            [
                tool_block("dvc_get_dataset_version", {}, "d"),
                tool_block("k8s_get_deployed_version", {}, "k"),
                tool_block("airflow_list_dag_runs", {}, "a"),
            ],
            [text_block("**Summary** - full answer.")],
        ]
    )
    resp = _agent(make_executor, settings, llm, clients).chat("Should we retrain the model?")
    nudge = llm.calls[2][-1]["content"]
    assert "[orchestrator]" in nudge and "airflow" in nudge and "kubernetes" in nudge
    assert {"mlflow", "kubernetes", "airflow", "dvc"} <= set(resp.systems_consulted)


def test_mutation_request_only_creates_pending_action(make_executor, settings):
    k8s = healthy_cluster()
    llm = ScriptedLLM(
        [
            [tool_block("k8s_restart_deployment", {"namespace": "mlops-agent", "deployment": "inference-api"})],
            [text_block("The restart awaits your approval.")],
        ]
    )
    agent = _agent(make_executor, settings, llm, {"kubernetes": k8s})
    resp = agent.chat("restart the inference api")
    assert resp.tool_calls[0].status == "requires_confirmation"
    assert len(resp.pending_confirmations) == 1 and k8s.apps.patches == []
    assert llm.calls[1][-1]["content"][0]["is_error"] is False


def test_human_approval_executes_and_decline_does_not(make_executor, settings):
    k8s = healthy_cluster()
    agent = _agent(make_executor, settings, None, {"kubernetes": k8s})
    pending = agent.chat("restart the inference api").pending_confirmations[0]
    out = agent.decide(pending.confirmation_id, approve=True, session_id=None)
    assert out["status"] == "approved" and out["result"]["status"] == "ok" and len(k8s.apps.patches) == 1
    again = agent.decide(pending.confirmation_id, approve=True)
    assert again["status"] == "rejected"  # one-time token
    pending2 = agent.chat("scale inference api to 3 replicas").pending_confirmations[0]
    assert agent.decide(pending2.confirmation_id, approve=False)["status"] == "declined"
    assert len(k8s.apps.patches) == 1


def test_unauthorized_namespace_request_rejected_end_to_end(make_executor, settings):
    k8s = healthy_cluster()
    resp = _agent(make_executor, settings, None, {"kubernetes": k8s}).chat(
        "restart inference-api in namespace kube-system"
    )
    assert resp.tool_calls[0].status == "denied" and resp.pending_confirmations == []
    assert "refused by policy" in resp.answer and k8s.apps.patches == []


def test_refusal_stop_reason_handled(make_executor, settings):
    llm = ScriptedLLM([[text_block("")]], stop_reasons=["refusal"])
    resp = _agent(make_executor, settings, llm).chat("something")
    assert "declined" in resp.answer and resp.tool_calls == []


def test_max_steps_guard(make_executor, settings):
    from dataclasses import replace

    llm = ScriptedLLM([[tool_block("k8s_list_workloads", {}, f"t{i}")] for i in range(10)])
    agent = Agent(
        replace(settings, max_agent_steps=3), executor=make_executor({"kubernetes": healthy_cluster()}), llm=llm
    )
    resp = agent.chat("loop forever")
    assert len(resp.tool_calls) == 3 and "maximum number of reasoning steps" in resp.answer


def test_session_memory_carries_previous_turns(make_executor, settings):
    llm = ScriptedLLM([[text_block("first answer")], [text_block("second answer")]])
    agent = _agent(make_executor, settings, llm)
    r1 = agent.chat("hello there")
    agent.chat("and again", session_id=r1.session_id)
    assert llm.calls[1][0] == {"role": "user", "content": "hello there"}


def test_internal_error_never_crashes_or_invents(make_executor, settings):
    class Broken:
        name = "broken"

        def respond(self, *a):
            raise RuntimeError("API overloaded")

    resp = _agent(make_executor, settings, Broken()).chat("anything")
    assert "internal error" in resp.answer and "No operational results" in resp.answer


# ------------------------------------------------------------------ mandatory integration scenario (offline)

MANDATORY = (
    "Find the best recent model, identify the dataset version used to train it, check the deployment "
    "status, and explain whether retraining is necessary."
)


def test_mandatory_scenario_consults_all_four_systems(make_executor, settings, monkeypatch):
    clients, mock = _all_clients()
    clients["mlflow"].search_runs = None  # find_best_run needs a real store; covered in test_mlflow_tools
    resp = _agent(make_executor, settings, None, clients).chat(MANDATORY)
    tools = [t.tool for t in resp.tool_calls]
    for expected in (
        "mlflow_find_best_run",
        "k8s_get_deployed_version",
        "k8s_diagnose_deployment",
        "airflow_list_dag_runs",
        "dvc_get_dataset_version",
        "assess_retraining_need",
    ):
        assert expected in tools
    assert {"kubernetes", "airflow", "insight"} <= set(resp.systems_consulted)
    assert "**Uncertainties**" in resp.answer and "**Recommended actions**" in resp.answer
    assert mock.triggered == []  # retraining is never started from this request


def test_required_systems_detection():
    assert required_systems(MANDATORY) == {"dvc", "mlflow", "airflow", "kubernetes"}
    assert required_systems("Why is my prediction API unavailable?") == {"kubernetes"}
    assert required_systems("which dataset version is current") == {"dvc"}


# ------------------------------------------------------------------ offline planner intents


@pytest.mark.parametrize(
    "message,intent,first_tool",
    [
        (MANDATORY, "integration", "mlflow_find_best_run"),
        ("Which dataset version was used for run " + "b" * 32 + "?", "dvc_run_provenance", "dvc_resolve_run_dataset"),
        ("Compare dataset v1.0 and v2.0", "dvc_compare", "dvc_compare_versions"),
        ("How do I reproduce dataset version v1.0?", "dvc_reproduce", "dvc_reproduce_dataset"),
        ("Has the tracked data changed?", "dvc_status", "dvc_check_data_status"),
        ("Which dataset version is current?", "dvc_version", "dvc_get_dataset_version"),
        (
            "Compare the last five training runs and recommend the best candidate based on F1 score.",
            "mlflow_compare",
            "mlflow_compare_runs",
        ),
        ("What is the best model by ROC AUC?", "mlflow_best", "mlflow_find_best_run"),
        ("Show the model registry", "mlflow_registry", "mlflow_get_model_registry"),
        (
            "Trigger the model training pipeline and report whether execution succeeded.",
            "airflow_trigger",
            "airflow_trigger_dag",
        ),
        ("Why did the last training pipeline run fail?", "airflow_failure", "airflow_get_failed_tasks"),
        ("Why is my prediction API unavailable?", "k8s_diagnose", "k8s_diagnose_deployment"),
        ("Which model version is deployed on kubernetes?", "k8s_version", "k8s_get_deployed_version"),
        ("Restart the inference deployment", "k8s_restart", "k8s_restart_deployment"),
        ("Scale the api to 4 replicas", "k8s_scale", "k8s_scale_deployment"),
        ("Roll back the deployment", "k8s_rollback", "k8s_rollback_deployment"),
    ],
)
def test_planner_maps_intents_to_tools(message, intent, first_tool):
    got_intent, steps = planner.plan(message)
    assert got_intent == intent
    first = steps[0]
    assert (first[0] if isinstance(first, tuple) else None) == first_tool


def test_planner_extracts_parameters():
    _, steps = planner.plan("scale the api to 4 replicas in namespace mlops-agent")
    assert steps[0][1] == {"namespace": "mlops-agent", "deployment": "inference-api", "replicas": 4}
    _, steps = planner.plan("trigger training with low quality injection")
    assert steps[0][1]["conf"]["inject_failure"] == "low_quality"


def test_zero_width_characters_stripped_from_citations(make_executor, settings):
    llm = ScriptedLLM([[text_block("**Summary** - healthy [​E1​].")]])
    resp = _agent(make_executor, settings, llm).chat("status")
    assert "[E1]" in resp.answer and "​" not in resp.answer


def test_fullwidth_citations_normalised(make_executor, settings):
    llm = ScriptedLLM([[text_block("F1 is high【E2】 and gate passed【E3†L4】.")]])
    resp = _agent(make_executor, settings, llm).chat("status")
    assert "[E2]" in resp.answer and "[E3]" in resp.answer and "【" not in resp.answer
