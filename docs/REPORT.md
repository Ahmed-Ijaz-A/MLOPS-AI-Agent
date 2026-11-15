# Technical report: Production-Ready MLOps AI Agent

**Course:** MLOps & Production AI Engineering, Assignment 03 · **Repository:** `Ahmed-Ijaz-A/MLOPS-AI-Agent`

## 1. Summary

I built a conversational agent that operates a real, running ML system: a credit-risk classifier whose
data is versioned with **DVC**, trained by an **Apache Airflow 3** DAG, tracked and registered in
**MLflow**, and served on **Kubernetes**. The agent turns natural-language requests into calls to 33
structured tools. It runs safe reads immediately, puts every state change behind a human approval, and
answers with cited evidence. A verifier checks that every value in an answer came from a tool.

All evidence in this report comes from the live stack (Docker Desktop Kubernetes v1.36, Airflow 3.1.0,
MLflow 3.18, DVC 3.67). The agent ran with `gpt-oss-120b` on Cerebras through the provider-agnostic
LLM layer. Claude and four other providers work the same way, and an offline planner covers the
no-credentials case.

| Requirement | Where it is met | Evidence |
|---|---|---|
| DVC module (≥3 tools) | 7 tools, `agent/tools/dvc_tools.py` | [01-provenance](conversations/01-provenance.md), [dvc.txt](evidence/dvc.txt) |
| MLflow module (≥3 tools) | 7 tools, `agent/tools/mlflow_tools.py` | [02-selection](conversations/02-selection.md), [MLflow runs](../screenshots/04-mlflow-runs.png) |
| Airflow module + DAG (≥3 tools) | 7 tools; `dags/training_pipeline.py` | [03-pipeline](conversations/03-pipeline.md), [Airflow grid](../screenshots/06-airflow-dag-runs.png) |
| Kubernetes module + manifests (≥3 tools) | 11 tools; `kubernetes/` | [04-diagnosis](conversations/04-diagnosis.md), [kubernetes.txt](evidence/kubernetes.txt) |
| Multi-tool integration | orchestrator + retraining policy tool | [05-multi](conversations/05-multi.md), [console](../screenshots/01-console-multitool.png) |
| Testing, reliability, safety | 287 tests, 84 % coverage | Section 8 |
| GitHub + CI | 6-job workflow including a kind deployment | Section 9 |

## 2. Architecture

The full design, with diagrams, is in [architecture.md](../architecture.md). In short:

- **Interface.** FastAPI (`api/main.py`) handles validation, request ids, an optional API key, a web
  console, approval endpoints and the audit log. A CLI (`agent/main.py`) offers the same features.
- **Orchestrator** (`agent/orchestrator.py`). An explicit tool-calling loop runs parallel tool calls
  and feeds back results tagged with evidence ids. Two checks run before an answer leaves it: a
  **coverage check** sends the model back if the question needs a system it never queried, and a
  **grounding check** looks up every decimal, percentage, hash and commit in the answer against the
  successful tool outputs.
- **Executor** (`agent/tools/base.py`). Every tool call goes through it, whether it comes from the LLM,
  the offline planner, or a human approval. It does Pydantic validation, runs policy pre-checks, gates
  approvals, enforces a hard timeout, and classifies errors as `ok`, `error`, `unavailable`, `timeout`,
  `invalid_request`, `denied` or `requires_confirmation`. It also redacts secrets and writes a
  structured log record.
- **Tools.** Modules for DVC, MLflow, Airflow, Kubernetes, and one cross-system policy tool.
- **ML platform.** `infra/docker-compose.yml` runs Postgres, MLflow, and the Airflow api-server,
  scheduler, dag-processor and triggerer. The repository is bind-mounted into Airflow, so the DAG
  trains on the exact DVC-tracked bytes and records the real git commit.

**Data and model.** The data is OpenML `credit-g` (German credit), with the positive class "bad credit
risk". DVC tracks two versions: **v1.0** (the first 700 applications, tag `data-v1.0`) and **v2.0**
(all 1000, tag `data-v2.0`). The quality gate requires F1 ≥ 0.60, recall ≥ 0.65 and ROC AUC ≥ 0.75.
Recall carries the most weight because approving a borrower who later defaults is the costly error.

| Run (seeded) | Data | F1 | Recall | ROC AUC | Gate | Registry |
|---|---|---|---|---|---|---|
| logistic_regression | v1.0 | 0.6122 | 0.7317 | 0.7615 | pass | v1 |
| logistic_regression | v2.0 | **0.6575** | 0.8000 | 0.8058 | pass | v2 → **champion** (deployed) |
| random_forest | v2.0 | 0.6466 | 0.7167 | 0.8019 | pass | v3 |
| gradient_boosting | v2.0 | 0.5872 | 0.5333 | 0.7904 | **fail** | not registered |
| logistic_regression C=0.1 | v2.0 | 0.6309 | 0.7833 | 0.8094 | pass | v4 |

Gradient boosting fails the gate on its own merits, without any injection, because its recall is too
low. `dvc repro` rebuilds the champion's metrics exactly (F1 0.6575, [dvc.txt](evidence/dvc.txt)).

## 3. Tool definitions and permissions

The full reference is generated from the code: [docs/TOOLS.md](TOOLS.md).

| Module | Tools | Write tools (need human approval) |
|---|---|---|
| DVC (7) | list tracked datasets · current version and commit · version history · compare versions (rows, class balance, schema, PSI drift) · detect changes (`dvc status`) · resolve an MLflow run to its dataset version · reproduce a version (materialise to a scratch dir and verify the md5) | none |
| MLflow (7) | list experiments · list runs · run details and artifacts · best run with runner-up, margin and caveats · compare runs with a justified recommendation · compare experiments · registry versions and aliases | none |
| Airflow (7) | list DAGs · list runs · run with task states · failed tasks · task logs with root-cause extraction · wait for run · trigger | `airflow_trigger_dag` (*operate*) |
| Kubernetes (11) | workloads · pod status · pod logs · services with ready endpoints · events · unhealthy pods · deployed version with consistency check against the registry · one-shot diagnosis · restart · scale · rollback | restart, scale, rollback (*mutate*) |
| Insight (1) | `assess_retraining_need`: an explicit policy evaluated on live evidence; never triggers anything | none |

**Permissions are enforced in code** (`config/agent_policy.yaml`, `agent/safety.py`):

- Kubernetes reads and writes are limited to the `mlops-agent` namespace. Only `inference-api` can be
  changed, with replicas between 1 and 5. Only `ml_training_pipeline` can be triggered, and only with
  the conf keys `inject_failure`, `model_type` and `reason`. DVC and git run allow-listed sub-commands
  only (no `push`, `gc`, `remove`, `checkout` or `reset`), with no shell and no metacharacters.
- Policy violations are **denied before any approval is offered**.
- Every write needs a one-time confirmation token. It expires after 10 minutes, is bound to the
  session, and carries a SHA-256 fingerprint of the exact arguments. Only a human action can approve
  it: `POST /api/confirmations/{id}` or the CLI's y/N prompt. The model has no approval tool. Each
  request, approval, rejection and expiry is written to `logs/audit.jsonl`.
- On the cluster: a namespaced Role (no delete, exec or secrets), a `restricted` Pod Security level, a
  non-root read-only container, a PDB, and `maxUnavailable: 0`.

## 4. Implementation highlights per module

**DVC.** Provenance runs end to end. Each MLflow run records `dataset.md5`, `dataset.version` and
`git.commit`. `dvc_resolve_run_dataset` walks the git history of the `.dvc` pointer to find which tagged
commit holds those exact bytes. It also flags runs whose data was never committed, and runs whose
declared version disagrees with the bytes. `dvc_reproduce_dataset` demonstrates a reproducible
checkout without touching the working tree: it pulls the version with `dvc get` into
`artifacts/repro/` and checks the md5.

**MLflow.** The tools use `MlflowClient` only. `mlflow_find_best_run` and `mlflow_compare_runs` return a
recommendation together with a caveat list: gate status, margins below noise, ties, mixed dataset
versions, failure-injection runs, run age, and the reminder that offline metrics alone don't make a
model production-ready. In [02-selection](conversations/02-selection.md) the agent spotted that the two
top runs **tie** at F1 0.6575, and that a third run leads on ROC AUC.

**Airflow.** The DAG uses the Airflow 3 Task SDK. Every task gets 2 retries, 15 seconds apart.
Deterministic failures (invalid data, data out of sync with DVC, a failed quality gate) raise
`AirflowFailException`, so they fail fast instead of retrying. `register_model` runs only if everything
upstream succeeded. Three injectable failure modes (`corrupt_data`, `low_quality`, `flaky_training`)
exercise these paths. The client authenticates with a JWT from `POST /auth/token` and refreshes it
once on a 401.

**Kubernetes.** The Deployment runs 2 replicas, a ClusterIP Service, readiness (`/ready`) and liveness
(`/health`) probes, CPU and memory requests and limits, and a ConfigMap. The model is baked into the
image, so the image tag is the model version, and provenance annotations allow
`k8s_get_deployed_version` to cross-check the image, label, ConfigMap and MLflow `champion` alias.
`k8s_diagnose_deployment` classifies:
- ImagePullBackOff
- CrashLoopBackOff, with the previous container's logs
- OOMKilled
- failing readiness probes
- real scheduling failures
- ConfigMap values that appear in crash logs
- Services with zero ready endpoints

Each classification comes with its evidence and a suggested fix.

**Integration.** The policy tool turns "should we retrain?" into auditable rules: freshness of 30 days,
minimum F1 0.60, data changed since training, latest genuine pipeline run failed its gate, and
user-supplied degradation. It ignores failure-injection drills. The LLM explains the decision but
never invents it, and retraining is never started from this question.

## 5. Evidence

| Screenshot | Shows |
|---|---|
| [01-console-multitool](../screenshots/01-console-multitool.png) | The mandatory scenario in the web console: cited answer, grounding verified, 7-step evidence trace |
| [02-console-approval-pending](../screenshots/02-console-approval-pending.png) | A restart request becomes an approval card; tool status `REQUIRES CONFIRMATION`, "NOT EXECUTED" |
| [03-console-approval-executed](../screenshots/03-console-approval-executed.png) | After the human approved: executed, result shown |
| [04-mlflow-runs](../screenshots/04-mlflow-runs.png) | Runs from the seed script and from Airflow, registered versions v1–v5, the three runs that crashed on missing `skops` |
| [05-mlflow-registry](../screenshots/05-mlflow-registry.png) | Registry versions, each carrying provenance tags (`dataset.version`, `dataset.md5`, `git.commit`, `quality_gate`, `f1`); `candidate` alias on the newest gate-passing version |
| [06-airflow-dag-runs](../screenshots/06-airflow-dag-runs.png) | Grid of real runs: `skops` crash (evaluate red), a full success, a quality-gate failure with `register_model` blocked |
| [07-airflow-dag-graph](../screenshots/07-airflow-dag-graph.png) | The six-stage DAG |

Exports: [dvc.txt](evidence/dvc.txt) (tags, `.dvc` pointers, `dvc diff`, `dvc dag`, `dvc repro`
metrics, `dvc status`), [kubernetes.txt](evidence/kubernetes.txt) (all objects, rollout history,
probes and resources, `/version`, `/predict` and `/metrics` through the ClusterIP Service), and
[failure-01-missing-skops.json](evidence/failure-01-missing-skops.json).

## 6. Five tested agent conversations (plus three more)

Each transcript lists the user message, every tool call with status, latency and evidence, the agent's
answer, and the grounding result. `scripts/run_demos.py` re-records them all against the live stack.

| # | Scenario | Tools the agent chose | Outcome |
|---|---|---|---|
| [1](conversations/01-provenance.md) | **Dataset provenance**: which data trained the champion, and how to reproduce it | `mlflow_get_model_registry` → `dvc_resolve_run_dataset` → `dvc_reproduce_dataset` | Champion v2 ← run `2510235a` ← dataset **v2.0** (md5 `21e9a847…`, commit `fb0014b`); reproduction verified by checksum; exact commands given |
| [2](conversations/02-selection.md) | **Model selection**: compare the last five runs by F1 | `mlflow_compare_runs` | Gate-passing recommendation; two runs flagged as an exact **tie** at 0.6575; failed-gate runs excluded; ROC AUC leader noted |
| [3](conversations/03-pipeline.md) | **Pipeline execution**: trigger and report | `airflow_trigger_dag` (pending) → *operator approves* → `airflow_wait_for_dag_run` | Real Airflow run **SUCCESS**; all six tasks reported; new model registered as `candidate` |
| [4](conversations/04-diagnosis.md) | **Deployment diagnosis**: "Why is my prediction API unavailable?" (injected) | `k8s_diagnose_deployment`, `mlflow_get_model_registry`, `mlflow_get_run` | Root cause: **ConfigMap `MODEL_PATH` points at a missing file** (quoted crash log); API degraded but serving thanks to `maxUnavailable: 0`; restart proposed and left pending |
| [5](conversations/05-multi.md) | **Multi-tool reasoning**: the mandatory integration scenario | `mlflow_find_best_run`, `mlflow_get_model_registry`, `k8s_get_deployed_version`, `assess_retraining_need`, `airflow_list_dag_runs`, `dvc_get_dataset_version` | All four systems + policy correlated; **no retraining needed** with per-rule evidence; uncertainty stated (no production metrics); promotion left as a human step |
| [6](conversations/06-pipeline-failure.md) | Failure: quality gate blocks a bad model | trigger (one denied for a bad conf key, then pending) → approve → wait → logs → registry | Gate failed (F1 0.3373, recall 0.4667, AUC 0.4813); `register_model` upstream-failed; registry confirms nothing was promoted |
| [7](conversations/07-image-failure.md) | Failure: ImagePullBackOff after a bad release | diagnose → `k8s_rollback_deployment` (pending) → approve → verify | Rolled back to the previous revision; serving model v2 again, 2/2 ready |
| [8](conversations/08-safety.md) | Safety: unauthorized changes | `k8s_list_workloads` (denied ×2), `k8s_scale_deployment` (denied) | Namespace `kube-system` refused; scaling to 10 refused by the 1–5 bound; nothing offered for approval |

Every answer passed the grounding check (`docs/conversations/*.md`, footer of each turn).

## 7. Failure investigations

### 7.1 Unexpected failure: first real DAG run crashed in `evaluate_and_log`

- **Symptom.** The first Airflow run triggered through the agent finished `failed`.
  `validate_dataset`, `preprocess_data` and `train_model` succeeded; `evaluate_and_log` failed after 3
  attempts; `quality_gate` and `register_model` were `upstream_failed`.
- **Diagnosis by the agent.** `airflow_get_failed_tasks` identified the failed task and the blocked
  ones. `airflow_get_task_logs` fetched the attempt's structured log and pulled out the root cause:
  `ModuleNotFoundError: No module named 'skops'` (raw output in
  [failure-01-missing-skops.json](evidence/failure-01-missing-skops.json)).
- **Root cause.** MLflow 3 saves sklearn models with **skops**, a format that doesn't execute arbitrary
  code on load. My development environment had the full `mlflow` package, but the Airflow image used
  `mlflow-skinny`, which does not install `skops`.
- **Resolution.** I pinned `skops==0.16.0` in `infra/airflow/Dockerfile` and rebuilt. Two side fixes
  came out of this. Airflow 3 reports exceptions as structured lists, so the log tool now renders them
  as readable `Type: message (at file:line)`. And skops refuses unknown types, so
  `sklearn.tree._tree.Tree` was added explicitly as the *only* trusted type, which the tree models need.
- **Verification.** The next run ([03-pipeline](conversations/03-pipeline.md)) finished **SUCCESS** on
  all six tasks.
- **What the safety design did.** The retries handled what could have been a transient error. The DAG
  dependencies guaranteed that no model was registered from the broken run.

### 7.2 Injected failure: CrashLoopBackOff from a bad ConfigMap

- **Injection.** `scripts/inject_failure.py bad-config` sets `MODEL_PATH=/app/model/model-MISSING.joblib`
  and restarts the deployment.
- **Diagnosis.** `k8s_diagnose_deployment` reported `DEGRADED`, the new pod in CrashLoopBackOff, and the
  previous container's log line `FileNotFoundError: model file not found:
  /app/model/model-MISSING.joblib`.
- **Engineering iteration.** In the first recording the model blamed the *container image*, because the
  diagnosis output never included the ConfigMap. I added a correlation rule: a ConfigMap value that
  appears in a crash log produces a "Misconfigured ConfigMap value" finding. The tool also now includes
  the ConfigMap data in its evidence. The re-recorded diagnosis names `inference-config/MODEL_PATH` as
  the root cause and says explicitly that the image is not implicated.
- **Why the API stayed up.** With `maxUnavailable: 0`, Kubernetes kept both old pods serving while the
  new one crash-looped, so the outage was a *degraded rollout*, not downtime. The agent reported
  exactly that.
- **Resolution.** Restore the ConfigMap and restart. The follow-up question confirmed `HEALTHY`, 2/2
  ready.

### 7.3 Injected failure: ImagePullBackOff, fixed with an approved rollback

- **Injection.** `inject_failure.py bad-image` deploys `mlops-inference:model-v404-does-not-exist`.
- **Diagnosis and fix.** The diagnosis reported `[critical] Image cannot be pulled`. The agent proposed
  `k8s_rollback_deployment`, the operator approved, and the deployment went from revision 14 back to 13.
  `k8s_get_deployed_version` then confirmed model v2, 2/2 ready, with metadata consistent with the
  registry.
- **Bug found from this evidence.** An earlier version of the diagnosis also reported "Pod cannot be
  scheduled". A pod waiting on an image is `Pending` but *is* scheduled. The rule now requires a missing
  node or a `FailedScheduling` event, and has a regression test.

### 7.4 Injected failure: quality gate stops a bad model

`{"inject_failure": "low_quality"}` shuffles the training labels. The gate failed (F1 0.3373 < 0.60,
recall 0.4667 < 0.65, ROC AUC 0.4813 < 0.75), `register_model` was blocked, and the registry still showed
champion v2. The policy tool later **ignores** this drill when it decides whether to retrain.

### 7.5 Other unexpected issues found while building, and how they were fixed

| Issue | How it was diagnosed | Fix |
|---|---|---|
| Gemini API: `403 free tier not available for this project` | live probe of the provider | provider-agnostic layer; switched to Cerebras with no code change in the agent |
| Docker builds failed with `input/output error` | C: drive had 0.1 GB free | freed space; README troubleshooting entry |
| Secret redaction masked `gate_pass_rate`, and would have masked `input_tokens` | failing MLflow test | tightened key patterns (`passw(or)?d`, `token` not followed by `s`) + tests |
| pandas 3 text columns are `str`, not `object` | DVC drift test failure | `is_numeric_dtype` |
| MLflow 3.18 returns alias versions as `int` | registry test failure | normalised to strings |
| Hallucinated value at end of sentence (`0.9431.`) not checked | grounding test failure | fixed regex boundary |
| Windows console (cp1252) crashed on model / MLflow unicode | live CLI run | UTF-8 reconfiguration in entry points |
| Model wrapped citations in zero-width / full-width brackets (`【E1】`, `[ E1 ]`) | review of transcripts | citation normalisation + tests |
| Cold start: first MLflow call took >20 s | red status pill in the console | background client warm-up + parallel health probes |
| **Security gap found by CI:** on Linux, `C:/Windows/system.ini` was not treated as absolute, so the dataset-path guard answered "not tracked" (`invalid_request`) instead of **denied** | the `test` job failed on GitHub Actions; reproduced in a clean Linux container (`test_path_traversal_denied`) | the guard checks with both `PureWindowsPath` and `PurePosixPath` (drive letters, UNC paths and `..` rejected on every OS); all 6 CI jobs now green |

## 8. Testing, reliability and safety

**287 tests, 84 % line coverage, about 2 minutes, no services required.** DVC tests run against a real
temporary git+DVC repository with two tagged versions, and MLflow tests against a real temporary MLflow
store. Kubernetes, Airflow (an in-memory REST API over `httpx.MockTransport`) and the LLM (a scripted
model) are faked.

| Required test | Test(s) |
|---|---|
| ≥ 8 meaningful tests | 287 across 14 test files |
| Mocked tool responses for unavailable services | `test_no_kubeconfig_is_unavailable`, `test_connection_refused_unavailable`, `test_unreachable_tracking_server_is_unavailable`, `test_insufficient_evidence_when_services_down` |
| Successful tool selection and execution | `test_llm_selects_and_executes_tools`, `test_parallel_tool_calls_return_in_one_message`, `test_planner_maps_intents_to_tools` (16 intents), `test_mandatory_scenario_consults_all_four_systems` |
| Tool timeouts | `test_tool_timeout_reported`, `test_http_timeout_reported_as_timeout` |
| Invalid requests | `test_invalid_arguments_rejected_before_execution`, `test_malformed_ids_rejected`, `test_malformed_run_id_rejected`, `test_invalid_requests_rejected` (API) |
| Service failures | `test_server_error_unavailable`, `test_cluster_unreachable_maps_to_unavailable`, `test_exceptions_mapped_to_statuses`, `test_internal_error_never_crashes_or_invents` |
| **Agent does not invent results when a tool is unavailable** | `test_agent_does_not_invent_results_when_tool_unavailable`: a scripted model claims F1 0.9412 after MLflow failed; the grounding check flags both the number and the commit. Also `test_offline_agent_reports_unavailability_instead_of_values` |
| **Unauthorized Kubernetes modifications rejected** | `test_unauthorized_kubernetes_modifications_rejected` (5 cases: wrong namespaces, out-of-bounds replicas, non-mutable deployment; nothing patched, no approval offered), `test_unauthorized_namespace_request_rejected_end_to_end`, `test_rbac_forbidden_maps_to_denied` |
| Structured logs (tool, status, duration, request id; no secrets) | `test_tool_call_log_record_has_required_fields`, `test_logs_never_contain_secrets`, redaction tests for keys, bearer tokens, JWTs, `sk-ant-` keys and GitHub tokens |
| Approval flow | one-time use, expiry, session binding, tamper detection, decline never executes, HTTP replay → 409 |

A live log record (`logs/agent.jsonl`) looks like this:
`{"ts": "...", "level": "INFO", "request_id": "req_f25fbffdc056", "event": "tool_call", "tool": "k8s_diagnose_deployment", "system": "kubernetes", "status": "ok", "duration_ms": 3641.2, "evidence_id": "E1", "arguments": {...}}`

## 9. Repository and CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pull requests and pushes. External
services are mocked in ordinary runs.
- **`lint`:** ruff check and ruff format.
- **`test`:** the full suite with a 75 % coverage gate and JUnit and coverage artifacts.
- **`training-smoke`:** fetches the real dataset and runs the pipeline.
- **`dag-integrity`:** the DAG is parsed by real Airflow 3.1.0 under the official constraints, and its
  tasks, dependencies and retries are asserted.
- **`manifests`:** strict kubeconform on the rendered kustomization.
- **`docker-and-kind`:** builds and smoke-tests the inference image, deploys it to an ephemeral kind
  cluster, and has the agent diagnose the live deployment through the real Kubernetes API.

**Status at submission: all six jobs pass** (run for commit `17430f4`, 4 min 27 s). Getting there took one
fix: the first runs failed in `test` on a Linux-only path-handling gap (section 7.5).

The history uses conventional commits, with each feature, fix and evidence update as a separate commit.

## 10. Known limitations and proposed improvements

| Limitation | Proposed improvement |
|---|---|
| The DVC remote is a local folder | S3 / GCS / MinIO remote with credentials from a secret store |
| Session memory and pending approvals are in-process (lost on restart, single replica) | Redis-backed store, so the agent API can scale horizontally |
| Confirmation tokens aren't tied to an authenticated identity (an optional shared API key only) | OIDC / SSO with role-based approvers, and a two-person rule for production namespaces |
| No live model monitoring, so "degradation" must be supplied by the user | Prometheus + an evaluation job feeding `observed_metrics` into the retraining policy automatically |
| Small free-tier LLMs vary between runs; quality depends on prompt rules and coverage checks | Evaluation set over the recorded scenarios, run in CI against the configured provider; a stronger default model when budget allows |
| Promotion to `champion` is a script, not an agent tool | An approval-gated `mlflow_promote_model` tool that also opens a deployment PR (GitOps with Argo CD) |
| The model is baked into the image, so the image must be rebuilt per version | Fine for this scale; at larger scale use a model server (KServe / MLServer) pulling from the registry |
| Tool timeouts can't cancel already-running threads | Process-level isolation for long-running tools |
