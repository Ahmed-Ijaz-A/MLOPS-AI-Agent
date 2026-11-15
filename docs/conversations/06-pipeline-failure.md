# Failure investigation: quality gate blocks a bad model

_Recorded 2026-11-15 11:54 UTC against the live stack (scenario `pipeline-failure`)._

## User

Trigger the training pipeline with the low_quality failure injection.

### Tool calls (llm mode, 1.7s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `airflow_trigger_dag` | denied | 0 | DAG conf keys ['failure_injection'] are not allowed. Allowed keys: ['inject_failure', 'model_type', 'reason']. |
| E2 | `airflow_trigger_dag` | requires_confirmation | 2 | NOT EXECUTED. 'Trigger Airflow DAG 'ml_training_pipeline' with conf {'inject_failure': 'low_quality', 'reason': 'User requested low_quality failure injection'}' requires explicit human approval (confirmation id cfm_yxyufIlNGAPM, expires 12:02:15 UTC). |

### Agent

**Summary**  
The training pipeline was not started; a pending confirmation is required to trigger the DAG with the requested failure injection.

**Verified facts**
- The request to trigger the DAG with `inject_failure: low_quality` generated a pending action that needs human approval. Details are in confirmation id `cfm_yxyufIlNGAPM` [E2].

**Uncertainties**
- No execution has occurred yet; the pipeline will only run after the pending confirmation is approved.

**Recommended actions**
- Approve the pending confirmation `cfm_yxyufIlNGAPM` to trigger the training pipeline with the low‑quality failure injection, or cancel if not desired.

_Grounding: verified (0 values checked) · systems: airflow · request `req_ba051f3f3be3`_

> **Operator APPROVED** `cfm_yxyufIlNGAPM` - airflow_trigger_dag {"dag_id": "ml_training_pipeline", "conf": {"inject_failure": "low_quality", "reason": "User requested low_quality failure injection"}} -> **ok**: Triggered ml_training_pipeline: run manual__2026-11-15T11:52:17.885724+00:00_HIAQP8yc is queued. Use airflow_wait_for_dag_run to follow it.

## User

Wait for that pipeline run to finish. Did it fail? If so, find the failed task, read its logs, and explain the root cause and whether any model was promoted.

### Tool calls (llm mode, 109.9s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `airflow_wait_for_dag_run` | ok | 50522 | Run manual__2026-11-15T11:52:17.885724+00:00_HIAQP8yc finished as FAILED: validate_dataset=success, preprocess_data=success, train_model=success, evaluate_and_log=success, quality_gate=failed, register_model=upstream_failed. |
| E2 | `airflow_get_task_logs` | ok | 307 | quality_gate try 1: 7 log lines; root-cause line: Exception: AirflowFailException: Quality gate failed: f1=0.3373 < required 0.6; recall=0.4667 < required 0.65; roc_auc=0.4813 < required 0.75 (at training_pipeline.py:192) |
| E3 | `mlflow_get_model_registry` | ok | 121 | Model 'credit-risk-classifier': 5 version(s); candidate=v5, champion=v2. |
| E4 | `mlflow_list_runs` | ok | 863 | 7 run(s) in 'credit-risk-classification': da8b1bd4 logistic_regression f1=0.3373 gate=failed; 0e0f38c7 logistic_regression f1=0.6575 gate=passed; 4a6e4255 logistic_regression f1=0.6309 gate=passed; 3fcf9fac gradient_boosting f1=0.5872 gate=failed; 2e132372 random_forest f1=0.6466 gate=passed |

### Agent

**Summary**  
The pipeline run failed at the `quality_gate` task because the model metrics did not meet the required thresholds; consequently the downstream `register_model` task was upstream‑failed and no new model was promoted.

**Verified facts**
- DAG run `manual__2026-11-15T11:52:17.885724+00:00_HIAQP8yc` completed with overall state **FAILED**; tasks `validate_dataset`, `preprocess_data`, `train_model`, `evaluate_and_log` succeeded, `quality_gate` failed, and `register_model` was upstream‑failed [E1].
- Logs of the failed `quality_gate` task show:  
  `Quality gate failed: f1=0.3373 < required 0.6; recall=0.4667 < required 0.65; roc_auc=0.4813 < required 0.75` [E2].
- The MLflow model registry currently lists version **5** as the candidate and version **2** as the champion; no promotion occurred from this run [E3].
- The failed run `da8b1bd4742f45668de3eeee982c0e59` (originating from this DAG) recorded metrics `f1=0.3373, recall=0.4667, roc_auc=0.4813` and is marked **quality_gate=failed** [E4].

**Uncertainties**
- None; all relevant information was retrieved from Airflow and MLflow.

**Recommended actions**
- Investigate data or feature drift that caused the low metrics (e.g., compare the current dataset version `v2.0` against previous versions via `dvc_compare_versions`).
- If the issue is resolved, trigger a new training run (via a pending Airflow DAG trigger) after fixing the underlying cause.

_Grounding: verified (10 values checked) · systems: airflow, mlflow · request `req_90096cd4e5a3`_
