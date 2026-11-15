# Failure investigation: quality gate blocks a bad model

_Recorded 2026-11-15 12:37 UTC against the live stack (scenario `pipeline-failure`)._

## User

Trigger the training pipeline with the low_quality failure injection.

### Tool calls (llm mode, 62.1s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `airflow_list_dags` | ok | 2217 | 1 DAG(s): ml_training_pipeline |
| E2 | `airflow_trigger_dag` | denied | 0 | DAG conf keys ['injection'] are not allowed. Allowed keys: ['inject_failure', 'model_type', 'reason']. |
| E3 | `airflow_trigger_dag` | requires_confirmation | 8 | NOT EXECUTED. 'Trigger Airflow DAG 'ml_training_pipeline' with conf {'inject_failure': 'low_quality'}' requires explicit human approval (confirmation id cfm_WhpCo7MvlN59, expires 12:44:05 UTC). |

### Agent

**Summary**  
The training pipeline cannot be started without a human approval. A pending trigger for the DAG `ml_training_pipeline` with the failure injection `low_quality` has been created.

**Verified facts**
- The Airflow environment contains a single DAG named `ml_training_pipeline` (not paused). [E1]
- The requested DAG configuration key `inject_failure` is allowed, but executing the trigger requires explicit human approval. [E3]

**Recommended actions**
- Approve the pending trigger using the confirmation ID `cfm_WhpCo7MvlN59` (expires 12:44:05 UTC). Once approved, the pipeline will run with the low‑quality failure injection.

_Grounding: verified (0 values checked) · systems: airflow · request `req_f944dfadce2b`_

> **Operator APPROVED** `cfm_WhpCo7MvlN59` - airflow_trigger_dag {"dag_id": "ml_training_pipeline", "conf": {"inject_failure": "low_quality"}} -> **ok**: Triggered ml_training_pipeline: run manual__2026-11-15T12:35:03.031763+00:00_uPKLehYo is queued. Use airflow_wait_for_dag_run to follow it.

## User

Wait for that pipeline run to finish. Did it fail? If so, find the failed task, read its logs, and explain the root cause and whether any model was promoted.

### Tool calls (llm mode, 123.8s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `airflow_wait_for_dag_run` | ok | 106060 | Run manual__2026-11-15T12:35:03.031763+00:00_uPKLehYo finished as FAILED: validate_dataset=success, preprocess_data=success, train_model=success, evaluate_and_log=success, quality_gate=failed, register_model=upstream_failed. |
| E2 | `airflow_get_task_logs` | ok | 435 | quality_gate try 1: 7 log lines; root-cause line: Exception: AirflowFailException: Quality gate failed: f1=0.3373 < required 0.6; recall=0.4667 < required 0.65; roc_auc=0.4813 < required 0.75 (at training_pipeline.py:192) |
| E3 | `mlflow_get_model_registry` | ok | 14527 | Model 'credit-risk-classifier': 6 version(s); candidate=v6, champion=v2. |

### Agent

**Summary**  
The pipeline run `manual__2026-11-15T12:35:03.031763+00:00_uPKLehYo` failed at the **quality_gate** task; consequently the **register_model** task was upstream‑failed and no new model was promoted. The current champion model (v2) remains unchanged.

**Verified facts**
- DAG run finished **FAILED** with task states: validate_dataset ✅, preprocess_data ✅, train_model ✅, evaluate_and_log ✅, **quality_gate FAILED**, register_model upstream_failed. [E1]
- **quality_gate** log shows the failure reason: `f1=0.3373 < required 0.6; recall=0.4667 < required 0.65; roc_auc=0.4813 < required 0.75`. [E2]
- Model registry `credit-risk-classifier` has **champion** version **v2** and **candidate** version **v6**; no promotion occurred during this run. [E3]

**Uncertainties**
- None; all relevant states and logs were successfully retrieved.

**Recommended actions**
- Review the training code or data preprocessing to improve the low metrics (e.g., feature engineering, hyper‑parameter tuning) before re‑running the pipeline.
- Once improvements are made, trigger a new DAG run (requires human approval) and ensure the **quality_gate** passes so the new run can be registered as a candidate and later promoted.

_Grounding: verified (6 values checked) · systems: airflow, mlflow · request `req_f235efcc9218`_
