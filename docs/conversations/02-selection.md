# Model selection

_Recorded 2026-11-15 12:30 UTC against the live stack (scenario `selection`)._

## User

Compare the last five training runs and recommend the best candidate based on F1 score.

### Tool calls (llm mode, 15.6s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `mlflow_compare_runs` | ok | 13284 | Compared 5 runs: 3776f205 logistic_regression f1=0.6575 recall=0.8000 gate=passed; da8b1bd4 logistic_regression f1=0.3373 recall=0.4667 gate=failed; 0e0f38c7 logistic_regression f1=0.6575 recall=0.8000 gate=passed; 4a6e4255 logistic_regression f1=0.6309 recall=0.7833 gate=passed; 3fcf9fac gradient_b |

### Agent

**Summary**  
Both runs `3776f205` and `0e0f38c7` achieved the highest F1 score (0.6575) among the last five training runs, but `3776f205` is recommended as the best candidate since it passed the quality gate and is the first of the tied runs.

**Verified facts**  
- Run `3776f205…` (logistic regression) → F1 = 0.6575, recall = 0.8000, quality‑gate = passed [E1].  
- Run `0e0f38c7…` (logistic regression) → F1 = 0.6575, recall = 0.8000, quality‑gate = passed [E1].  
- Run `4a6e4255…` (logistic regression) → F1 = 0.6309, quality‑gate = passed [E1]; it leads on ROC‑AUC = 0.8094.  
- Runs `da8b1bd4…` (logistic regression) and `3fcf9fac…` (gradient boosting) failed the quality gate [E1].  
- All five runs used dataset version `v2.0` with identical MD5 `21e9a847a6998b24add1efc1fc1ed885` [E1].

**Uncertainties**  
None – all required information was retrieved from the MLflow comparison.

**Recommended actions**  
- Promote run `3776f205…` to the **candidate** alias in the MLflow Model Registry (requires human approval).  
- Verify that the candidate model is registered with the correct dataset version and passes any additional production‑readiness checks before deployment.

_Grounding: verified (17 values checked) · systems: mlflow · request `req_61035135d1f3`_
