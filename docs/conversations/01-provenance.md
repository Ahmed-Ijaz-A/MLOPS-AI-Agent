# Dataset provenance

_Recorded 2026-11-15 12:26 UTC against the live stack (scenario `provenance`)._

## User

Which dataset version was used to train the model that is currently the champion in the registry, which git commit does it correspond to, and how can I reproduce that exact dataset?

### Tool calls (llm mode, 28.2s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `mlflow_get_model_registry` | ok | 12077 | Model 'credit-risk-classifier': 5 version(s); candidate=v5, champion=v2. |
| E2 | `dvc_resolve_run_dataset` | ok | 510 | Run 2510235a (logistic_regression-v2.0) trained on data/raw/credit_risk.csv version v2.0 (md5 21e9a847a6998b24add1efc1fc1ed885); recorded git commit fb0014b. |
| E3 | `dvc_reproduce_dataset` | ok | 11115 | Dataset v2.0 = commit fb0014b, md5 21e9a847a6998b24add1efc1fc1ed885. Verified: checksum matches. Reproduce with: git checkout fb0014b -- data/raw/credit_risk.csv.dvc && dvc checkout data/raw/credit_risk.csv |

### Agent

**Summary**  
The champion model (MLflow version 2) was trained on dataset **v2.0**, which corresponds to git commit **fb0014b2b97514a2954fb2564816e31016471d19**. The exact dataset can be reproduced from the DVC cache using the commands provided by the `dvc_reproduce_dataset` tool.

**Verified facts**
- The MLflow model registry shows the champion alias points to model version 2, tagged with `dataset.version: v2.0` and `git.commit: fb0014b2b97514a2954fb2564816e31016471d19`[E1].  
- Resolving the run that produced this model confirms it was trained on `data/raw/credit_risk.csv` version **v2.0** with MD5 `21e9a847a6998b24add1efc1fc1ed885` and git commit **fb0014b2b97514a2954fb2564816e31016471d19**[E2].  
- Reproducing dataset version **v2.0** succeeds, verifying the MD5 matches, and the tool supplies the exact commands to materialise the dataset from the DVC cache[E3].

**Recommended actions**
- To recreate the exact training data, run the commands from **E3** (e.g., `git checkout fb0014b -- data/raw/credit_risk.csv.dvc && dvc checkout data/raw/credit_risk.csv` or `dvc get . data/raw/credit_risk.csv --rev fb0014b -o <output.csv>`). No further MLOps actions are required.

_Grounding: verified (6 values checked) · systems: dvc, mlflow · request `req_765e0170e00b`_
