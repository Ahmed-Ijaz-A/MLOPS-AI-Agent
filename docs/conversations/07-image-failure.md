# Failure investigation: ImagePullBackOff after a bad release

_Recorded 2026-11-15 11:56 UTC against the live stack (scenario `image-failure`)._

> **Operator ran:** `python.exe scripts/inject_failure.py bad-image`

## User

The last release seems broken. Diagnose the inference-api deployment and propose a fix.

### Tool calls (llm mode, 37.4s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_diagnose_deployment` | ok | 10452 | inference-api is DEGRADED: 2/2 replicas ready, 2 ready endpoint(s); [critical] Image cannot be pulled; [high] Pod cannot be scheduled. |
| E2 | `mlflow_get_model_registry` | timeout | 20015 | mlflow_get_model_registry did not respond within 20s. |
| E3 | `mlflow_get_model_registry` | ok | 1013 | Model 'credit-risk-classifier': 5 version(s); candidate=v5, champion=v2. |
| E4 | `k8s_get_deployed_version` | ok | 105 | inference-api runs mlops-inference:model-v404-does-not-exist = model credit-risk-classifier v2 (run 2510235a, dataset v2.0), revision 7, ready 2/2. Consistency issues: ["image tag 'model-v404-does-not-exist' does not encode model version 2"] |

### Agent

**Summary**  
The inference‑api deployment is degraded because a newly created pod tries to pull a non‑existent image tag (`mlops-inference:model‑v404‑does‑not‑exist`), causing an `ImagePullBackOff`. The deployment’s image tag does not match the champion model version (model‑v2) registered in MLflow, creating a consistency issue.

**Verified facts**  
- Deployment state: DEGRADED, 2/2 replicas ready, 1 pod pending with `ImagePullBackOff` [ E1 ].  
- The failing pod (`inference-api-8697655964-2jgkz`) is using image `mlops-inference:model‑v404‑does‑not‑exist` [ E1 ].  
- MLflow model registry shows the current champion is version 2 of *credit‑risk‑classifier* (run `2510235a…`, dataset v2.0) [ E3 ].  
- Deployed version metadata reveals the deployment is configured with image tag `model‑v404‑does‑not‑exist` while the app version label is `model‑v2`; the image tag does **not** encode the champion model version 2 [ E4 ].  
- The deployment’s ConfigMap also reports `MODEL_VERSION=2`, confirming the intended model version 2 [ E4 ].

**Uncertainties**  
- No evidence was obtained for the exact state of the previous successful revision (revision 6) – the rollback would rely on Kubernetes’ stored rollout history.

**Recommended actions**  
1. **Rollback to the last good revision** – create a pending rollback action to revert the `inference-api` deployment to its previous revision, which used the correct image tag.  
   ```json
   {"namespace":"mlops-agent","deployment":"inference-api","to_revision":"6"}
   ```  
   *(Pending approval; confirmation ID will be provided by the system.)*  

2. **Redeploy the champion model correctly** – after the rollback (or directly), run the human‑approved deployment script to ensure the image tag aligns with the champion version:  
   ```bash
   python scripts/deploy_model.py --version 2
   ```  
   This will build/push the correct Docker image (`mlops-inference:model-v2`) and update the deployment to use it.  

3. **Verify post‑deployment health** – once the rollout completes, run `k8s_diagnose_deployment` again to confirm the deployment is `HEALTHY` and that no `ImagePullBackOff` or readiness‑probe errors remain.  

These steps address the critical image‑pull failure and restore consistency between the deployment and the MLflow champion model.

_Grounding: verified (1 values checked) · systems: kubernetes, mlflow · request `req_5f3750d613bd`_

## User

Roll the inference-api deployment back to the previous revision.

### Tool calls (llm mode, 1.2s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_rollback_deployment` | requires_confirmation | 9 | NOT EXECUTED. 'Roll back mlops-agent/inference-api to the previous revision' requires explicit human approval (confirmation id cfm_GFjiDuUS87hL, expires 12:05:56 UTC). |

### Agent

**Summary**  
A rollback of the `inference-api` deployment to its previous revision has been prepared and now awaits human approval.

**Verified facts**  
- The rollback request was created successfully and returned a confirmation ID `cfm_GFjiDuUS87hL` [ E1 ].  
- The rollback targets the `mlops-agent` namespace and the `inference-api` deployment, with no explicit revision supplied (so it will revert to the most recent prior revision).  

**Uncertainties**  
- The exact revision number that will be restored is not known until the rollback is approved and executed.  

**Recommended actions**  
- **Approve the pending rollback** using the provided confirmation ID (`cfm_GFjiDuUS87hL`). Once approved, the deployment will revert to the prior revision, restoring the previous working image tag.  
- After the rollback completes, run `k8s_diagnose_deployment` again to confirm the deployment is healthy and serving the correct champion model version.

_Grounding: verified (0 values checked) · systems: kubernetes · request `req_20a78381cf80`_

> **Operator APPROVED** `cfm_GFjiDuUS87hL` - k8s_rollback_deployment {"namespace": "mlops-agent", "deployment": "inference-api", "to_revision": null} -> **ok**: Rolled back mlops-agent/inference-api from revision 7 to the template of revision 6 (image mlops-inference:model-v2).

## User

Is the inference API healthy now, and which model version is it serving?

### Tool calls (llm mode, 56.3s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_diagnose_deployment` | ok | 134 | inference-api is HEALTHY: 2/2 replicas ready, 2 ready endpoint(s); no problems found. |

### Agent

**Summary**  
The `inference-api` deployment is now healthy and serving model version **v2**.

**Verified facts**  
- Deployment status: **HEALTHY** – 2/2 replicas ready, both endpoints ready, no problems detected [ E1 ].  
- Image used by both pods: `mlops-inference:model‑v2` [ E1 ].  
- Deployment metadata shows `version_label: "model‑v2"` and the associated ConfigMap reports `MODEL_VERSION=2` [ E1 ].  
- The rollout revision is now `8` (the rollback applied revision 6’s template) [ E1 ].

**Uncertainties**  
- None; all relevant health and version information was obtained from the diagnosis.

_Grounding: verified (0 values checked) · systems: kubernetes · request `req_0a133a58a2a7`_
