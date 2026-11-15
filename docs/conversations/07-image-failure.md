# Failure investigation: ImagePullBackOff after a bad release

_Recorded 2026-11-15 12:44 UTC against the live stack (scenario `image-failure`)._

> **Operator ran:** `python.exe scripts/inject_failure.py bad-image`

## User

The last release seems broken. Diagnose the inference-api deployment and propose a fix.

### Tool calls (llm mode, 8.9s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_diagnose_deployment` | ok | 6084 | inference-api is DEGRADED: 2/2 replicas ready, 2 ready endpoint(s); [critical] Image cannot be pulled. |
| E2 | `k8s_rollback_deployment` | requires_confirmation | 6 | NOT EXECUTED. 'Roll back mlops-agent/inference-api to the previous revision' requires explicit human approval (confirmation id cfm__ebaI4BxPABd, expires 12:53:16 UTC). |

### Agent

**Summary**  
The inference‑api deployment is degraded because a new replica is stuck in **ImagePullBackOff** for the non‑existent image `mlops-inference:model-v404-does-not-exist`[E1].

**Verified facts**  
- Deployment `inference-api` reports **DEGRADED** status, with 2/2 replicas ready but one new replica pending due to image pull failure[E1].  
- The pending pod `inference-api-5c87784b78-76b4p` shows `ImagePullBackOff` and the associated events confirm the image cannot be pulled (repository does not exist)[E1].  
- The other two pods run the previous image `mlops-inference:model-v2` successfully[E1].  
- A rollback action to the previous revision has been prepared but awaits human approval, with confirmation id `cfm__ebaI4BxPABd`[E2].

**Uncertainties**  
- No evidence about the intended new image tag (why `model-v404-does-not-exist` was specified) – the source of the erroneous tag is unknown.  
- No current info on the model version registered as *champion* in MLflow; confirming it matches the running image would require an additional query (if needed).

**Recommended actions**  
1. **Approve the pending rollback** to restore the previous working revision of the deployment.  
   - Confirmation ID: `cfm__ebaI4BxPABd` (expires 12:53 UTC).  
2. After rollback succeeds, **verify the model version** in the MLflow Model Registry matches the image tag `model-v2` (run `mlflow_get_model_registry` if you need confirmation).  
3. Investigate the source of the bad image tag (`model-v404-does-not-exist`) – likely a mis‑configuration in the deployment manifest or CI/CD pipeline. Correct the tag to a valid image (e.g., `mlops-inference:model-v2` or the intended new version) and, once verified, you can redeploy with a new approved rollout.

_Grounding: verified (1 values checked) · systems: kubernetes · request `req_18de6ffd9054`_

## User

Roll the inference-api deployment back to the previous revision.

### Tool calls (llm mode, 1.1s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_rollback_deployment` | requires_confirmation | 4 | NOT EXECUTED. 'Roll back mlops-agent/inference-api to the previous revision' requires explicit human approval (confirmation id cfm_8wpTH8WV7vAn, expires 12:53:17 UTC). |

### Agent

**Summary**  
A rollback of the `inference-api` deployment to its previous revision has been prepared and now awaits your approval.

**Verified facts**  
- The rollback request was created successfully and is pending human confirmation, with confirmation ID **`cfm_8wpTH8WV7vAn`** and expiration at **2026‑11‑15 12:53:17 UTC**[E1].

**Uncertainties**  
- None; the rollback action is ready but not yet executed.

**Recommended actions**  
- **Approve the rollback** by confirming the pending action using the provided confirmation ID `cfm_8wpTH8WV7vAn` before it expires. This will restore the previous, known‑good revision of the `inference-api` deployment.

_Grounding: verified (0 values checked) · systems: kubernetes · request `req_9c2b61b193c7`_

> **Operator APPROVED** `cfm_8wpTH8WV7vAn` - k8s_rollback_deployment {"namespace": "mlops-agent", "deployment": "inference-api", "to_revision": null} -> **ok**: Rolled back mlops-agent/inference-api from revision 14 to the template of revision 13 (image mlops-inference:model-v2).

## User

Is the inference API healthy now, and which model version is it serving?

### Tool calls (llm mode, 63.5s)

| id | tool | status | ms | evidence |
|---|---|---|---|---|
| E1 | `k8s_get_deployed_version` | ok | 11055 | inference-api runs mlops-inference:model-v2 = model credit-risk-classifier v2 (run 2510235a, dataset v2.0), revision 15, ready 2/2. Version metadata consistent. |

### Agent

**Summary**  
Yes – the `inference-api` deployment is healthy (2 / 2 pods ready) and is serving model version **2** (`mlops-inference:model‑v2`).

**Verified facts**  
- Deployment `inference-api` reports **ready 2/2** pods, image `mlops-inference:model-v2`, rollout revision 15, and no consistency issues[E1].  
- The image tag `model‑v2` corresponds to the registered model **credit‑risk‑classifier** version 2, which was produced by MLflow run `2510235a1b05490bab5f71a3770ea793` on dataset version v2.0[E1].  

**Uncertainties**  
- None; all queried indicators are consistent.

**Recommended actions**  
- Continue monitoring the API (e.g., request health checks, track production metrics).  
- If a new model version is ready, follow the usual promotion workflow (quality‑gate → candidate alias → human‑approved deployment).

_Grounding: verified (1 values checked) · systems: kubernetes · request `req_24c2ecb79596`_
