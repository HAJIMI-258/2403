# Stage EVAL-0 Report

## Verdict

run external passive/baseline benchmark subset; do not integrate active evidence until E4A controls pass

## Compact

```json
{
  "stage": "EVAL-0",
  "internal_eval_trust_passed": 1,
  "oracle_leakage_found": 0,
  "metric_consistency_passed": 1,
  "negative_controls_summary": {
    "total": 5,
    "not_ready_count": 2,
    "not_ready": [
      {
        "control_name": "shuffled_true_historical_descriptor",
        "stage": "E4A.1b",
        "passed": "0",
        "failure_reason": "control_not_clean"
      },
      {
        "control_name": "wrong_descriptor_control",
        "stage": "E4A.1b",
        "passed": "0",
        "failure_reason": "control_not_clean"
      }
    ]
  },
  "active_evidence_integration_ready": 0,
  "external_protocol_created": 1,
  "external_adapters_created": 1,
  "external_smoke_ready": 1,
  "next_recommendation": "run external passive/baseline benchmark subset; do not integrate active evidence until E4A controls pass"
}
```
